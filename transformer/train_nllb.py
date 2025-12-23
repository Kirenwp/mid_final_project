import argparse, json, os, time
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import sacrebleu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ZhEnJsonl(Dataset):
    def __init__(self, filename: str):
        self.data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {"src": ex["zh"], "tgt": ex["en"]}


def build_collate(tokenizer, max_src_len: int, max_tgt_len: int):
    pad_id = tokenizer.pad_token_id

    def collate_fn(examples: List[Dict]):
        src_texts = [e["src"] for e in examples]
        tgt_texts = [e["tgt"] for e in examples]

        model_inputs = tokenizer(
            src_texts,
            max_length=max_src_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # NLLB：目标端建议用 tokenizer(text_target=...)，并且 tokenizer.tgt_lang 已设置
        tgt = tokenizer(
            text_target=tgt_texts,
            max_length=max_tgt_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].clone()
        labels[labels == pad_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    return collate_fn


@torch.no_grad()
def eval_bleu(model, loader, tokenizer, device, gen_max_new_tokens=128, num_beams=4):
    model.eval()
    preds, refs = [], []

    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            max_new_tokens=gen_max_new_tokens,
            num_beams=num_beams,
            forced_bos_token_id=model.config.forced_bos_token_id,  # ✅ NLLB关键
        )
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    return float(sacrebleu.corpus_bleu(preds, [refs], lowercase=True, force=True).score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--valid_file", type=str, required=True)
    ap.add_argument("--exp_name", type=str, default="nllb_zh_en")
    ap.add_argument("--save_dir", type=str, default="checkpoints_nllb")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=256)
    ap.add_argument("--gen_max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)

    # NLLB 语言码（中文简体/英文）
    ap.add_argument("--src_lang", type=str, default="zho_Hans")
    ap.add_argument("--tgt_lang", type=str, default="eng_Latn")

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    # ✅ NLLB 关键：设置源/目标语言，以及生成时强制目标语言 BOS
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang
    def get_lang_id(tok, lang_code: str) -> int:
        lang_id = tok.convert_tokens_to_ids(lang_code)
        if lang_id == tok.unk_token_id:
            # 有些版本语言码可能不在“普通词表”，但会在 added vocab 里
            vocab = tok.get_vocab()
            lang_id = vocab.get(lang_code, tok.unk_token_id)
        if lang_id == tok.unk_token_id:
            raise ValueError(
                f"Cannot find language token id for {lang_code}. "
                f"Try checking tokenizer.special_tokens_map / tokenizer.get_vocab()."
            )
        return lang_id

    tokenizer.src_lang = args.src_lang if hasattr(tokenizer, "src_lang") else None
    tokenizer.tgt_lang = args.tgt_lang if hasattr(tokenizer, "tgt_lang") else None

    model.config.forced_bos_token_id = get_lang_id(tokenizer, args.tgt_lang)
    train_ds = ZhEnJsonl(args.train_file)
    valid_ds = ZhEnJsonl(args.valid_file)

    collate_fn = build_collate(tokenizer, args.max_src_len, args.max_tgt_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_bleu = -1.0
    best_dir = os.path.join(args.save_dir, f"{args.exp_name}_best")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, start = 0.0, time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(**batch).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)             # ✅ AMP裁剪前必须
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        valid_bleu = eval_bleu(
            model, valid_loader, tokenizer, device,
            gen_max_new_tokens=args.gen_max_new_tokens,
            num_beams=args.num_beams
        )
        print(f"Epoch {epoch} | Loss {total_loss/len(train_loader):.4f} | BLEU {valid_bleu:.2f} | Time {int(time.time()-start)}s")

        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved best to: {best_dir} (BLEU={best_bleu:.2f})")


if __name__ == "__main__":
    main()