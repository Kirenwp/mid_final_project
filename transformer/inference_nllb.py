import argparse, json
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU


def get_lang_id(tok, lang_code: str) -> int:
    # 兼容不同 transformers/tokenizer 版本：不要用 lang_code_to_id
    lang_id = tok.convert_tokens_to_ids(lang_code)
    if hasattr(tok, "unk_token_id") and lang_id == tok.unk_token_id:
        lang_id = tok.get_vocab().get(lang_code, tok.unk_token_id)
    if hasattr(tok, "unk_token_id") and lang_id == tok.unk_token_id:
        raise ValueError(f"Language token '{lang_code}' not found in vocab.")
    return int(lang_id)


class TestPairs(Dataset):
    def __init__(self, filename: str, task: str):
        self.items: List[Tuple[str, str, str, str]] = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                zh, en = ex["zh"], ex["en"]
                if task in ("zh2en", "both"):
                    self.items.append(("zho_Hans", "eng_Latn", zh, en))
                if task in ("en2zh", "both"):
                    self.items.append(("eng_Latn", "zho_Hans", en, zh))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        src_lang, tgt_lang, src, ref = self.items[idx]
        return {"src_lang": src_lang, "tgt_lang": tgt_lang, "src": src, "ref": ref}


def build_collate(tok, max_src_len: int):
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    def collate_fn(batch):
        # 这里假设一个 batch 内方向一致（下面我们会按方向分别跑）
        src_lang = batch[0]["src_lang"]
        tgt_lang = batch[0]["tgt_lang"]
        src_lang_id = get_lang_id(tok, src_lang)
        tgt_lang_id = get_lang_id(tok, tgt_lang)

        src_texts = [x["src"] for x in batch]
        refs = [x["ref"] for x in batch]

        # 用 tokenizer 正常做 padding/truncation
        # 有些版本设置 tok.src_lang 会自动加语言 token，有些不会；所以我们后面会检查补齐
        if hasattr(tok, "src_lang"):
            tok.src_lang = src_lang

        enc = tok(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_src_len,
        )
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", (input_ids != pad_id).long())

        # 如果 tokenizer 没自动加 src_lang token，则手动补一个： [src_lang] + ids (并补 eos)
        # 判断规则：每条的第一个 token 是否等于 src_lang_id
        if input_ids.size(1) == 0 or (input_ids[:, 0] != src_lang_id).any():
            # 去掉左侧 padding 的影响：我们直接重构，不在意原 padding
            # 先把每条序列去掉 pad，再拼 [src_lang] ... [eos]
            seqs = []
            for i in range(input_ids.size(0)):
                ids = input_ids[i].tolist()
                ids = [t for t in ids if t != pad_id]
                # 保证以 src_lang 开头，并以 eos 结尾
                if len(ids) == 0 or ids[0] != src_lang_id:
                    ids = [src_lang_id] + ids
                if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
                    ids = ids + [eos_id]
                # 截断
                ids = ids[:max_src_len]
                seqs.append(ids)

            max_len = max(len(s) for s in seqs)
            new_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            new_attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                new_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                new_attn[i, :len(s)] = 1
            input_ids, attn = new_ids, new_attn

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "refs": refs,
            "forced_bos_token_id": tgt_lang_id,
            "tgt_lang": tgt_lang,
        }

    return collate_fn


@torch.no_grad()
def eval_one_direction(model, tok, loader, device, max_new_tokens: int, num_beams: int, bleu: BLEU):
    preds, refs = [], []
    model.eval()
    for batch in tqdm(loader, desc="Infer", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        forced_bos = int(batch["forced_bos_token_id"])

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos,
        )
        pred_text = tok.batch_decode(gen, skip_special_tokens=True)
        preds.extend(pred_text)
        refs.extend(batch["refs"])

    return float(bleu.corpus_score(preds, [refs]).score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="e.g. checkpoints_nllb/nllb_experiment_best")
    ap.add_argument("--test_file", type=str, required=True, help="jsonl with keys zh/en")
    ap.add_argument("--task", type=str, default="both", choices=["zh2en", "en2zh", "both"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt).to(device)

    # 分方向分别评测，避免一个 batch 混方向导致 forced_bos 不一致
    results = {}

    for one_task in (["zh2en", "en2zh"] if args.task == "both" else [args.task]):
        ds = TestPairs(args.test_file, task=one_task)
        collate_fn = build_collate(tok, args.max_src_len)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
        )

        # 英文 BLEU：13a + lowercase；中文 BLEU：tokenize="zh"
        if one_task == "zh2en":
            bleu = BLEU(tokenize="13a", lowercase=True)
        else:
            bleu = BLEU(tokenize="zh", lowercase=False)

        score = eval_one_direction(
            model, tok, loader, device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            bleu=bleu
        )
        results[one_task] = score

    if "zh2en" in results:
        print(f"TEST zh->en BLEU: {results['zh2en']:.2f}")
    if "en2zh" in results:
        print(f"TEST en->zh BLEU: {results['en2zh']:.2f}")
    if len(results) == 2:
        print(f"TEST avg BLEU: {(results['zh2en'] + results['en2zh'])/2:.2f}")


if __name__ == "__main__":
    main()