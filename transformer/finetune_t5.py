import argparse, json, os, time
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5TranslationDataset(Dataset):
    def __init__(self, filename, tokenizer, max_src_len=128, max_tgt_len=128, prefix="translate Chinese to English: "):
        self.data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.prefix = prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        src = self.prefix + ex["zh"]
        tgt = ex["en"]

        # 编码优化：一次性处理 padding 和 truncation
        src_enc = self.tok(src, max_length=self.max_src_len, padding="max_length", truncation=True, return_tensors="pt")
        tgt_enc = self.tok(tgt, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt")

        labels = tgt_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tok.pad_token_id] = -100 # 忽略 Loss 计算中的 Pad

        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }

@torch.no_grad()
def eval_bleu(model, loader, tokenizer, device, gen_max_len=128, num_beams=4):
    model.eval()
    preds, refs = [], []
    for batch in tqdm(loader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        # 使用 generate 方法进行自回归生成
        gen_ids = model.generate(input_ids=input_ids, max_length=gen_max_len, num_beams=num_beams)
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
        
        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    return float(sacrebleu.corpus_bleu(preds, [refs], force=True).score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="t5-small")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--valid_file", type=str, required=True)
    ap.add_argument("--exp_name", type=str, default="t5_fast")
    ap.add_argument("--save_dir", type=str, default="checkpoints_t5")
    ap.add_argument("--batch_size", type=int, default=32) # 建议增大以加速
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4) # 微调学习率建议略高
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    # 优化点 1: 使用 Fast Tokenizer (Rust 实现)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    # 优化点 2: Linux 下启用模型编译 (Torch 2.0+)
    if os.name != 'nt' and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    train_ds = T5TranslationDataset(args.train_file, tokenizer)
    valid_ds = T5TranslationDataset(args.valid_file, tokenizer)

    # 优化点 3: 增加 persistent_workers 减少 Epoch 切换开销
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # 混合精度训练

    best_bleu = -1.0
    best_dir = os.path.join(args.save_dir, f"{args.exp_name}_best")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, start = 0.0, time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device)).loss
            
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        valid_bleu = eval_bleu(model, valid_loader, tokenizer, device)
        print(f"Epoch {epoch} | Loss {total_loss/len(train_loader):.4f} | BLEU {valid_bleu:.2f} | Time {int(time.time()-start)}s")

        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

if __name__ == "__main__":
    main()
