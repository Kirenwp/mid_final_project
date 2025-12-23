\
import argparse, os, time, math
from dataclasses import asdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    import jieba
except Exception:
    jieba = None

try:
    import nltk
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None
    word_tokenize = None

from utils import read_jsonl, infer_fields, Vocab, SPECIALS, set_seed, save_checkpoint, ids_to_sentence
from models import Encoder, Attention, Decoder, Seq2Seq

def zh_tokenize(s: str) -> List[str]:
    if jieba is None:
        # fallback: character-level
        return list(s.strip())
    return [t for t in jieba.lcut(s.strip()) if t and not t.isspace()]

def en_tokenize(s: str) -> List[str]:
    s = s.strip()
    if word_tokenize is not None:
        return word_tokenize(s)
    return s.split()

class TranslationDataset(Dataset):
    def __init__(self, jsonl_path: str, src_field: Optional[str], tgt_field: Optional[str],
                 src_tok, tgt_tok, src_vocab: Optional[Vocab]=None, tgt_vocab: Optional[Vocab]=None,
                 max_len: int = 100):
        self.pairs = []
        it = read_jsonl(jsonl_path)
        first = next(iter(it), None)
        if first is None:
            raise ValueError(f"Empty file: {jsonl_path}")
        # reload generator
        examples = [first] + list(read_jsonl(jsonl_path))
        sf, tf = infer_fields(first, src_field, tgt_field)

        for ex in examples:
            src = ex[sf]; tgt = ex[tf]
            if not isinstance(src, str) or not isinstance(tgt, str): 
                continue
            src_toks = [SPECIALS["sos"]] + src_tok(src)[:max_len] + [SPECIALS["eos"]]
            tgt_toks = [SPECIALS["sos"]] + tgt_tok(tgt)[:max_len] + [SPECIALS["eos"]]
            self.pairs.append((src_toks, tgt_toks))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def build_vocabs(train_path: str, src_field: Optional[str], tgt_field: Optional[str],
                 min_freq: int, max_vocab: Optional[int], max_len: int):
    ds = TranslationDataset(train_path, src_field, tgt_field, zh_tokenize, en_tokenize, max_len=max_len)
    src_vocab = Vocab(min_freq=min_freq, max_size=max_vocab)
    tgt_vocab = Vocab(min_freq=min_freq, max_size=max_vocab)
    src_vocab.build([p[0] for p in ds.pairs])
    tgt_vocab.build([p[1] for p in ds.pairs])
    return src_vocab, tgt_vocab

def collate_fn(batch, src_vocab: Vocab, tgt_vocab: Vocab):
    src_ids, tgt_ids = [], []
    for src_toks, tgt_toks in batch:
        src_ids.append(torch.tensor(src_vocab.encode(src_toks), dtype=torch.long))
        tgt_ids.append(torch.tensor(tgt_vocab.encode(tgt_toks), dtype=torch.long))
    src_pad = pad_sequence(src_ids, padding_value=src_vocab.pad_idx)  # [src_len, batch]
    tgt_pad = pad_sequence(tgt_ids, padding_value=tgt_vocab.pad_idx)  # [tgt_len, batch]
    return src_pad, tgt_pad

def epoch_time(start, end):
    s = int(end - start)
    return s // 60, s % 60

def train_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(model.device), tgt.to(model.device)
        optimizer.zero_grad()
        out = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        # out: [tgt_len, batch, vocab]
        out_dim = out.shape[-1]
        out = out[1:].reshape(-1, out_dim)
        tgt_gold = tgt[1:].reshape(-1)
        loss = criterion(out, tgt_gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(model.device), tgt.to(model.device)
        out = model(src, tgt, teacher_forcing_ratio=0.0)  # Free running for eval loss
        out_dim = out.shape[-1]
        out = out[1:].reshape(-1, out_dim)
        tgt_gold = tgt[1:].reshape(-1)
        loss = criterion(out, tgt_gold)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def greedy_decode(model, src_tensor, max_len: int, sos_idx: int, eos_idx: int):
    model.eval()
    src = src_tensor.to(model.device)  # [src_len, 1]
    enc_outputs, hidden = model.encoder(src)
    src_mask = model.make_src_mask(src)
    input_tok = torch.tensor([sos_idx], device=model.device)
    out_ids = []
    for _ in range(max_len):
        pred, hidden, _ = model.decoder(input_tok, hidden, enc_outputs, src_mask=src_mask)
        next_id = int(pred.argmax(1).item())
        if next_id == eos_idx:
            break
        out_ids.append(next_id)
        input_tok = torch.tensor([next_id], device=model.device)
    return out_ids

@torch.no_grad()
def beam_search_decode(model, src_tensor, max_len: int, sos_idx: int, eos_idx: int, beam_size: int = 5, length_norm_alpha: float = 0.6):
    """
    Simple beam search for batch=1.
    length_norm_alpha: 0 disables length norm; typical 0.6.
    """
    model.eval()
    src = src_tensor.to(model.device)  # [src_len, 1]
    enc_outputs, hidden0 = model.encoder(src)
    src_mask = model.make_src_mask(src)

    # Each beam: (tokens, hidden, logp, ended)
    beams = [([], hidden0, 0.0, False)]

    for step in range(max_len):
        new_beams = []
        for toks, h, logp, ended in beams:
            if ended:
                new_beams.append((toks, h, logp, True))
                continue
            input_tok = torch.tensor([sos_idx if step == 0 else toks[-1]], device=model.device)
            pred, h2, _ = model.decoder(input_tok, h, enc_outputs, src_mask=src_mask)
            log_probs = torch.log_softmax(pred, dim=1).squeeze(0)  # [vocab]
            topk = torch.topk(log_probs, k=beam_size)
            for next_id, next_lp in zip(topk.indices.tolist(), topk.values.tolist()):
                ended2 = (next_id == eos_idx)
                new_toks = toks if ended2 else toks + [next_id]
                new_beams.append((new_toks, h2, logp + next_lp, ended2))

        # prune
        def score(b):
            toks, _, logp, ended = b
            L = max(1, len(toks))
            if length_norm_alpha <= 0:
                return logp
            # GNMT length norm
            norm = ((5 + L) / 6) ** length_norm_alpha
            return logp / norm

        new_beams.sort(key=score, reverse=True)
        beams = new_beams[:beam_size]

        # If all ended, stop early
        if all(b[3] for b in beams):
            break

        def final_score(b):
            toks, _, logp, _ = b
            L = max(1, len(toks))
            if length_norm_alpha <= 0:
                return logp
            norm = ((5 + L) / 6) ** length_norm_alpha
            return logp / norm

    best = max(beams, key=final_score)
    return best[0]
@torch.no_grad()
def compute_bleu(model, data_loader, tgt_vocab: Vocab, decoding: str = "greedy", beam_size: int = 5, max_len: int = 100):
    # We compute corpus BLEU using sacrebleu if available; fallback to nltk.
    try:
        import sacrebleu
        use_sacrebleu = True
    except Exception:
        use_sacrebleu = False
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            smooth = SmoothingFunction().method3
        except Exception:
            corpus_bleu = None

    hyps, refs = [], []
    for src, tgt in data_loader:
        # decode sentence-by-sentence (beam needs batch=1)
        # iterate over batch dimension
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        batch = src.shape[1]
        for b in range(batch):
            src_b = src[:, b:b+1]
            if decoding == "beam":
                out_ids = beam_search_decode(model, src_b, max_len=max_len, sos_idx=tgt_vocab.sos_idx, eos_idx=tgt_vocab.eos_idx, beam_size=beam_size)
            else:
                out_ids = greedy_decode(model, src_b, max_len=max_len, sos_idx=tgt_vocab.sos_idx, eos_idx=tgt_vocab.eos_idx)

            hyp = ids_to_sentence(out_ids, tgt_vocab)
            # reference is tgt sentence without <sos>, stop at <eos>
            tgt_ids = tgt[:, b].tolist()[1:]  # drop <sos>
            ref = ids_to_sentence(tgt_ids, tgt_vocab)
            hyps.append(hyp)
            refs.append(ref)

    if use_sacrebleu:
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        return bleu
    if corpus_bleu is not None:
        # nltk expects tokenized
        hyp_tok = [h.split() for h in hyps]
        ref_tok = [[r.split()] for r in refs]
        return corpus_bleu(ref_tok, hyp_tok, smoothing_function=smooth) * 100.0
    return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--valid", type=str, required=True)
    ap.add_argument("--test", type=str, required=False)
    ap.add_argument("--src_field", type=str, default=None)
    ap.add_argument("--tgt_field", type=str, default=None)

    ap.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--attn_type", type=str, default="dot", choices=["dot", "general", "additive"])
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hid_dim", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=100)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--decode_valid", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--run_name", type=str, default="rnn_nmt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NLTK tokenizer might need punkt
    if nltk is not None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    os.makedirs(args.save_dir, exist_ok=True)

    src_vocab, tgt_vocab = build_vocabs(args.train, args.src_field, args.tgt_field,
                                       min_freq=args.min_freq, max_vocab=args.max_vocab, max_len=args.max_len)

    train_ds = TranslationDataset(args.train, args.src_field, args.tgt_field, zh_tokenize, en_tokenize, max_len=args.max_len)
    valid_ds = TranslationDataset(args.valid, args.src_field, args.tgt_field, zh_tokenize, en_tokenize, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))

    enc = Encoder(len(src_vocab.itos), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type)
    attn = Attention(args.hid_dim, attn_type=args.attn_type)
    dec = Decoder(len(tgt_vocab.itos), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, attn, rnn_type=args.rnn_type)
    model = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    best_valid = float("inf")
    best_valid_bleu = -1e9   # 放在 epoch 循环外初始化

    best_path = os.path.join(args.save_dir, f"{args.run_name}.pt")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.clip, args.teacher_forcing_ratio)
        valid_loss = eval_epoch(model, valid_loader, criterion)
        mins, secs = epoch_time(start, time.time())

    
        # epoch 循环里：
        # 仍然可以算 valid_loss 但不拿来当保存标准（可留作日志）
        valid_loss = eval_epoch(model, valid_loader, criterion)

        # 用 BLEU 当唯一标准
        valid_bleu = compute_bleu(
            model, valid_loader, tgt_vocab,
            decoding=args.decode_valid,
            beam_size=args.beam_size,
            max_len=args.max_len
        )

        improved = valid_bleu > best_valid_bleu
        if improved:
            best_valid_bleu = valid_bleu
            config = vars(args)
            save_checkpoint(
                best_path, model, optimizer, config, src_vocab, tgt_vocab,
                extra={"best_valid_bleu": best_valid_bleu, "valid_loss_at_best": valid_loss}
            )
        print(f"Epoch {epoch:02d} |time {mins}m{secs:02d}s| train {train_loss:.3f} | valid_loss {valid_loss:.3f} "
        f"| valid BLEU({args.decode_valid}) {valid_bleu:.2f} | saved {('YES' if improved else 'no')}")

    if args.test:
        # load best for test eval
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_ds = TranslationDataset(args.test, args.src_field, args.tgt_field, zh_tokenize, en_tokenize, max_len=args.max_len)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))
        test_bleu_g = compute_bleu(model, test_loader, tgt_vocab, decoding="greedy", beam_size=args.beam_size, max_len=args.max_len)
        test_bleu_b = compute_bleu(model, test_loader, tgt_vocab, decoding="beam", beam_size=args.beam_size, max_len=args.max_len)
        print(f"Test BLEU greedy={test_bleu_g:.2f} | beam(k={args.beam_size})={test_bleu_b:.2f}")

if __name__ == "__main__":
    main()
  