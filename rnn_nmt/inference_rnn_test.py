import argparse, json, os
import torch
from tqdm import tqdm

from utils import load_checkpoint, Vocab, SPECIALS, ids_to_sentence

from models import Encoder, Attention, Decoder, Seq2Seq

try:
    import jieba
except Exception:
    jieba = None

def zh_tokenize(s: str):
    if jieba is None:
        return list(s.strip())
    return [t for t in jieba.lcut(s.strip()) if t and not t.isspace()]

@torch.no_grad()
def greedy_decode(model, src_ids, max_len, sos_idx, eos_idx):
    model.eval()
    src = torch.tensor(src_ids, dtype=torch.long, device=model.device).unsqueeze(1)  # [src_len, 1]
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
def beam_decode(model, src_ids, max_len, sos_idx, eos_idx, beam_size=5):
    """Simple beam search for your Seq2Seq decoder API. Returns best token ids (no eos)."""
    model.eval()
    device = model.device

    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)  # [src_len, 1]
    enc_outputs, hidden = model.encoder(src)
    src_mask = model.make_src_mask(src)

    # beams: (tokens, hidden, logprob, ended)
    beams = [([sos_idx], hidden, 0.0, False)]

    for _ in range(max_len):
        candidates = []
        for toks, h, lp, ended in beams:
            if ended:
                candidates.append((toks, h, lp, True))
                continue

            input_tok = torch.tensor([toks[-1]], device=device)
            pred, h2, _ = model.decoder(input_tok, h, enc_outputs, src_mask=src_mask)  # pred: [1, V]
            logp = torch.log_softmax(pred, dim=-1).squeeze(0)  # [V]

            topk_logp, topk_ids = torch.topk(logp, k=beam_size)
            for add_lp, wid in zip(topk_logp.tolist(), topk_ids.tolist()):
                new_toks = toks + [wid]
                new_lp = lp + add_lp
                new_ended = (wid == eos_idx)
                candidates.append((new_toks, h2, new_lp, new_ended))

        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_size]

        if all(b[-1] for b in beams):
            break

    finished = [b for b in beams if b[-1]]
    best = max(finished, key=lambda x: x[2]) if finished else max(beams, key=lambda x: x[2])

    out = best[0][1:]  # drop sos
    if out and out[-1] == eos_idx:
        out = out[:-1]
    return out

def compute_bleu(hyps, refs):
    # prefer sacrebleu
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(hyps, [refs]).score
    except Exception:
        # fallback nltk
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            hyp_tok = [h.split() for h in hyps]
            ref_tok = [[r.split()] for r in refs]
            return corpus_bleu(ref_tok, hyp_tok, smoothing_function=SmoothingFunction().method1) * 100.0
        except Exception:
            return float("nan")

def rebuild_vocabs_from_ckpt(ckpt):
    # ckpt["src_vocab"] and ckpt["tgt_vocab"] are dicts in your project
    sv = ckpt["src_vocab"]
    tv = ckpt["tgt_vocab"]

    src_vocab = Vocab(min_freq=sv.get("min_freq", 1), max_size=sv.get("max_size", None))
    tgt_vocab = Vocab(min_freq=tv.get("min_freq", 1), max_size=tv.get("max_size", None))

    src_vocab.stoi = sv["stoi"]; src_vocab.itos = sv["itos"]
    tgt_vocab.stoi = tv["stoi"]; tgt_vocab.itos = tv["itos"]

    # ensure pad/sos/eos indices exist (your Vocab likely defines these)
    # If your Vocab computes them from SPECIALS, these fields should be ready.
    return src_vocab, tgt_vocab

def build_model_from_ckpt(cfg, src_vocab, tgt_vocab, device):
    enc = Encoder(
        len(src_vocab.itos),
        cfg["emb_dim"],
        cfg["hid_dim"],
        cfg["n_layers"],
        cfg["dropout"],
        rnn_type=cfg["rnn_type"],
    )
    attn = Attention(cfg["hid_dim"], attn_type=cfg["attn_type"])
    dec = Decoder(
        len(tgt_vocab.itos),
        cfg["emb_dim"],
        cfg["hid_dim"],
        cfg["n_layers"],
        cfg["dropout"],
        attn,
        rnn_type=cfg["rnn_type"],
    )
    model = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--out", default=None, help="save hypotheses to a txt file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = ckpt["config"]

    src_vocab, tgt_vocab = rebuild_vocabs_from_ckpt(ckpt)

    model = build_model_from_ckpt(cfg, src_vocab, tgt_vocab, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    hyps, refs = [], []

    out_f = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")

    with open(args.test, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Test decoding={args.decode}"):
            ex = json.loads(line)
            zh = ex["zh"]
            en = ex["en"]

            src_toks = [SPECIALS["sos"]] + zh_tokenize(zh) + [SPECIALS["eos"]]
            src_ids = src_vocab.encode(src_toks)

            if args.decode == "greedy":
                out_ids = greedy_decode(model, src_ids, args.max_len, tgt_vocab.sos_idx, tgt_vocab.eos_idx)
            else:
                out_ids = beam_decode(model, src_ids, args.max_len, tgt_vocab.sos_idx, tgt_vocab.eos_idx, args.beam_size)

            hyp = ids_to_sentence(out_ids, tgt_vocab)
            hyps.append(hyp)
            refs.append(en)

            if out_f:
                out_f.write(hyp + "\n")

    if out_f:
        out_f.close()

    bleu = compute_bleu(hyps, refs)
    print("=" * 70)
    print(f"CKPT: {args.ckpt}")
    print(f"Decode: {args.decode} | beam_size={args.beam_size if args.decode=='beam' else '-'} | max_len={args.max_len}")
    print(f"TEST BLEU: {bleu:.2f}")
    if args.out:
        print(f"Saved hypotheses to: {args.out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
