\
import argparse
import torch

from utils import load_checkpoint, Vocab, SPECIALS, ids_to_sentence
from models import Encoder, Attention, Decoder, Seq2Seq

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

def zh_tokenize(s: str):
    if jieba is None:
        return list(s.strip())
    return [t for t in jieba.lcut(s.strip()) if t and not t.isspace()]

def en_tokenize(s: str):
    s = s.strip()
    if word_tokenize is not None:
        return word_tokenize(s)
    return s.split()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to .pt checkpoint saved by train_rnn_nmt.py")
    ap.add_argument("--text", type=str, required=True, help="Chinese source sentence")
    ap.add_argument("--max_len", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = ckpt["config"]

    # rebuild vocabs
    src_vocab = Vocab(min_freq=ckpt["src_vocab"]["min_freq"], max_size=ckpt["src_vocab"]["max_size"])
    tgt_vocab = Vocab(min_freq=ckpt["tgt_vocab"]["min_freq"], max_size=ckpt["tgt_vocab"]["max_size"])
    src_vocab.stoi = ckpt["src_vocab"]["stoi"]; src_vocab.itos = ckpt["src_vocab"]["itos"]
    tgt_vocab.stoi = ckpt["tgt_vocab"]["stoi"]; tgt_vocab.itos = ckpt["tgt_vocab"]["itos"]

    # rebuild model
    enc = Encoder(len(src_vocab.itos), cfg["emb_dim"], cfg["hid_dim"], cfg["n_layers"], cfg["dropout"], rnn_type=cfg["rnn_type"])
    attn = Attention(cfg["hid_dim"], attn_type=cfg["attn_type"])
    dec = Decoder(len(tgt_vocab.itos), cfg["emb_dim"], cfg["hid_dim"], cfg["n_layers"], cfg["dropout"], attn, rnn_type=cfg["rnn_type"])
    model = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)
    model.load_state_dict(ckpt["model_state"])

    toks = [SPECIALS["sos"]] + zh_tokenize(args.text) + [SPECIALS["eos"]]
    src_ids = src_vocab.encode(toks)
    out_ids = greedy_decode(model, src_ids, max_len=args.max_len, sos_idx=tgt_vocab.sos_idx, eos_idx=tgt_vocab.eos_idx)
    print(ids_to_sentence(out_ids, tgt_vocab))

if __name__ == "__main__":
    main()
