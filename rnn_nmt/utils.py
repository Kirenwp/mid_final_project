\
import json, re, random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch

SPECIALS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "sos": "<sos>",
    "eos": "<eos>",
}

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infer_fields(example: Dict, src_field: Optional[str], tgt_field: Optional[str]) -> Tuple[str, str]:
    """Infer src/tgt fields from a json object if not explicitly provided."""
    if src_field and tgt_field:
        return src_field, tgt_field

    keys = list(example.keys())
    # Common conventions
    candidates = [
        ("zh", "en"),
        ("cn", "en"),
        ("ch", "en"),
        ("src", "tgt"),
        ("source", "target"),
        ("chinese", "english"),
        ("zh_text", "en_text"),
    ]
    for s, t in candidates:
        if s in example and t in example and isinstance(example[s], str) and isinstance(example[t], str):
            return s, t

    # Fallback: pick first two string-valued fields
    str_keys = [k for k in keys if isinstance(example.get(k), str)]
    if len(str_keys) >= 2:
        return str_keys[0], str_keys[1]

    raise ValueError(f"Could not infer src/tgt fields from example keys={keys}. "
                     f"Pass --src_field and --tgt_field explicitly.")

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

class Vocab:
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def build(self, tokenized_texts: List[List[str]]):
        counter = Counter()
        for toks in tokenized_texts:
            counter.update(toks)

        # specials first
        self.itos = [SPECIALS["pad"], SPECIALS["unk"], SPECIALS["sos"], SPECIALS["eos"]]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        # sort by freq then alpha for determinism
        items = sorted([(t, c) for t, c in counter.items() if c >= self.min_freq],
                       key=lambda x: (-x[1], x[0]))

        if self.max_size is not None:
            items = items[: max(0, self.max_size - len(self.itos))]

        for tok, _ in items:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    @property
    def pad_idx(self): return self.stoi[SPECIALS["pad"]]
    @property
    def unk_idx(self): return self.stoi[SPECIALS["unk"]]
    @property
    def sos_idx(self): return self.stoi[SPECIALS["sos"]]
    @property
    def eos_idx(self): return self.stoi[SPECIALS["eos"]]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> List[str]:
        toks = []
        for i in ids:
            if stop_at_eos and i == self.eos_idx:
                break
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else SPECIALS["unk"])
        return toks

def save_checkpoint(path: str, model, optimizer, config: dict, src_vocab: Vocab, tgt_vocab: Vocab, extra: Optional[dict]=None):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "src_vocab": {"stoi": src_vocab.stoi, "itos": src_vocab.itos, "min_freq": src_vocab.min_freq, "max_size": src_vocab.max_size},
        "tgt_vocab": {"stoi": tgt_vocab.stoi, "itos": tgt_vocab.itos, "min_freq": tgt_vocab.min_freq, "max_size": tgt_vocab.max_size},
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)

def load_checkpoint(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt

def safe_detokenize(tokens: List[str]) -> str:
    """A simple detokenizer for BLEU; leave spaces for English-like targets."""
    return " ".join(tokens)

def ids_to_sentence(ids: List[int], vocab: Vocab) -> str:
    toks = vocab.decode(ids, stop_at_eos=True)
    # remove specials if still present
    toks = [t for t in toks if t not in (SPECIALS["sos"], SPECIALS["pad"])]
    return safe_detokenize(toks)
