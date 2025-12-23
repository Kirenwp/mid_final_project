import json
import random
import torch
import nltk # 导入 nltk
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from collections import Counter

# 特殊 Token 定义
SPECIALS = {"pad": "<pad>", "unk": "<unk>", "sos": "<sos>", "eos": "<eos>"}

def set_seed(seed: int):
    """固定随机种子，保证实验可复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str):
    """生成器读取 JSONL 文件，防止内存溢出"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def zh_tokenize(s: str) -> List[str]:
    """
    中文分词：使用 jieba。
    """
    s = s.strip()
    try:
        import jieba
        return [t for t in jieba.lcut(s) if t and not t.isspace()]
    except ImportError:
        print("Warning: Jieba not found. Fallback to character-level tokenization.")
        return [c for c in s if not c.isspace()]

def en_tokenize(s: str) -> List[str]:
    """
    英文分词：使用 NLTK。
    相比简单的 split()，NLTK 能更好处理标点符号（例如 "don't" -> "do", "n't"）。
    """
    s = s.strip().lower() # 同样建议先转小写
    try:
        # 尝试使用 nltk 分词
        return nltk.word_tokenize(s)
    except LookupError:
        # 如果忘记下载 punkt，尝试自动下载 (虽然不建议在循环中这样做，但为了防报错)
        print("NLTK punkt data not found. Downloading...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        return nltk.word_tokenize(s)
    except ImportError:
        # 如果没装 nltk 包
        print("Warning: NLTK not installed. Fallback to split().")
        return s.split()

class Vocab:
    """词表管理类：处理 词 <-> ID 的映射"""
    def __init__(self, stoi: Dict[str, int], itos: List[str]):
        self.stoi = stoi
        self.itos = itos
        self.pad_id = stoi[SPECIALS["pad"]]
        self.unk_id = stoi[SPECIALS["unk"]]
        self.sos_id = stoi[SPECIALS["sos"]]
        self.eos_id = stoi[SPECIALS["eos"]]

    @classmethod
    def build(cls, token_seqs: List[List[str]], min_freq: int = 1, max_size: int = 50000):
        cnt = Counter()
        for seq in token_seqs:
            cnt.update(seq)
        
        itos = [SPECIALS["pad"], SPECIALS["sos"], SPECIALS["eos"], SPECIALS["unk"]]
        
        for tok, c in cnt.most_common():
            if c < min_freq:
                continue
            if tok in itos:
                continue
            itos.append(tok)
            if len(itos) >= max_size:
                break
        
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi, itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.sos_id):
                continue
            token = self.itos[i] if 0 <= i < len(self.itos) else SPECIALS["unk"]
            out.append(token)
        return out

    def to_dict(self):
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, data):
        return cls(data["stoi"], data["itos"])

class JsonlTranslationDataset(Dataset):
    def __init__(self, path: str, src_field: str = "zh", tgt_field: str = "en", max_len: int = 100):
        self.data = []
        # 你的数据已经是 {"zh":..., "en":...}，默认参数即可直接运行
        print(f"Loading data from {path} (src='{src_field}', tgt='{tgt_field}')...")
        
        for ex in read_jsonl(path):
            if src_field in ex and tgt_field in ex:
                src = ex[src_field]
                tgt = ex[tgt_field]
                self.data.append((src, tgt))
                
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int = 100):
    src_ids_list = []
    tgt_in_ids_list = []
    tgt_out_ids_list = []

    for src_text, tgt_text in batch:
        # 这里会调用上面修改后的 en_tokenize
        src_tokens = zh_tokenize(src_text)[:max_len - 2]
        tgt_tokens = en_tokenize(tgt_text)[:max_len - 2]

        full_src = [SPECIALS["sos"]] + src_tokens + [SPECIALS["eos"]]
        full_tgt = [SPECIALS["sos"]] + tgt_tokens + [SPECIALS["eos"]]

        s_ids = src_vocab.encode(full_src)
        t_ids = tgt_vocab.encode(full_tgt)

        src_ids_list.append(torch.tensor(s_ids, dtype=torch.long))
        tgt_in_ids_list.append(torch.tensor(t_ids[:-1], dtype=torch.long))
        tgt_out_ids_list.append(torch.tensor(t_ids[1:], dtype=torch.long))

    def pad_seqs(seqs, pad_val):
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)

    src_tensor = pad_seqs(src_ids_list, src_vocab.pad_id)
    tgt_in_tensor = pad_seqs(tgt_in_ids_list, tgt_vocab.pad_id)
    tgt_out_tensor = pad_seqs(tgt_out_ids_list, tgt_vocab.pad_id)

    return src_tensor, tgt_in_tensor, tgt_out_tensor

def save_checkpoint(path: str, model, optimizer, config: dict, src_vocab: Vocab, tgt_vocab: Vocab):
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer else None,
        "config": config,
        "src_vocab": src_vocab.to_dict(),
        "tgt_vocab": tgt_vocab.to_dict()
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)