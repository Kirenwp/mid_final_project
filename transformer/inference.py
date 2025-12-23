import torch
import json
import os
import argparse
from tqdm import tqdm
import math
import sys
import sacrebleu

# 导入你的 utils 和模型
from utils import (
    Vocab, zh_tokenize, en_tokenize, SPECIALS
)
from model_rnn import EncoderRNN, AttnDecoderRNN
from model_transformer_relative import TransformerModel

# ==========================================
# Beam Search Classes & Functions
# ==========================================
class BeamHypothesis:
    def __init__(self, sequence, log_prob, hidden=None):
        self.sequence = sequence
        self.log_prob = log_prob
        self.hidden = hidden

    @property
    def last_token(self):
        return self.sequence[-1]

    def score(self):
        alpha = 0.7
        return self.log_prob / (len(self.sequence) ** alpha)

def beam_search_rnn(model, src_tensor, beam_width, max_len, sos_id, eos_id, device):
    encoder, decoder = model
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        enc_out, hidden = encoder(src_tensor)
        start_hyp = BeamHypothesis([sos_id], 0.0, hidden)
        beams = [start_hyp]
        completed_hyps = []
        for _ in range(max_len):
            candidates = []
            for hyp in beams:
                if hyp.last_token == eos_id:
                    completed_hyps.append(hyp)
                    continue
                dec_input = torch.tensor([[hyp.last_token]], device=device)
                output, new_hidden, _ = decoder(dec_input, hyp.hidden, enc_out)
                log_probs = torch.log_softmax(output.squeeze(0).squeeze(0), dim=0)
                topk_probs, topk_ids = log_probs.topk(beam_width)
                for k in range(beam_width):
                    word_id = topk_ids[k].item()
                    score = topk_probs[k].item()
                    new_hyp = BeamHypothesis(hyp.sequence + [word_id], hyp.log_prob + score, new_hidden)
                    candidates.append(new_hyp)
            if not candidates: break
            candidates.sort(key=lambda x: x.score(), reverse=True)
            beams = candidates[:beam_width]
        completed_hyps.extend(beams)
        best_hyp = max(completed_hyps, key=lambda x: x.score())
        return best_hyp.sequence[1:]

def beam_search_transformer(model, src_tensor, beam_width, max_len, sos_id, eos_id, pad_id, device):
    model.eval()
    with torch.no_grad():
        src_mask = (src_tensor == pad_id)
        memory = model.transformer_encoder(
            model.pos_encoder(model.src_embedding(src_tensor) * math.sqrt(model.d_model)),
            src_key_padding_mask=src_mask
        )
        start_hyp = BeamHypothesis([sos_id], 0.0)
        beams = [start_hyp]
        completed_hyps = []
        for _ in range(max_len):
            candidates = []
            for hyp in beams:
                if hyp.last_token == eos_id:
                    completed_hyps.append(hyp)
                    continue
                ys = torch.tensor([hyp.sequence], device=device)
                tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
                out = model.transformer_decoder(
                    model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model)),
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_mask
                )
                prob = model.out(out[:, -1])
                log_probs = torch.log_softmax(prob.squeeze(0), dim=0)
                topk_probs, topk_ids = log_probs.topk(beam_width)
                for k in range(beam_width):
                    word_id = topk_ids[k].item()
                    score = topk_probs[k].item()
                    new_hyp = BeamHypothesis(hyp.sequence + [word_id], hyp.log_prob + score, None)
                    candidates.append(new_hyp)
            if not candidates: break
            candidates.sort(key=lambda x: x.score(), reverse=True)
            beams = candidates[:beam_width]
        completed_hyps.extend(beams)
        best_hyp = max(completed_hyps, key=lambda x: x.score())
        return best_hyp.sequence[1:]

# ==========================================
# Main Logic
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found!")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    try:
        src_vocab = Vocab.from_dict(checkpoint['src_vocab'])
        tgt_vocab = Vocab.from_dict(checkpoint['tgt_vocab'])
    except:
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        
    model_type = config.get('model_type', 'rnn')
    if model_type == 'rnn':
        encoder = EncoderRNN(len(src_vocab.itos), config['hidden_size']).to(device)
        decoder = AttnDecoderRNN(config['hidden_size'], len(tgt_vocab.itos), attn_method=config['attn_method']).to(device)
        encoder.load_state_dict(checkpoint['model_state']['encoder'])
        decoder.load_state_dict(checkpoint['model_state']['decoder'])
        model = (encoder, decoder)
    else:
        # 兼容 RoPE 版本的动态加载逻辑（如果文件名包含 rope，需确保 model_transformer 加载的是相对位置代码）
        model = TransformerModel(
            src_vocab_size=len(src_vocab.itos),
            tgt_vocab_size=len(tgt_vocab.itos),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            norm_type=config['norm_type'],
            dropout=0
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])

    results = []
    references = []
    print(f"Testing: {os.path.basename(args.checkpoint)} | Beam={args.beam_width}")
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Translating"):
        data = json.loads(line)
        src_text = data['zh']
        references.append(data['en']) # 收集参考答案
        
        src_tokens = zh_tokenize(src_text)
        src_ids = [src_vocab.sos_id] + src_vocab.encode(src_tokens) + [src_vocab.eos_id]
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        if model_type == 'rnn':
            pred_ids = beam_search_rnn(model, src_tensor, args.beam_width, args.max_len, tgt_vocab.sos_id, tgt_vocab.eos_id, device)
        else:
            pred_ids = beam_search_transformer(model, src_tensor, args.beam_width, args.max_len, tgt_vocab.sos_id, tgt_vocab.eos_id, src_vocab.pad_id, device)
            
        pred_tokens = tgt_vocab.decode(pred_ids)
        pred_str = " ".join(pred_tokens).replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
        results.append(pred_str)

    # 保存结果
    output_data = [{"zh": json.loads(l)['zh'], "en": r} for l, r in zip(lines, results)]
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in output_data:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    # --- 自动计算并输出 BLEU ---
    bleu = sacrebleu.corpus_bleu(results, [references], force=True)
    print("\n" + "="*30)
    print(f"Model: {os.path.basename(args.checkpoint)}")
    print(f"BLEU Score: {bleu.score:.2f}")
    print("="*30)

    # ... 后续推理逻辑保持不变 ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEFAULT_MODEL = 'checkpoints/transformer_abs_layer_best.pt' 
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--input_file', type=str, default='test.jsonl')
    parser.add_argument('--output_file', type=str, default='submission.jsonl')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=100)
    args = parser.parse_args()
    main(args)