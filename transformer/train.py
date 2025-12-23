import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sacrebleu
import math
import random
import time

# 导入你的 utils
from utils import (
    set_seed, Vocab, JsonlTranslationDataset, collate_batch, save_checkpoint,
    zh_tokenize, en_tokenize
)

# 导入模型
from model_rnn import EncoderRNN, AttnDecoderRNN
from model_transformer import TransformerModel
#from model_transformer_relative import TransformerModel
# ==========================================
# 辅助函数：日志打印
# ==========================================
def log_print(msg, log_file):
    """同时打印到屏幕和写入文件"""
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# ==========================================
# 辅助函数：Transformer Mask 生成
# ==========================================
def create_mask(src, tgt, pad_idx, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device) == 1).transpose(0, 1)
    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    
    return tgt_mask, src_padding_mask, tgt_padding_mask

# ==========================================
# 验证逻辑
# ==========================================
def greedy_decode_rnn(encoder, decoder, src, max_len, sos_id, eos_id, device):
    batch_size = src.size(0)
    enc_out, hidden = encoder(src)
    dec_input = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
    predictions = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        output, hidden, _ = decoder(dec_input, hidden, enc_out)
        top1 = output.argmax(2)
        predictions.append(top1)
        dec_input = top1
        finished = finished | (top1.squeeze(1) == eos_id)
        if finished.all(): break
    return torch.cat(predictions, dim=1)

def greedy_decode_transformer(model, src, max_len, sos_id, eos_id, pad_id, device):
    model.eval()
    batch_size = src.size(0)
    src_mask = (src == pad_id)
    memory = model.transformer_encoder(
        model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model)),
        src_key_padding_mask=src_mask
    )
    ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
    for i in range(max_len):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
        out = model.transformer_decoder(
            model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.d_model)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        prob = model.out(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys[:, 1:]

def validate(model_obj, dataloader, criterion, tgt_vocab, device, args):
    if args.model_type == 'rnn':
        encoder, decoder = model_obj
        encoder.eval(); decoder.eval()
    else:
        model = model_obj
        model.eval()

    total_loss = 0
    refs, preds = [], []

    with torch.no_grad():
        for src, tgt_in, tgt_out in dataloader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            # Loss Calculation
            if args.model_type == 'rnn':
                enc_out, hidden = encoder(src)
                dec_input = tgt_in[:, 0].unsqueeze(1)
                loss_batch = 0
                for t in range(1, tgt_in.size(1)):
                    output, hidden, _ = decoder(dec_input, hidden, enc_out)
                    loss_batch += criterion(output.squeeze(1), tgt_out[:, t-1])
                    dec_input = tgt_in[:, t].unsqueeze(1)
                total_loss += loss_batch.item()
            elif args.model_type == 'transformer':
                tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_in, tgt_vocab.pad_id, device)
                output = model(src, tgt_in, tgt_mask=tgt_mask, src_padding_mask=src_pad_mask, tgt_padding_mask=tgt_pad_mask)
                loss_batch = criterion(output.reshape(-1, len(tgt_vocab.itos)), tgt_out.reshape(-1))
                total_loss += loss_batch.item()

            # Greedy Search for BLEU
            # 注意：如果想消除sacrebleu警告，可在这里加 force=True
            if args.model_type == 'rnn':
                pred_ids = greedy_decode_rnn(encoder, decoder, src, 50, tgt_vocab.sos_id, tgt_vocab.eos_id, device)
            else:
                pred_ids = greedy_decode_transformer(model, src, 50, tgt_vocab.sos_id, tgt_vocab.eos_id, tgt_vocab.pad_id, device)

            for i in range(src.size(0)):
                ref_ids = tgt_out[i].tolist()
                p_ids = pred_ids[i].tolist()
                refs.append(" ".join(tgt_vocab.decode(ref_ids)))
                preds.append(" ".join(tgt_vocab.decode(p_ids)))

    avg_loss = total_loss / len(dataloader)
    # 使用 force=True 避免 tokenize 警告
    bleu_score = sacrebleu.corpus_bleu(preds, [refs], force=True).score
    return avg_loss, bleu_score

# ==========================================
# 主训练逻辑
# ==========================================
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 确定实验名称
    if args.exp_name:
        experiment_name = args.exp_name
    else:
        sub_type = args.attn_method if args.model_type == 'rnn' else args.norm_type
        experiment_name = f"{args.model_type}_{sub_type}"

    # 准备日志和保存路径
    if not os.path.exists("logs"): os.makedirs("logs")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")

    log_file = f"logs/{experiment_name}.log"
    checkpoint_path = f"checkpoints/{experiment_name}_best.pt"

    # 初始化日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Training Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Config: {vars(args)}\n")
        f.write("-" * 50 + "\n")
    
    log_print(f"Mode: {args.model_type.upper()} | Save Name: {experiment_name}", log_file)
    
    # 2. Load Data
    temp_ds = JsonlTranslationDataset(args.train_file, src_field=args.src_field, tgt_field=args.tgt_field)
    from utils import zh_tokenize, en_tokenize
    src_tokens = [zh_tokenize(item[0]) for item in temp_ds]
    tgt_tokens = [en_tokenize(item[1]) for item in temp_ds]
    
    src_vocab = Vocab.build(src_tokens, min_freq=2, max_size=15000)
    tgt_vocab = Vocab.build(tgt_tokens, min_freq=2, max_size=15000)
    log_print(f"Vocab: Src={len(src_vocab.itos)}, Tgt={len(tgt_vocab.itos)}", log_file)

    # 3. Dataloaders
    train_ds = JsonlTranslationDataset(args.train_file, src_field=args.src_field, tgt_field=args.tgt_field)
    valid_ds = JsonlTranslationDataset(args.valid_file, src_field=args.src_field, tgt_field=args.tgt_field)
    from functools import partial
    collate_fn = partial(collate_batch, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Model Init
    if args.model_type == 'rnn':
        encoder = EncoderRNN(len(src_vocab.itos), args.hidden_size).to(device)
        decoder = AttnDecoderRNN(args.hidden_size, len(tgt_vocab.itos), attn_method=args.attn_method).to(device)
        model_obj = (encoder, decoder)
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
        log_print(f"RNN Init: Hidden={args.hidden_size}, Attn={args.attn_method}, TF_Ratio={args.tf_ratio}", log_file)
    elif args.model_type == 'transformer':
        model = TransformerModel(
            src_vocab_size=len(src_vocab.itos),
            tgt_vocab_size=len(tgt_vocab.itos),
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            norm_type=args.norm_type,
            dropout=0.1
        ).to(device)
        model_obj = model
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        log_print(f"Transformer Init: d_model={args.d_model}, layers={args.num_layers}, norm={args.norm_type}", log_file)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    
    # [修改点1] 初始化 Best BLEU (之前是 Loss)
    best_valid_bleu = -1.0 

    # 5. Training Loop
    for epoch in range(args.epochs):
        if args.model_type == 'rnn': encoder.train(); decoder.train()
        else: model.train()
        start_time = time.time()
        train_loss_sum = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        
        for src, tgt_in, tgt_out in pbar:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()

            if args.model_type == 'rnn':
                enc_out, hidden = encoder(src)
                loss = 0
                dec_input = tgt_in[:, 0].unsqueeze(1)
                use_tf = True if random.random() < args.tf_ratio else False
                for t in range(1, tgt_in.size(1)):
                    output, hidden, _ = decoder(dec_input, hidden, enc_out)
                    loss += criterion(output.squeeze(1), tgt_out[:, t-1])
                    if use_tf: dec_input = tgt_in[:, t].unsqueeze(1)
                    else: dec_input = output.argmax(2)
                loss = loss / tgt_in.size(1)
            elif args.model_type == 'transformer':
                tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_in, tgt_vocab.pad_id, device)
                output = model(src, tgt_in, tgt_mask=tgt_mask, src_padding_mask=src_pad_mask, tgt_padding_mask=tgt_pad_mask)
                loss = criterion(output.reshape(-1, len(tgt_vocab.itos)), tgt_out.reshape(-1))

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # 6. Validation
        print(f"Epoch {epoch+1} Validating...")
        avg_train_loss = train_loss_sum / len(train_loader)
        valid_loss, valid_bleu = validate(model_obj, valid_loader, criterion, tgt_vocab, device, args)
        epoch_mins, epoch_secs = divmod(time.time() - start_time, 60)
        log_msg = f"Epoch {epoch+1} | Time: {int(epoch_mins)}m {int(epoch_secs)}s｜ Train Loss: {avg_train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid BLEU: {valid_bleu:.2f}"
        log_print(log_msg, log_file)

        # 7. Save Best Checkpoint (Based on BLEU)
        # [修改点2] 比较 BLEU 而不是 Loss
        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            log_print(f"!! New Best BLEU ({best_valid_bleu:.2f}) !! Saving to {checkpoint_path}", log_file)
            
            state_dict = {
                'model_type': args.model_type,
                'model_state': model.state_dict() if args.model_type == 'transformer' else {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                'config': vars(args),
                'src_vocab': src_vocab.to_dict(),
                'tgt_vocab': tgt_vocab.to_dict()
            }
            torch.save(state_dict, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Name Param
    parser.add_argument('--exp_name', type=str, default=None, help='Custom name for log and checkpoint files')

    # Data Params
    parser.add_argument('--train_file', type=str, default='train_100k.jsonl')
    parser.add_argument('--valid_file', type=str, default='valid.jsonl')
    parser.add_argument('--src_field', type=str, default='zh')
    parser.add_argument('--tgt_field', type=str, default='en')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model Selector
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer'])
    
    # RNN Params
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--attn_method', type=str, default='dot', choices=['dot', 'general', 'concat'])
    parser.add_argument('--tf_ratio', type=float, default=0.5)
    
    # Transformer Params
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--norm_type', type=str, default='layer', choices=['layer', 'rms'])

    args = parser.parse_args()
    train(args)