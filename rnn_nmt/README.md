\
# RNN NMT (Chinese -> English) with Ablations

This folder contains a clean RNN-based NMT implementation that supports:
- RNN type: GRU / LSTM
- Attention alignment: dot / general (multiplicative) / additive
- Training policy: teacher forcing ratio (1.0 -> TF, 0.0 -> free running)
- Decoding: greedy / beam search (beam_size, length norm)
- BLEU evaluation (sacrebleu if installed; fallback to NLTK)

## Train (single run)

Example (adjust file paths to your jsonl):
```bash
python train_rnn_nmt.py \
  --train train_10k.jsonl --valid valid.jsonl --test test.jsonl \
  --rnn_type gru --attn_type dot \
  --teacher_forcing_ratio 0.5 \
  --decode_valid greedy --beam_size 5 \
  --epochs 10 --batch_size 64 --lr 1e-3 \
  --save_dir checkpoints --run_name rnn_model
```

Outputs:
- `checkpoints/rnn_model.pt`  (includes model + vocabs + config)

## Inference (one sentence)

```bash
python inference.py --ckpt checkpoints/rnn_model.pt --text "广州是一座城市"
```

## Run ablation grid

```bash
python ablate_rnn.py \
  --train train_10k.jsonl --valid valid.jsonl --test test.jsonl \
  --attn_types dot,general,additive \
  --tf_ratios 1.0,0.5,0.0 \
  --decodings greedy,beam \
  --beam_sizes 3,5 \
  --epochs 10
```

The script writes `checkpoints_ablation/ablation_results.csv`.
