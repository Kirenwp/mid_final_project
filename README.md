# mid_final_project

## nllb 
cd transformer
python inference_nllb.py \
  --ckpt checkpoints_nllb/nllb_experiment_best \
  --test_file test.jsonl \
  --task both \
  --batch_size 16 \
  --num_beams 4 \
  --max_new_tokens 128

## transformer
python inference.py --checkpoint checkpoints/transformer_rlt_rope_best.pt --output_file results/rlt_rope.jsonl



## rnn
cd rnn_nmt
python inference_rnn_test.py \
  --ckpt checkpoints/rnn_additive_tf_beam.pt \
  --test test.jsonl \
  --out preds_additive_tf_beam.txt
