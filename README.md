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

python inference.py --checkpoint checkpoints/transformer_abs_layer_best.pt --output_file results/abs_layer.jsonl
python inference.py --checkpoint checkpoints/transformer_abs_rms_best.pt --output_file results/abs_rms.jsonl
python inference.py --checkpoint checkpoints/transformer_rlt_rope_best.pt --output_file results/rlt_rope.jsonl

python inference.py --checkpoint checkpoints/transformer_bs128_best.pt --output_file results/bs128.jsonl
python inference.py --checkpoint checkpoints/transformer_lr001_best.pt --output_file results/lr001.jsonl

python inference.py --checkpoint checkpoints/transformer_scale_small_best.pt --output_file results/scale_small.jsonl

## rnn
cd rnn_nmt
python inference_rnn_test.py \
  --ckpt checkpoints/rnn_additive_tf_beam.pt \
  --test test.jsonl \
  --out preds_additive_tf_beam.txt
