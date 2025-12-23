import subprocess

MODELS = [
    ("rnn_ablation_dot.pt", "beam"),
    ("rnn_ablation_general.pt", "beam"),
    ("rnn_ablation_tf.pt", "beam"),
    ("rnn_best_expected.pt", "beam"),
    ("rnn_decoding_greedy.pt", "greedy"),
]

for ckpt, decode in MODELS:
    cmd = [
        "python", "inference_rnn_test.py",
        "--ckpt", f"checkpoints/{ckpt}",
        "--test", "test.jsonl",
        "--decode", decode,
    ]
    if decode == "beam":
        cmd += ["--beam_size", "5"]

    print("\n" + "="*80)
    print("RUN:", " ".join(cmd))
    print("="*80)

    subprocess.run(cmd)
