\
import argparse, itertools, os, subprocess, json, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--valid", type=str, required=True)
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="checkpoints_ablation")
    ap.add_argument("--csv_out", type=str, default="ablation_results.csv")

    ap.add_argument("--attn_types", type=str, default="dot,general,additive")
    ap.add_argument("--tf_ratios", type=str, default="1.0,0.5,0.0")
    ap.add_argument("--decodings", type=str, default="greedy,beam")
    ap.add_argument("--beam_sizes", type=str, default="5")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rnn_type", type=str, default="gru")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    attn_types = args.attn_types.split(",")
    tf_ratios = [float(x) for x in args.tf_ratios.split(",")]
    decodings = args.decodings.split(",")
    beam_sizes = [int(x) for x in args.beam_sizes.split(",")]

    rows = []
    for attn, tf, dec, beam in itertools.product(attn_types, tf_ratios, decodings, beam_sizes):
        run_name = f"rnn_{args.rnn_type}_attn-{attn}_tf-{tf}_dec-{dec}_beam-{beam}"
        cmd = [
            "python", "train_rnn_nmt.py",
            "--train", args.train,
            "--valid", args.valid,
            "--save_dir", args.save_dir,
            "--run_name", run_name,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--rnn_type", args.rnn_type,
            "--attn_type", attn,
            "--teacher_forcing_ratio", str(tf),
            "--decode_valid", dec,
            "--beam_size", str(beam),
        ]
        if args.test:
            cmd += ["--test", args.test]
        print("\n=== RUN:", run_name, "===\n", " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout)
        if p.returncode != 0:
            print(p.stderr)
            rows.append({"run_name": run_name, "status": "FAIL", "stderr": p.stderr[:2000]})
            continue

        # parse the last "Test BLEU" line if present, else parse valid BLEU line
        test_g = test_b = None
        valid_bleu = None
        for line in p.stdout.strip().splitlines():
            if "valid BLEU" in line:
                # ... valid BLEU(greedy) 12.34 ...
                m = None
                m = __import__("re").search(r"valid BLEU\((.+?)\)\s+([0-9.]+)", line)
                if m:
                    valid_bleu = float(m.group(2))
            if line.startswith("Test BLEU"):
                m = __import__("re").search(r"greedy=([0-9.]+)\s+\|\s+beam\(k=([0-9]+)\)=([0-9.]+)", line)
                if m:
                    test_g = float(m.group(1))
                    test_b = float(m.group(3))

        rows.append({
            "run_name": run_name,
            "rnn_type": args.rnn_type,
            "attn_type": attn,
            "teacher_forcing_ratio": tf,
            "decode_valid": dec,
            "beam_size": beam,
            "valid_bleu": valid_bleu,
            "test_bleu_greedy": test_g,
            "test_bleu_beam": test_b,
            "status": "OK"
        })

        with open(os.path.join(args.save_dir, args.csv_out), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print("Done. Results at:", os.path.join(args.save_dir, args.csv_out))

if __name__ == "__main__":
    main()
