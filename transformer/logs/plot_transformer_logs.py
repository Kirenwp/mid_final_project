import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -------- regex：兼容你给的两种 log 风格 --------
PATTERNS = [
    re.compile(
        r"Epoch\s+(\d+).*?train\s+([0-9.]+).*?valid_loss\s+([0-9.]+).*?valid BLEU.*?([0-9.]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Epoch\s+(\d+).*?Train Loss:\s*([0-9.]+).*?Valid Loss:\s*([0-9.]+).*?Valid BLEU:\s*([0-9.]+)",
        re.IGNORECASE,
    ),
]

def parse_log(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pat in PATTERNS:
                m = pat.search(line)
                if m:
                    rows.append({
                        "epoch": int(m.group(1)),
                        "train_loss": float(m.group(2)),
                        "valid_loss": float(m.group(3)),
                        "valid_bleu": float(m.group(4)),
                    })
                    break
    if not rows:
        raise RuntimeError(f"No epoch info parsed from {path}")
    return pd.DataFrame(rows).sort_values("epoch")

def plot_one_run(df, name, out_dir):
    # ---- BLEU ----
    plt.figure()
    plt.plot(df["epoch"], df["valid_bleu"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Valid BLEU")
    plt.title(f"Valid BLEU vs Epoch ({name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_bleu.png"), dpi=200)
    plt.close()

    # ---- Loss ----
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_loss.png"), dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default=".", help="directory with *.log")
    ap.add_argument("--out_dir", type=str, default="plots", help="where to save figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logs = [f for f in os.listdir(args.log_dir) if f.endswith(".log")]
    if not logs:
        raise SystemExit("No .log files found")

    all_rows = []

    for log in logs:
        path = os.path.join(args.log_dir, log)
        name = os.path.splitext(log)[0]
        df = parse_log(path)

        # 保存单个 run 的 csv
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)

        # 画图
        plot_one_run(df, name, args.out_dir)

        df["run"] = name
        all_rows.append(df)

    # 合并所有 run
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(args.out_dir, "all_runs_epoch_metrics.csv"), index=False)

    print(f"Done. Parsed {len(logs)} logs.")
    print(f"Figures & CSV saved to: {args.out_dir}")

if __name__ == "__main__":
    main()