import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def best_summary(df_run: pd.DataFrame):
    # best BLEU epoch
    i_bleu = df_run["valid_bleu"].idxmax()
    # best valid loss epoch
    i_vloss = df_run["valid_loss"].idxmin()
    return {
        "best_bleu": float(df_run.loc[i_bleu, "valid_bleu"]),
        "best_bleu_epoch": int(df_run.loc[i_bleu, "epoch"]),
        "best_valid_loss": float(df_run.loc[i_vloss, "valid_loss"]),
        "best_valid_loss_epoch": int(df_run.loc[i_vloss, "epoch"]),
        "last_bleu": float(df_run.sort_values("epoch")["valid_bleu"].iloc[-1]),
        "last_valid_loss": float(df_run.sort_values("epoch")["valid_loss"].iloc[-1]),
    }

def plot_bleu(df, runs, title, outpath):
    plt.figure()
    for r in runs:
        g = df[df["run"] == r].sort_values("epoch")
        plt.plot(g["epoch"], g["valid_bleu"], label=r)
    plt.xlabel("Epoch")
    plt.ylabel("Valid BLEU")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_loss(df, runs, title, outpath):
    plt.figure()
    for r in runs:
        g = df[df["run"] == r].sort_values("epoch")
        plt.plot(g["epoch"], g["train_loss"], label=f"{r} train")
        plt.plot(g["epoch"], g["valid_loss"], label=f"{r} valid", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def make_table(df, runs, out_csv):
    rows = []
    for r in runs:
        g = df[df["run"] == r].copy()
        if len(g) == 0:
            continue
        s = best_summary(g)
        rows.append({"run": r, **s})
    tab = pd.DataFrame(rows).sort_values("best_bleu", ascending=False)
    tab.to_csv(out_csv, index=False)
    return tab

def pick_existing(df, candidates):
    return [c for c in candidates if (df["run"] == c).any()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="all_runs_epoch_metrics.csv")
    ap.add_argument("--out", default="ablation_out", help="output dir")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ensure_dir(args.out)

    # 你的 run 名字（来自截图）
    RUN_ABS_LAYER   = "transformer_abs_layer"
    RUN_REL_ROPE    = "transformer_rlt_rope"
    RUN_ABS_RMS     = "transformer_abs_rms"
    RUN_BS128       = "transformer_bs128"
    RUN_LR001       = "transformer_lr001"
    RUN_SCALE_SMALL = "transformer_scale_small"
    RUN_SCALE_LARGE = "transformer_scale_large"  # 如果你后续补了 large，就用这个名字或改成你的实际文件名

    # 1) absolute vs relative
    runs = pick_existing(df, [RUN_ABS_LAYER, RUN_REL_ROPE])
    if len(runs) >= 2:
        plot_bleu(df, runs, "Valid BLEU: Absolute vs Relative", os.path.join(args.out, "bleu_abs_vs_rel.png"))
        plot_loss(df, runs, "Loss: Absolute vs Relative", os.path.join(args.out, "loss_abs_vs_rel.png"))
        make_table(df, runs, os.path.join(args.out, "table_abs_vs_rel.csv"))
    else:
        print("[WARN] Missing run for abs vs rel:", runs)

    # 2) layernorm vs rmsnorm (absolute)
    runs = pick_existing(df, [RUN_ABS_LAYER, RUN_ABS_RMS])
    if len(runs) >= 2:
        plot_bleu(df, runs, "Valid BLEU: LayerNorm vs RMSNorm (Absolute)", os.path.join(args.out, "bleu_layer_vs_rms.png"))
        plot_loss(df, runs, "Loss: LayerNorm vs RMSNorm (Absolute)", os.path.join(args.out, "loss_layer_vs_rms.png"))
        make_table(df, runs, os.path.join(args.out, "table_layer_vs_rms.csv"))
    else:
        print("[WARN] Missing run for layer vs rms:", runs)

    # 3) batch size 64 vs 128
    # 你没有显式 bs64 的文件名，但 rlt_rope 的 batch 很可能是 64（取它作为 bs64 对比）
    runs = pick_existing(df, [RUN_REL_ROPE, RUN_BS128])
    if len(runs) >= 2:
        plot_bleu(df, runs, "Valid BLEU: Batch Size 64 vs 128", os.path.join(args.out, "bleu_bs64_vs_bs128.png"))
        plot_loss(df, runs, "Loss: Batch Size 64 vs 128", os.path.join(args.out, "loss_bs64_vs_bs128.png"))
        make_table(df, runs, os.path.join(args.out, "table_bs64_vs_bs128.csv"))
    else:
        print("[WARN] Missing run for bs64 vs bs128:", runs)

    # 4) learning rate 1e-3 vs 5e-4
    # 你 lr=1e-3 是 lr001；lr=5e-4 用 abs_layer 或 rlt_rope 之一做 baseline（这里用 abs_layer）
    runs = pick_existing(df, [RUN_LR001, RUN_ABS_LAYER])
    if len(runs) >= 2:
        plot_bleu(df, runs, "Valid BLEU: LR 1e-3 vs 5e-4", os.path.join(args.out, "bleu_lr001_vs_lr0005.png"))
        plot_loss(df, runs, "Loss: LR 1e-3 vs 5e-4", os.path.join(args.out, "loss_lr001_vs_lr0005.png"))
        make_table(df, runs, os.path.join(args.out, "table_lr001_vs_lr0005.csv"))
    else:
        print("[WARN] Missing run for lr ablation:", runs)

    # 5) normal vs large model scale
    # 你现在只有 small；如果你补 large（文件名 transformer_scale_large.csv / log），脚本会自动画
    runs = pick_existing(df, [RUN_ABS_LAYER, RUN_SCALE_LARGE])  # 这里把 abs_layer 当 normal baseline
    # 如果你希望 normal 用 rlt_rope 也可以改成 RUN_REL_ROPE
    if len(runs) >= 2:
        plot_bleu(df, runs, "Valid BLEU: Normal vs Large Model Scale", os.path.join(args.out, "bleu_normal_vs_large.png"))
        plot_loss(df, runs, "Loss: Normal vs Large Model Scale", os.path.join(args.out, "loss_normal_vs_large.png"))
        make_table(df, runs, os.path.join(args.out, "table_normal_vs_large.csv"))
    else:
        print("[WARN] Large scale run not found. If you have it, ensure run name matches:", RUN_SCALE_LARGE)

    # 额外：small vs normal（你当前就能画，防止最后一组空着）
    runs = pick_existing(df, [RUN_ABS_LAYER, RUN_SCALE_SMALL])
    if len(runs) >= 2:
        plot_bleu(df, runs, "Valid BLEU: Normal vs Small Model Scale", os.path.join(args.out, "bleu_normal_vs_small.png"))
        plot_loss(df, runs, "Loss: Normal vs Small Model Scale", os.path.join(args.out, "loss_normal_vs_small.png"))
        make_table(df, runs, os.path.join(args.out, "table_normal_vs_small.csv"))

    print("Done. Outputs saved to:", args.out)

if __name__ == "__main__":
    main()