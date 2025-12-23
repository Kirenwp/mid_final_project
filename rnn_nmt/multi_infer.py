import subprocess

CKPT = "rnn_nmt/checkpoints/rnn_best_expected.pt"
SENTS = [
    "记录指出 HMX-1 曾询问此次活动是否违反了该法案。",
    "该指挥官写道“我们问的一个问题是这是否违反了《哈奇法案》，并被告知没有违反。”",
    "“听起来你被锁住了啊，”副司令回复道。",
    "白宫将此次“美国制造”活动定义为官方活动，因此不受《哈奇法案》管辖。",
    "但是即使是官方活动也带有政治色彩。",
    "活动中，总统推行当时参议院的议题：医疗改革，并大肆吹捧对政府法规的管控。",
]

for i, s in enumerate(SENTS):
    print(f"\n[{i}] ZH: {s}")
    subprocess.run(["python", "rnn_nmt/inference.py", "--ckpt", CKPT, "--text", s], check=False)
