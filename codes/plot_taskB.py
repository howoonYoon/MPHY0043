import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = {
    "Baseline": Path("/myriadfs/home/rmaphyo/Scratch/surgery_time_project/runs_taskB/baseline/test_metrics.json"),
    "Timed": Path("/myriadfs/home/rmaphyo/Scratch/surgery_time_project/runs_taskB/time_concat_resume/test_metrics.json"),
    "Oracle": Path("/myriadfs/home/rmaphyo/Scratch/surgery_time_project/runs_taskB/time_concat_oracle/test_metrics.json"),
}

data = {name: json.loads(p.read_text()) for name, p in files.items()}

phases = list(data["Baseline"]["phasewise"].keys())

scores = {}
for name in files.keys():
    scores[name] = [
        data[name]["phasewise"][ph]["mAP(AUPRC_macro)"]
        for ph in phases
    ]

x = np.arange(len(phases))
w = 0.25

plt.figure(figsize=(9, 4))

plt.bar(x - w, scores["Baseline"], w, label="Baseline")
plt.bar(x,     scores["Timed"],    w, label="Timed")
plt.bar(x + w, scores["Oracle"],   w, label="Oracle")

plt.xticks(x, phases, rotation=30, ha="right")
plt.ylabel("Macro AUPRC")
plt.title("Phase-wise Tool Detection Performance")
plt.legend()
plt.tight_layout()

plt.savefig("phasewise_comparison.png", dpi=300)
plt.show()
