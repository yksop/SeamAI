import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "outputs", "results", "rq3_system_summary.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.sort_values('e2e_mean_ms', ascending=True).reset_index(drop=True)

# --- FIGURE 1: E2E latency breakdown (stacked horizontal bar) ---
fig, ax = plt.subplots(figsize=(12, 10))

bars_perception = df['perception_mean_ms'].values
bars_features = df['features_mean_ms'].values
bars_gru = df['gru_mean_ms'].values
bars_capture = df['capture_mean_ms'].values

y = np.arange(len(df))

ax.barh(y, bars_perception, label='Perception', color='#4472C4')
ax.barh(y, bars_features, left=bars_perception, label='Feature extraction', color='#70AD47')
ax.barh(y, bars_gru, left=bars_perception + bars_features, label='GRU inference', color='#ED7D31')
ax.barh(y, bars_capture, left=bars_perception + bars_features + bars_gru, label='Video capture', color='#A5A5A5')

ax.axvline(x=33.33, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='33.3 ms (30 FPS)')

for i, row in df.iterrows():
    ax.text(row['e2e_mean_ms'] + 0.5, i, f"{row['e2e_fps']:.1f} FPS", va='center', fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(df['config'], fontsize=8)
ax.set_xlabel('End-to-end latency per frame (ms)', fontsize=11)
ax.set_title('RQ3: System Latency Breakdown (Video Capture → Intent Prediction)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 50)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "rq3_system_latency_breakdown.png")
plt.savefig(out1, dpi=200, bbox_inches='tight')
print(f"Saved: {out1}")

# --- FIGURE 2: FP32 vs FP16 comparison ---
fig2, ax2 = plt.subplots(figsize=(10, 5))

frontend_avg = df.groupby(['perception', 'precision']).agg(
    e2e_mean=('e2e_mean_ms', 'mean'),
).reset_index()

base_models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8n-pose', 'YOLOv8s-pose']
fp32_vals = []
fp16_vals = []
for m in base_models:
    row32 = frontend_avg[(frontend_avg['perception'] == m) & (frontend_avg['precision'] == 'fp32')]
    row16 = frontend_avg[(frontend_avg['perception'] == f'{m} [FP16]') & (frontend_avg['precision'] == 'fp16')]
    fp32_vals.append(row32['e2e_mean'].values[0] if len(row32) > 0 else 0)
    fp16_vals.append(row16['e2e_mean'].values[0] if len(row16) > 0 else 0)

x = np.arange(len(base_models))
w = 0.35
bars1 = ax2.bar(x - w/2, fp32_vals, w, label='PyTorch FP32', color='#4472C4')
bars2 = ax2.bar(x + w/2, fp16_vals, w, label='TensorRT FP16', color='#ED7D31')

ax2.axhline(y=33.33, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='33.3 ms (30 FPS)')
ax2.set_xticks(x)
ax2.set_xticklabels(base_models, fontsize=10)
ax2.set_ylabel('Mean E2E latency (ms)', fontsize=11)
ax2.set_title('RQ3: PyTorch FP32 vs TensorRT FP16 — System-Level Latency', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)

for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.1f}', ha='center', fontsize=9)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.1f}', ha='center', fontsize=9)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "rq3_fp32_vs_fp16.png")
plt.savefig(out2, dpi=200, bbox_inches='tight')
print(f"Saved: {out2}")
