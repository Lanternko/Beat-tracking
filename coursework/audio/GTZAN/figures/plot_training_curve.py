"""Plot CNN v4 training curve from train_v4_long.log"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Parse the log file
epochs, train_accs, val_accs = [], [], []
with open(PROJECT_ROOT / 'train_v4_long.log') as f:
    for line in f:
        m = re.match(r'Epoch (\d+) \| loss=[\d.]+ \| train=([\d.]+)% \| val=([\d.]+)%', line)
        if m:
            epochs.append(int(m.group(1)))
            train_accs.append(float(m.group(2)))
            val_accs.append(float(m.group(3)))

epochs = np.array(epochs)
train_accs = np.array(train_accs)
val_accs = np.array(val_accs)

# Find best val accuracy
best_idx = np.argmax(val_accs)
best_epoch = epochs[best_idx]
best_val = val_accs[best_idx]

# LR restart points (T_0=30, so restarts at epoch 30, 60, 90)
lr_restarts = [30, 60, 90]

# Plot
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_accs, color='#2176FF', linewidth=2, label='Train Accuracy', alpha=0.9)
ax.plot(epochs, val_accs, color='#FF8C00', linewidth=2, label='Val Accuracy', alpha=0.9)

# Mark best val accuracy
ax.plot(best_epoch, best_val, marker='*', markersize=18, color='#FF8C00',
        markeredgecolor='black', markeredgewidth=0.8, zorder=5)
ax.annotate(f'Best: {best_val:.0f}% (epoch {best_epoch})',
            xy=(best_epoch, best_val), xytext=(best_epoch - 20, best_val + 2.5),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#FF8C00', alpha=0.9))

# LR restart lines
for i, restart in enumerate(lr_restarts):
    if restart <= epochs[-1]:
        ax.axvline(x=restart, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)
        label_y = ax.get_ylim()[0] + 2
        ax.text(restart + 0.8, 33, 'LR Restart', rotation=90, fontsize=9,
                color='#666666', va='bottom')

ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('CNN v3 Training Curve (patience=50)', fontsize=15, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9, edgecolor='#cccccc')

ax.set_xlim(0, epochs[-1] + 2)
ax.set_ylim(25, 100)
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'figures' / 'training_curve.png', dpi=300, bbox_inches='tight')
print(f'Saved. Total epochs: {len(epochs)}, Best val: {best_val}% at epoch {best_epoch}')
