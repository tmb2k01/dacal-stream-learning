import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Point to your result csv
csv_path = "out/res_initial.csv"

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}. Make sure the EXP script finished running.")
    exit()


df = pd.read_csv(csv_path)
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# ---------------------------------------------------------
# Plot: ROC-AUC Score
# ---------------------------------------------------------

df_sorted_score = df.sort_values("score", ascending=False)

sns.barplot(
    data=df_sorted_score,
    x="score",
    y="localizer",
    hue="loader",
    ax=ax,
    palette="viridis" )

ax.set_title("Concept Drift Localization Performance on Synthetic Time-Series\nDetection Accuracy: ROC-AUC Score (Higher is Better)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("ROC-AUC Score", fontsize=12)
ax.set_ylabel("Localizer Model", fontsize=12)
ax.set_xlim(0, 1.05) 
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random Guessing (0.5)')
ax.legend(loc='lower right')

plt.tight_layout()
filename = csv_path.split('/')[-1].replace('.csv', '.png')
save_path = f"out/results_{filename}"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()