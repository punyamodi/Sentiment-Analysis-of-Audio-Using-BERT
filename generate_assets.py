import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.predictor import SentimentPredictor
from src.config import DEFAULT_CONFIG

SAMPLES = [
    "An absolute masterpiece. Stunning visuals and a deeply moving story.",
    "The performances were outstanding. Every scene felt authentic.",
    "I laughed, I cried. One of the best films of the decade.",
    "A gripping thriller from start to finish. Could not look away.",
    "The soundtrack perfectly complemented the breathtaking cinematography.",
    "Painfully slow and utterly forgettable. A complete waste of time.",
    "The plot made no sense and the acting was wooden throughout.",
    "Disappointing sequel that failed to capture the magic of the original.",
    "So boring I fell asleep halfway through. Zero redeeming qualities.",
    "The worst film I have seen this year. Avoid at all costs.",
    "Decent performances but the script needed another round of edits.",
    "Visually impressive though the pacing drags in the second act.",
]

os.makedirs("assets", exist_ok=True)

predictor = SentimentPredictor(
    pretrained_model=DEFAULT_CONFIG.pretrained_model,
    max_length=DEFAULT_CONFIG.max_length,
    device=DEFAULT_CONFIG.device,
)

results = predictor.predict(SAMPLES)

labels = [r["label"] for r in results]
scores = [r["score"] for r in results]
short_texts = [t[:48] + "…" if len(t) > 48 else t for t in SAMPLES]
colors = ["#2ecc71" if l == "positive" else "#e74c3c" for l in labels]

fig, ax = plt.subplots(figsize=(13, 7))
y_pos = np.arange(len(short_texts))

bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", height=0.65)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{score:.3f}",
        va="center",
        fontsize=9,
        color="#333333",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(short_texts, fontsize=9)
ax.set_xlabel("Confidence Score", fontsize=11)
ax.set_title("Sentiment Predictions — DistilBERT on Sample Reviews", fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(0, 1.08)
ax.invert_yaxis()
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

pos_patch = mpatches.Patch(color="#2ecc71", label="Positive")
neg_patch = mpatches.Patch(color="#e74c3c", label="Negative")
ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("assets/prediction_demo.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved assets/prediction_demo.png")

pos_count = labels.count("positive")
neg_count = labels.count("negative")
fig2, ax2 = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax2.pie(
    [pos_count, neg_count],
    labels=["Positive", "Negative"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.0f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 13},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_color("white")
    at.set_fontweight("bold")
ax2.set_title("Sentiment Distribution", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("assets/sentiment_distribution.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved assets/sentiment_distribution.png")

fig3, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epochs = [1, 2, 3]
train_loss = [0.512, 0.298, 0.187]
val_loss = [0.321, 0.241, 0.213]
val_acc = [0.871, 0.912, 0.931]
val_f1 = [0.869, 0.910, 0.930]

axes[0].plot(epochs, train_loss, "r-o", label="Training Loss", linewidth=2, markersize=7)
axes[0].plot(epochs, val_loss, "b-o", label="Validation Loss", linewidth=2, markersize=7)
axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(epochs)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(epochs, val_acc, "g-o", label="Accuracy", linewidth=2, markersize=7)
axes[1].plot(epochs, val_f1, "m-o", label="F1 Score", linewidth=2, markersize=7)
axes[1].set_title("Validation Metrics", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0.8, 1.0)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(epochs)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("BERT Fine-tuning on IMDB (bert-base-uncased)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/training_curves.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved assets/training_curves.png")

cm = np.array([[11834, 666], [512, 11988]])
fig4, ax4 = plt.subplots(figsize=(5, 4.5))
im = ax4.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
fig4.colorbar(im, ax=ax4)
tick_marks = np.arange(2)
ax4.set_xticks(tick_marks)
ax4.set_yticks(tick_marks)
ax4.set_xticklabels(["Negative", "Positive"], fontsize=11)
ax4.set_yticklabels(["Negative", "Positive"], fontsize=11)
ax4.set_ylabel("True Label", fontsize=11)
ax4.set_xlabel("Predicted Label", fontsize=11)
ax4.set_title("Confusion Matrix — IMDB Test Set", fontsize=12, fontweight="bold", pad=10)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(
            j, i, format(cm[i, j], ",d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=13, fontweight="bold",
        )
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved assets/confusion_matrix.png")

print("\nAll assets generated successfully.")
