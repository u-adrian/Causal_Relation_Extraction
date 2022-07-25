import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    f1_scores = np.asarray([0.71033777,0.72798434,0.75642343,0.72654691,0.7258567])
    mean = f1_scores.mean()
    std = f1_scores.std()
    plt.ylim([0, 1.0])
    plt.xlim([0, 1.0])
    plt.bar(
        0.5,
        mean,
        yerr=std,
        width=0.1,
        color="blue",
        edgecolor="black",
        align="center"
    )
    plt.xticks([0.5],["BERT"])
    plt.ylabel("F1-Score")
    plt.show()
