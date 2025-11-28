import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def create_client_num_figures(
        data_path: Path,
        figures_dir: Path
):
    df = pd.read_csv(data_path)

    # Filter for the two models you want
    df_plot = df[df['model'].isin(['Shokri-CNN', 'ResNet-18'])]

    plt.figure()

    sns.lineplot(
        data=df_plot,
        x="num-clients",
        y="metrics/AUC",
        hue="model",  # two lines, different colors
        marker="o"
    )

    plt.xlabel("Anzahl der Clients")
    plt.ylabel("MIA AUC")
    plt.legend(title="Modell", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(figures_dir / "auc_per_client_number_per_model.png", dpi=300)
    plt.close()

    print("✅ Figures (client number) created successfully.")
