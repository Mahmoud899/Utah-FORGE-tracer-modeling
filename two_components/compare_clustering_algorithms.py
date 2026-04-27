from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_ENSEMBLE_CSV = "two_components/outputs/phase_exploration/exploration_ensemble.csv"
OUTPUT_DIR = "two_components/outputs/clustering_evaluation"
PARAM_COLS = ["best_MRT1", "best_MRT2", "best_Pe1", "best_Pe2", "best_fr1", "best_fr2"]
K_VALUES = range(1, 26)


def load_normalized_points():
    df = pd.read_csv(INPUT_ENSEMBLE_CSV)
    x_all = df[PARAM_COLS].to_numpy(dtype=float)
    x_min = np.min(x_all, axis=0)
    x_max = np.max(x_all, axis=0)
    z_all = (x_all - x_min) / (x_max - x_min)
    return df, z_all


def evaluate_labels(z_points, labels):
    labels = np.asarray(labels, dtype=int)
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    size_values = counts.to_numpy(dtype=float)
    size_probs = size_values / np.sum(size_values)

    within_ss = 0.0
    for cluster_id in counts.index.to_numpy(dtype=int):
        cluster_points = z_points[labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        sq_dist = np.sum((cluster_points - centroid) ** 2, axis=1)
        within_ss += float(np.sum(sq_dist))

    total_ss = float(np.sum((z_points - np.mean(z_points, axis=0)) ** 2))
    explained_fraction = 0.0 if total_ss == 0.0 else 1.0 - within_ss / total_ss
    normalized_entropy = np.nan
    if len(size_values) > 1:
        entropy = float(-np.sum(size_probs * np.log(size_probs)))
        normalized_entropy = float(entropy / np.log(len(size_values)))

    return {
        "n_points": int(len(z_points)),
        "n_clusters": int(len(counts)),
        "explained_fraction": explained_fraction,
        "normalized_entropy": normalized_entropy,
    }


def run_kmeans_grid(z_points):
    rows = []
    labels_by_k = {}

    for k in K_VALUES:
        labels = KMeans(n_clusters=k, init="k-means++", random_state=0).fit_predict(z_points) + 1
        metrics = evaluate_labels(z_points, labels)
        selection_score = np.nan
        if k >= 2:
            selection_score = 0.5 * (
                metrics["explained_fraction"] + metrics["normalized_entropy"]
            )

        rows.append(
            {
                "k": k,
                "selection_score": selection_score,
                **metrics,
            }
        )
        labels_by_k[k] = labels

    evaluation_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_row = (
        evaluation_df.loc[evaluation_df["k"] >= 2]
        .sort_values(["selection_score", "k"], ascending=[False, True])
        .iloc[0]
    )
    best_k = int(best_row["k"])
    evaluation_df["is_selected"] = evaluation_df["k"] == best_k
    return evaluation_df, best_k, labels_by_k[best_k]


def save_evaluation_plot(evaluation_df, best_k, out_dir):
    best_row = evaluation_df.loc[evaluation_df["k"] == best_k].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        evaluation_df["k"],
        evaluation_df["explained_fraction"],
        marker="o",
        label="Explained Fraction",
    )
    ax.plot(
        evaluation_df["k"],
        evaluation_df["normalized_entropy"],
        marker="o",
        label="Normalized Entropy",
    )
    ax.axvline(best_k, color="#808080", linestyle="--", linewidth=1.0)
    ax.scatter(
        [best_k],
        [best_row["explained_fraction"]],
        color="#1f77b4",
        s=70,
        zorder=3,
    )
    ax.scatter(
        [best_k],
        [best_row["normalized_entropy"]],
        color="#ff7f0e",
        s=70,
        zorder=3,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title(f"K-means Evaluation (selected k={best_k})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_evaluation.png", dpi=200)
    plt.close(fig)


def save_pca_outputs(assignments_df, z_points, best_k, out_dir):
    centered = z_points - np.mean(z_points, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[:2].T

    pca_df = assignments_df[["run_idx", "seed", "J_m", "cluster_id"]].copy()
    pca_df["PC1"] = projection[:, 0]
    pca_df["PC2"] = projection[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=assignments_df["cluster_id"].to_numpy(dtype=int),
        cmap="tab20",
        s=24,
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA View of Selected K-means Clustering (k={best_k})")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(out_dir / "best_clustering_pca.png", dpi=200)
    plt.close(fig)

    pca_df.to_csv(out_dir / "best_clustering_pca_points.csv", index=False)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_df, z_points = load_normalized_points()
    evaluation_df, best_k, best_labels = run_kmeans_grid(z_points)

    assignments_df = ensemble_df.copy()
    assignments_df["cluster_id"] = best_labels

    evaluation_df.to_csv(out_dir / "kmeans_evaluation.csv", index=False)
    assignments_df.to_csv(out_dir / "best_clustering_assignments.csv", index=False)

    save_evaluation_plot(evaluation_df, best_k, out_dir)
    save_pca_outputs(assignments_df, z_points, best_k, out_dir)

    best_row = evaluation_df.loc[evaluation_df["k"] == best_k].iloc[0]
    print(f"Loaded ensemble rows: {len(ensemble_df)}")
    print(f"Selected best k: {best_k}")
    print(f"Explained fraction: {best_row['explained_fraction']:.6g}")
    print(f"Normalized entropy: {best_row['normalized_entropy']:.6g}")
    print(f"Selection score: {best_row['selection_score']:.6g}")
    print(f"Wrote: {out_dir / 'kmeans_evaluation.csv'}")
    print(f"Wrote: {out_dir / 'kmeans_evaluation.png'}")
    print(f"Wrote: {out_dir / 'best_clustering_assignments.csv'}")
    print(f"Wrote: {out_dir / 'best_clustering_pca_points.csv'}")
    print(f"Wrote: {out_dir / 'best_clustering_pca.png'}")


if __name__ == "__main__":
    main()
