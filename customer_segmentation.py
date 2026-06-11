"""Customer segmentation pipeline for financial product targeting.

End-to-end unsupervised learning workflow that:
    1. Loads customer financial data.
    2. Reduces dimensionality with PCA (retains 80% of variance).
    3. Picks the optimal number of clusters via the elbow method.
    4. Fits a K-Means model and validates it with three internal metrics
       (silhouette, Calinski-Harabasz, Davies-Bouldin).
    5. Persists the fitted scaler, PCA, and K-Means models so new
       customers can be classified in production without re-training.
    6. Scores a new batch of customers using the same pipeline.

All plots are saved as PNG files under ``plots/`` and fitted artifacts
under ``models/``. The script is fully reproducible thanks to a fixed
random seed.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Fixed seed so segment IDs stay stable across reruns - required for any
# downstream system (marketing CRM, scoring jobs) that hard-codes them.
RANDOM_STATE = 10

# The notebook explored k = 1..12 visually; the elbow inflects at 5 and
# every internal metric agrees, so we lock it here for reproducibility.
OPTIMAL_CLUSTERS = 5

# Spanish column names from the source CSV are remapped to English snake_case
# so downstream code reads as professional analytics, not raw export dumps.
COLUMN_RENAME = {
    "Salario": "salary",
    "Limite_TC": "credit_card_limit",
    "Tiempo_cliente": "customer_tenure",
    "Producto_ofrecido": "product_offered",
    "Credito_tipo_1": "credit_type_1",
    "Credito_tipo_2": "credit_type_2",
    "Tipo_de_cliente": "customer_type",
}

# Continuous features whose per-cluster means describe each segment.
PROFILE_FEATURES = ["salary", "credit_card_limit", "customer_tenure"]

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_FILE = PROJECT_ROOT / "customers.csv"
NEW_DATA_FILE = PROJECT_ROOT / "new_customers.csv"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create output directories so the script is safe to run from scratch."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_customers(path: Path) -> pd.DataFrame:
    """Load the customers CSV and rename columns to English snake_case."""
    df = pd.read_csv(path)
    return df.rename(columns=COLUMN_RENAME)


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure to ``plots/`` and close it.

    Centralising the save/close call keeps the script headless-friendly
    (no GUI backend required) and prevents the figure stack from leaking.
    """
    out_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Exploratory analysis
# ---------------------------------------------------------------------------

def plot_salary_vs_credit_limit(df: pd.DataFrame) -> None:
    """Visualise raw separation between offered vs not-offered customers.

    This pre-clustering view motivates the segmentation: the two groups
    overlap heavily in the original feature space, so a smarter
    representation (PCA + K-Means) is required to tell them apart.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="credit_card_limit",
        y="salary",
        hue="product_offered",
        data=df,
    )
    plt.title("Customers: Salary vs Credit Limit by Product Offered", fontsize=14)
    plt.xlabel("Credit Limit")
    plt.ylabel("Salary")
    save_plot("salary_vs_credit_limit.png")


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def fit_pipeline(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, PCA]:
    """Standardise features and fit PCA retaining 80% of total variance.

    Standardisation is required because salary and credit limit are on
    a much larger scale than the binary credit-type flags - without it
    K-Means' Euclidean distance would be dominated by the monetary axes.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # ``n_components=0.8`` lets PCA decide how many axes to keep, trading
    # a small amount of variance for a much simpler, less noisy space.
    pca = PCA(n_components=0.8, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA components retained: {pca.n_components_}")
    print(f"Explained variance per component: {pca.explained_variance_ratio_}")
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.3f}")

    return X_pca, scaler, pca


def plot_elbow(X_pca: np.ndarray, k_range: range = range(1, 10)) -> None:
    """Plot inertia vs k to confirm the optimal number of clusters.

    The "elbow" is the k where adding another cluster stops materially
    reducing inertia - here it occurs at k=5.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, "bo-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow Method to Determine Optimal Clusters")
    plt.axvline(OPTIMAL_CLUSTERS, color="red", linestyle="--", alpha=0.5,
                label=f"Selected k = {OPTIMAL_CLUSTERS}")
    plt.legend()
    save_plot("elbow_method.png")


def fit_kmeans(X_pca: np.ndarray, n_clusters: int = OPTIMAL_CLUSTERS) -> tuple[KMeans, np.ndarray]:
    """Fit K-Means on the PCA-reduced space and return model + labels."""
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    cluster_labels = kmeans_model.fit_predict(X_pca)
    return kmeans_model, cluster_labels


def evaluate_clustering(X_pca: np.ndarray, labels: np.ndarray) -> dict:
    """Compute three complementary internal validation metrics.

    Using more than one metric guards against a single index masking a
    weak partition: a strong segmentation should be good on all three.
    """
    metrics = {
        "silhouette": silhouette_score(X_pca, labels),
        "calinski_harabasz": calinski_harabasz_score(X_pca, labels),
        "davies_bouldin": davies_bouldin_score(X_pca, labels),
    }
    print("\nClustering quality metrics:")
    print(f"  Silhouette Score:        {metrics['silhouette']:.3f}   (higher is better)")
    print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.3f}   (higher is better)")
    print(f"  Davies-Bouldin Score:    {metrics['davies_bouldin']:.3f}   (lower is better)")
    return metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_clusters_3d(
    pca_components: np.ndarray,
    labels: np.ndarray,
    filename: str,
    title: str,
) -> None:
    """Render the 3 PCA components as a 3D scatter coloured by cluster."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pca_components[:, 0],
        pca_components[:, 1],
        pca_components[:, 2],
        c=labels,
        cmap="tab10",
        s=40,
        alpha=0.8,
    )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster",
                       loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend)
    save_plot(filename)


def plot_combined_3d(original: pd.DataFrame, new: pd.DataFrame, filename: str) -> None:
    """Overlay original customers (dots) and newly scored ones (X markers)."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        original["PCA1"], original["PCA2"], original["PCA3"],
        c=original["Cluster"], cmap="tab10", s=30, alpha=0.5, label="Original",
    )
    ax.scatter(
        new["PCA1"], new["PCA2"], new["PCA3"],
        c=new["Cluster"], cmap="tab10", s=80, marker="X",
        edgecolors="black", linewidths=0.8, label="New",
    )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.set_title("Customer Clusters with New Data Overlaid")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    save_plot(filename)


# ---------------------------------------------------------------------------
# Cluster profiling
# ---------------------------------------------------------------------------

def build_cluster_profile(
    df: pd.DataFrame,
    n_clusters: int = OPTIMAL_CLUSTERS,
) -> pd.DataFrame:
    """Aggregate per-cluster means and category counts.

    The ``reindex`` calls guarantee every cluster id appears in the
    output even if a cluster has zero rows for a given category - this
    keeps downstream charts and joins predictable.
    """
    all_clusters = range(n_clusters)

    profile = (
        df.groupby("cluster")[PROFILE_FEATURES]
        .mean()
        .reindex(all_clusters)
        .reset_index()
    )

    count_columns = {
        "count_credit_type_1": ("credit_type_1", 1),
        "count_credit_type_2": ("credit_type_2", 1),
        "count_product_offered": ("product_offered", 1),
        "count_product_not_offered": ("product_offered", 0),
    }
    for out_col, (source_col, target_value) in count_columns.items():
        profile[out_col] = (
            df[df[source_col] == target_value]
            .groupby("cluster")[source_col]
            .count()
            .reindex(all_clusters, fill_value=0)
            .values
        )

    profile["total_customers"] = (
        df.groupby("cluster")["cluster"]
        .count()
        .reindex(all_clusters, fill_value=0)
        .values
    )

    return profile.fillna(0)


def plot_profile_bars(profile: pd.DataFrame, prefix: str) -> None:
    """Render bar charts for the three continuous profiling features."""
    chart_specs = [
        ("salary", "Average Salary per Cluster", "Average Salary"),
        ("credit_card_limit", "Average Credit Limit per Cluster", "Average Credit Limit"),
        ("customer_tenure", "Average Years as Customer per Cluster", "Average Years"),
    ]
    for column, title, ylabel in chart_specs:
        plt.figure(figsize=(10, 4))
        sns.barplot(
            x="cluster", y=column, data=profile,
            hue="cluster", palette="mako", legend=False,
        )
        plt.title(f"{title} ({prefix.replace('_', ' ').title()})")
        plt.xlabel("Cluster")
        plt.ylabel(ylabel)
        save_plot(f"{prefix}_avg_{column}.png")


# ---------------------------------------------------------------------------
# Persistence and inference
# ---------------------------------------------------------------------------

def save_artifacts(scaler: StandardScaler, pca: PCA, model: KMeans) -> None:
    """Persist the trained pipeline so new customers can be scored later."""
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "pca_model.pkl")
    joblib.dump(model, MODELS_DIR / "kmeans_model.pkl")
    print(f"\nArtifacts saved to {MODELS_DIR.relative_to(PROJECT_ROOT)}/")


def load_artifacts() -> tuple[StandardScaler, PCA, KMeans]:
    """Load the persisted pipeline back from disk."""
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    pca = joblib.load(MODELS_DIR / "pca_model.pkl")
    model = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    return scaler, pca, model


def score_new_customers(
    new_df: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
    model: KMeans,
    training_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the trained pipeline to a fresh batch of customers.

    Any columns not present at training time (e.g. ``customer_type``) are
    dropped so the input matches the scaler's expected shape exactly.
    """
    feature_df = new_df[training_columns].copy()
    scaled = scaler.transform(feature_df)
    pca_components = pca.transform(scaled)
    labels = model.predict(pca_components)
    return pca_components, labels


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full segmentation pipeline end-to-end."""
    ensure_directories()

    # ---- 1. Load and explore -------------------------------------------------
    print("=" * 60)
    print("STEP 1 - Load and explore customer data")
    print("=" * 60)
    customers = load_customers(DATA_FILE)
    print(f"Loaded {len(customers)} customers with columns: {list(customers.columns)}")
    plot_salary_vs_credit_limit(customers)

    # ---- 2. PCA + K-Means ----------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 - Standardise, reduce dimensionality, and cluster")
    print("=" * 60)
    training_columns = list(customers.columns)
    X_pca, scaler, pca = fit_pipeline(customers)

    plot_elbow(X_pca)
    kmeans_model, cluster_labels = fit_kmeans(X_pca, n_clusters=OPTIMAL_CLUSTERS)
    evaluate_clustering(X_pca, cluster_labels)
    plot_clusters_3d(X_pca, cluster_labels,
                     filename="clusters_3d.png",
                     title="3D Visualization of Customer Clusters")

    # ---- 3. Profile clusters -------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 - Profile and visualise each segment")
    print("=" * 60)
    customers_with_clusters = customers.copy()
    customers_with_clusters["cluster"] = cluster_labels
    cluster_profile = build_cluster_profile(customers_with_clusters)
    print("\nCluster profile (original customers):")
    print(cluster_profile.to_string(index=False))
    plot_profile_bars(cluster_profile, prefix="original")

    # ---- 4. Persist artifacts ------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 - Persist scaler, PCA, and K-Means artifacts")
    print("=" * 60)
    save_artifacts(scaler, pca, kmeans_model)

    # ---- 5. Score new customers ----------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 - Score a fresh batch of customers with the saved pipeline")
    print("=" * 60)
    scaler_l, pca_l, kmeans_l = load_artifacts()
    new_customers = load_customers(NEW_DATA_FILE)
    print(f"Loaded {len(new_customers)} new customers")

    new_pca, new_labels = score_new_customers(
        new_customers, scaler_l, pca_l, kmeans_l, training_columns
    )

    # Build comparable frames for the combined 3D overlay.
    original_plot_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2", "PCA3"])
    original_plot_df["Cluster"] = cluster_labels

    new_plot_df = pd.DataFrame(new_pca, columns=["PCA1", "PCA2", "PCA3"])
    new_plot_df["Cluster"] = new_labels

    plot_combined_3d(original_plot_df, new_plot_df, filename="new_customers_3d.png")

    new_customers_with_clusters = new_customers[training_columns].copy()
    new_customers_with_clusters["cluster"] = new_labels
    new_profile = build_cluster_profile(new_customers_with_clusters)
    print("\nCluster profile (new customers):")
    print(new_profile.to_string(index=False))
    plot_profile_bars(new_profile, prefix="new")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
