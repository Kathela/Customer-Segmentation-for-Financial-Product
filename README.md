# Customer Segmentation for Financial Products

### Unsupervised ML · PCA · K-Means · Python

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end customer segmentation pipeline that turns raw financial
behaviour data into actionable marketing segments and persists a
reusable scoring pipeline for new customers.

---

## Project overview

Financial institutions typically broadcast the same offers to their
entire customer base, which produces low conversion and wasted spend.
This project builds an **unsupervised segmentation model** that groups
customers by their financial profile so that marketing, retention, and
risk teams can target each segment with the right product.

The full workflow — data prep, dimensionality reduction, clustering,
validation, profiling, persistence, and inference on new customers — is
implemented as a single reproducible Python script.

## Business problem

- Generic campaigns convert poorly and waste marketing budget.
- The institution has rich behavioural signals (salary, credit limit,
  tenure, credit type, prior product responses) but no structured way
  to act on them.
- **Goal:** discover distinct customer archetypes and operationalise
  the model so new customers can be scored on arrival.

## Methodology

| Step | Technique | Why it matters |
|------|-----------|----------------|
| Feature scaling | `StandardScaler` | Salary and credit limit are orders of magnitude larger than the binary credit-type flags, so unscaled distances would be dominated by the monetary axes. |
| Dimensionality reduction | `PCA(n_components=0.8)` | Retains 80% of variance, denoises the space, and lets PCA decide how many axes to keep. |
| Cluster count selection | Elbow method on inertia | The curve inflects cleanly at k=5. |
| Segmentation | `KMeans` (k=5, fixed seed) | Reproducible labels for downstream CRM systems. |
| Validation | Silhouette · Calinski-Harabasz · Davies-Bouldin | Three complementary metrics so no single index can mask a weak partition. |
| Persistence | `joblib` | Scaler, PCA, and K-Means are saved as `.pkl` files so new customers can be scored without re-training. |

## Key findings

- **5 distinct customer profiles** emerge from the data:
  - **Cluster 0 — New, high-income, low credit exposure.** Recently
    acquired and high-earning, but with low credit-card limits.
    Opportunity to grow share-of-wallet.
  - **Cluster 1 — Loyal, low-income, high credit, *zero* products
    offered.** A clear missed-opportunity segment.
  - **Cluster 2 — Long-tenured, mid-income, high credit, fully
    addressed.** Premium retention target.
  - **Cluster 3 — New, low-income, type-2 credit.** Higher-risk profile
    that warrants monitoring and financial-education outreach.
  - **Cluster 4 — Long-tenured, low-income, type-2 credit, highly
    receptive.** Best ROI for offer-driven campaigns.
- Clusters are well-separated: silhouette ≈ **0.41**, Calinski-Harabasz
  ≈ **338**, Davies-Bouldin ≈ **0.86**.
- The saved pipeline scores **50 new customers** without re-training,
  confirming the system is production-ready.

## Project structure

```
.
├── customer_segmentation.py     # End-to-end pipeline (entry point)
├── customers.csv                # Training dataset
├── new_customers.csv            # Hold-out batch scored at the end
├── requirements.txt             # Pinned dependencies
├── plots/                       # All generated PNG charts
├── models/                      # Persisted scaler, PCA, K-Means (created at runtime)
├── Customer_Segmentation_for_Financial_Product.ipynb  # Original exploratory notebook
└── README.md
```

## How to run

```bash
git clone https://github.com/Kathela/Customer-Segmentation-for-Financial-Product
cd Customer-Segmentation-for-Financial-Product
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python customer_segmentation.py
```

Running the script:

1. Creates `plots/` and `models/` if missing.
2. Trains the full pipeline on `customers.csv`.
3. Saves all charts (elbow, 3D cluster maps, per-cluster bar charts) to `plots/`.
4. Saves the fitted `scaler`, `pca`, and `kmeans` models to `models/`.
5. Reloads those artifacts and scores `new_customers.csv`.
6. Prints the cluster profile tables and quality metrics to stdout.

## Tech stack

- **Python 3.10+**
- **Data:** pandas, NumPy
- **Modelling:** scikit-learn — `StandardScaler`, `PCA`, `KMeans`,
  `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`
- **Visualisation:** matplotlib (including 3D projection), seaborn
- **Persistence:** joblib

## Variables used

`salary` · `credit_card_limit` · `customer_tenure`
· `credit_type_1` · `credit_type_2` · `product_offered`

## Business impact

Each cluster maps to a distinct archetype with its own risk profile and
product affinity. Marketing teams can use these segments to prioritise
high-value customers, design retention strategies, and cut campaign
costs by suppressing irrelevant outreach. Because the scoring pipeline
is persisted, new customers can be classified in seconds without
re-training.
