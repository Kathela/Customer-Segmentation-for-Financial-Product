# Customer Segmentation for Financial Products
### Unsupervised ML · PCA · K-Means · Python

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Problem statement

Financial companies often apply generic campaigns to their entire customer base, resulting in low conversion rates and wasted marketing spend. This project builds a data-driven segmentation model to identify distinct customer profiles based on financial behavior, enabling targeted product recommendations.

## Approach

| Step | Technique | Result |
|------|-----------|--------|
| Dimensionality reduction | PCA | Retained 80% of variance |
| Optimal clusters | Elbow method | k=5 identified |
| Segmentation | K-Means clustering | 5 distinct customer profiles |
| Validation | Silhouette Score + Calinski-Harabasz + Davies-Bouldin | Clusters well-defined and separated |

## Key results

- Identified **5 distinct customer profiles** based on salary, credit limit, tenure, and credit type
- PCA reduced feature space while retaining **80% of total variance**
- Model validated on **50 new customers** using the same scaler and PCA pipeline
- Reusable system allows real-time classification of new customers

## Variables used

`salary` · `credit_card_limit` · `customer_tenure` · `credit_type` · `product_offered`

## Business impact

Each cluster maps to a customer archetype with different risk profiles and product affinity. Marketing teams can use these segments to prioritize high-value customers, design retention strategies, and reduce campaign costs by eliminating irrelevant outreach.

## How to run

```bash
git clone https://github.com/Kathela/Customer-Segmentation-for-Financial-Product
cd Customer-Segmentation-for-Financial-Product
pip install -r requirements.txt
jupyter notebook Customer_Segmentation_for_Financial_Product.ipynb
```

## Tech stack

- **Python** — pandas, NumPy, matplotlib, seaborn
- **scikit-learn** — PCA, KMeans, silhouette_score, Calinski-Harabasz, Davies-Bouldin
- **Jupyter Notebook** — analysis and visualization
