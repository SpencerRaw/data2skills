"""Experiment: unsupervised phenotype discovery on synthetic + real data.

Simulates a one-class SLE-like dataset and demonstrates phenotype extraction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import make_blobs

from src.unsupervised import discover_phenotypes, discover_associations


def create_synthetic_sle_data(n_patients=500, random_state=42):
    """Create synthetic data mimicking SLE patient subgroups.
    
    Generates 4 phenotypes:
    - Renal-dominant: high creatinine, high proteinuria, high anti-dsDNA
    - Hematological: low WBC, low platelets, high ANA
    - Musculoskeletal: high CRP, high ESR, joint involvement
    - Mild: near-normal labs, low disease activity
    """
    np.random.seed(random_state)
    
    n_per_group = n_patients // 4
    
    feature_names = [
        "anti_dsDNA", "ANA_titer", "C3", "C4",
        "creatinine", "proteinuria", "eGFR",
        "WBC", "platelets", "hemoglobin",
        "CRP", "ESR",
        "SLEDAI_score",
    ]
    
    n_features = len(feature_names)
    X_list = []
    labels = []
    
    # Group 1: Renal-dominant
    X1 = np.random.normal(loc=0, scale=1, size=(n_per_group, n_features))
    X1[:, 0] += 2.0  # anti-dsDNA high
    X1[:, 4] += 2.5  # creatinine high
    X1[:, 5] += 2.0  # proteinuria high
    X1[:, 6] -= 1.5  # eGFR low
    X_list.append(X1)
    labels.extend([0] * n_per_group)
    
    # Group 2: Hematological
    X2 = np.random.normal(loc=0, scale=1, size=(n_per_group, n_features))
    X2[:, 1] += 2.5  # ANA high
    X2[:, 7] -= 2.0  # WBC low
    X2[:, 8] -= 2.5  # platelets low
    X2[:, 12] += 1.5 # SLEDAI moderate
    X_list.append(X2)
    labels.extend([1] * n_per_group)
    
    # Group 3: Musculoskeletal/inflammatory
    X3 = np.random.normal(loc=0, scale=1, size=(n_per_group, n_features))
    X3[:, 10] += 2.5  # CRP high
    X3[:, 11] += 2.0  # ESR high
    X3[:, 12] += 1.0  # SLEDAI mild
    X_list.append(X3)
    labels.extend([2] * n_per_group)
    
    # Group 4: Mild
    X4 = np.random.normal(loc=0, scale=1, size=(n_per_group, n_features))
    X4[:, 2] += 1.5  # C3 normal-high
    X4[:, 3] += 1.5  # C4 normal-high
    X4[:, 12] -= 1.5 # SLEDAI low
    X_list.append(X4)
    labels.extend([3] * n_per_group)
    
    X = np.vstack(X_list)
    y = np.array(labels)
    
    return X, y, feature_names


def main():
    print("=" * 60)
    print("Unsupervised Phenotype Discovery — Synthetic SLE Cohort")
    print("=" * 60)
    
    X, y_true, feature_names = create_synthetic_sle_data(n_patients=500)
    print(f"\nCohort: {X.shape[0]} patients, {X.shape[1]} features")
    
    # Discover phenotypes
    print("\n--- Phenotype Discovery ---")
    skill = discover_phenotypes(
        X, feature_names,
        n_clusters=4,
        method="kmeans",
    )
    
    # Evaluate against ground truth
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y_true, [int(p.id[1])-1 for p in skill.phenotypes for _ in range(p.size)])
    print(f"  Adjusted Rand Index vs ground truth: coming soon (needs label assignment)")
    
    for pheno in skill.phenotypes:
        print(f"\n  {pheno.name} (n={pheno.size}, {pheno.fraction:.0%})")
        if pheno.elevated_features:
            top = pheno.elevated_features[:3]
            print(f"    ↑ {', '.join(f'{f}({m:.1f})' for f, m, _ in top)}")
        if pheno.reduced_features:
            top = pheno.reduced_features[:3]
            print(f"    ↓ {', '.join(f'{f}({m:.1f})' for f, m, _ in top)}")
    
    # Feature associations
    print("\n--- Feature Associations ---")
    rules = discover_associations(X, feature_names, min_support=0.1, min_lift=1.5)
    for rule in rules[:10]:
        print(f"  {rule['antecedent']} ↑ → {rule['consequent']} ↑ "
              f"(lift={rule['lift']:.2f}, conf={rule['confidence']:.2f})")
    
    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "skills", "sle_phenotypes.md")
    with open(out_path, "w") as f:
        f.write(skill.text)
    print(f"\nPhenotype report saved to {out_path}")


if __name__ == "__main__":
    main()
