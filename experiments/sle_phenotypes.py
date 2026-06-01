"""SLE Phenotype Discovery — Real Patient Data.

Unsupervised phenotype extraction from 1049 SLE patient records.
Discovers clinically meaningful patient subgroups with interpretable rules.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.unsupervised import discover_phenotypes, discover_associations


def load_sle_data():
    """Load and preprocess the anonymized SLE dataset."""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "sle_safe.csv"))
    
    # Select numeric medical features (skip demographics, dates, categorical)
    skip_patterns = ['年龄', '入院', '性别', '月经', '流产', '肾活检', 'ANA滴度', 'ds-DNA滴度']
    
    numeric_cols = []
    for col in df.columns:
        if any(s in col for s in skip_patterns):
            continue
        if df[col].dtype in ['float64', 'int64']:
            numeric_cols.append(col)
    
    print(f"Selected {len(numeric_cols)} numeric features from {len(df.columns)} columns")
    
    X = df[numeric_cols].values
    feature_names = [c.strip().replace('\n', ' ') for c in numeric_cols]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    return X, feature_names, df


def main():
    print("=" * 70)
    print("SLE Phenotype Discovery — Real Patient Data")
    print("=" * 70)
    
    X, feature_names, df = load_sle_data()
    n_patients, n_features = X.shape
    print(f"\nCohort: {n_patients} patients, {n_features} features")
    
    # Discover phenotypes
    print("\n--- Discovering Phenotypes ---")
    skill = discover_phenotypes(
        X, feature_names,
        n_clusters=None,  # auto-detect
        method="kmeans",
    )
    
    # Print phenotypes
    for pheno in skill.phenotypes:
        print(f"\n{'─'*60}")
        print(f"  {pheno.name}")
        print(f"  Size: {pheno.size} patients ({pheno.fraction:.1%} of cohort)")
        
        if pheno.elevated_features:
            print(f"  🔺 Elevated (top 5):")
            for feat, mean, global_mean in pheno.elevated_features[:5]:
                fold = mean / max(global_mean, 1e-8)
                print(f"      {feat}: {mean:.1f} vs cohort avg {global_mean:.1f} ({fold:.1f}x)")
        
        if pheno.reduced_features:
            print(f"  🔻 Reduced (top 5):")
            for feat, mean, global_mean in pheno.reduced_features[:5]:
                print(f"      {feat}: {mean:.1f} vs cohort avg {global_mean:.1f}")
    
    # Feature associations
    print(f"\n{'='*70}")
    print("Feature Associations (co-occurring patterns)")
    print(f"{'='*70}")
    
    rules = discover_associations(X, feature_names, min_support=0.15, min_lift=1.3, n_top=15)
    
    for rule in rules:
        print(f"  {rule['antecedent'][:30]:<30} ↑ → {rule['consequent'][:30]:<30} ↑  "
              f"lift={rule['lift']:.2f}  supp={rule['support']:.2f}")
    
    # Save phenotype report
    out_path = os.path.join(os.path.dirname(__file__), "..", "skills", "sle_phenotypes_real.md")
    with open(out_path, "w") as f:
        f.write(skill.text)
        f.write("\n\n## Feature Associations\n\n")
        for rule in rules:
            f.write(f"- **{rule['antecedent']}** ↑ → **{rule['consequent']}** ↑  "
                   f"(lift={rule['lift']:.2f}, support={rule['support']:.2f})\n")
    
    print(f"\nPhenotype report saved to {out_path}")
    
    # Summary for paper
    print(f"\n{'='*70}")
    print("PAPER SUMMARY")
    print(f"{'='*70}")
    print(f"Cohort: {n_patients} SLE patients")
    print(f"Phenotypes discovered: {len(skill.phenotypes)}")
    for pheno in skill.phenotypes:
        top_feat = pheno.elevated_features[0][0] if pheno.elevated_features else \
                   pheno.reduced_features[0][0] if pheno.reduced_features else "unknown"
        print(f"  {pheno.name}: n={pheno.size} ({pheno.fraction:.0%}), "
              f"top feature: {top_feat}")


if __name__ == "__main__":
    main()
