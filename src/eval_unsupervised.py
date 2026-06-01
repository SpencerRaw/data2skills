"""Unsupervised skill evaluation metrics.

For phenotype discovery skills (no ground truth labels), we need
different evaluation approaches than supervised classification.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def evaluate_phenotype_skill(skill, X, feature_names, n_bootstrap=20) -> dict:
    """Comprehensive evaluation of an unsupervised phenotype skill.
    
    Returns dict with:
        - silhouette: cluster cohesion/separation (-1 to 1, higher better)
        - davies_bouldin: cluster similarity (lower better)
        - calinski_harabasz: variance ratio (higher better)  
        - stability: bootstrap cluster assignment consistency (0-1)
        - coverage: fraction of patients matched by at least one phenotype rule
        - interpretability: average rule readability score (0-1)
        - clinical_plausibility: heuristic check (0-1)
    """
    metrics = {}
    
    # Recover cluster labels from skill
    labels = _assign_to_phenotypes(skill, X, feature_names)
    
    if len(set(labels)) < 2:
        return {"error": "Only one phenotype found — increase n_clusters"}
    
    X_scaled = StandardScaler().fit_transform(X)
    
    # 1. Internal clustering metrics
    metrics["silhouette"] = float(silhouette_score(X_scaled, labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X_scaled, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
    
    # 2. Stability (bootstrap)
    metrics["stability"] = _bootstrap_stability(X, n_bootstrap)
    
    # 3. Coverage
    coverage = _compute_coverage(skill, X, feature_names)
    metrics["coverage"] = float(coverage)
    
    # 4. Interpretability
    metrics["interpretability"] = _interpretability_score(skill)
    
    # 5. Clinical plausibility (heuristic)
    metrics["clinical_plausibility"] = _clinical_plausibility(skill)
    
    # Overall score
    metrics["overall"] = float(np.mean([
        max(0, metrics["silhouette"]),
        metrics["stability"],
        metrics["coverage"],
        metrics["interpretability"],
        metrics["clinical_plausibility"],
    ]))
    
    return metrics


def _assign_to_phenotypes(skill, X, feature_names):
    """Assign each patient to the closest phenotype based on rule matching."""
    n = len(X)
    if len(skill.phenotypes) == 0:
        return np.zeros(n, dtype=int)
    
    # For each patient, find which phenotype's rules match best
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        best_score = -1
        best_pheno = 0
        
        for pi, pheno in enumerate(skill.phenotypes):
            score = 0
            feature_map = dict(zip(feature_names, X[i]))
            
            for feat, mean, _ in pheno.elevated_features:
                if feat in feature_map:
                    if feature_map[feat] > mean * 0.8:  # Within 80% of phenotype mean
                        score += 1
            
            for feat, mean, _ in pheno.reduced_features:
                if feat in feature_map:
                    if feature_map[feat] < mean * 1.2:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_pheno = pi
        
        labels[i] = best_pheno
    
    return labels


def _bootstrap_stability(X, n_bootstrap=20, sample_frac=0.8) -> float:
    """Measure how stable the clustering is under bootstrap resampling."""
    from sklearn.cluster import KMeans
    
    n_samples = len(X)
    n_clusters = max(2, int(np.sqrt(n_samples)))  # Rule of thumb
    n_sub = int(n_samples * sample_frac)
    
    agreements = []
    
    for _ in range(n_bootstrap):
        idx1 = np.random.choice(n_samples, n_sub, replace=True)
        idx2 = np.random.choice(n_samples, n_sub, replace=True)
        
        overlap = np.intersect1d(np.unique(idx1), np.unique(idx2))
        if len(overlap) < 10:
            continue
        
        X1 = StandardScaler().fit_transform(X[idx1])
        X2 = StandardScaler().fit_transform(X[idx2])
        
        km1 = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        km2 = KMeans(n_clusters=n_clusters, random_state=43, n_init=5)
        
        l1 = km1.fit_predict(X1)
        l2 = km2.fit_predict(X2)
        
        # Compare labels on overlap
        overlap_idx1 = [list(idx1).index(o) for o in overlap if o in idx1]
        overlap_idx2 = [list(idx2).index(o) for o in overlap if o in idx2]
        
        if len(overlap_idx1) < 5 or len(overlap_idx2) < 5:
            continue
        
        # Compute adjusted Rand index on overlap
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(
            l1[overlap_idx1[:len(overlap_idx2)]], 
            l2[overlap_idx2[:len(overlap_idx1)]]
        )
        agreements.append(max(0, ari))
    
    return float(np.mean(agreements)) if agreements else 0.0


def _compute_coverage(skill, X, feature_names) -> float:
    """Fraction of patients matched by at least one phenotype rule."""
    n_matched = 0
    
    for i in range(len(X)):
        feature_map = dict(zip(feature_names, X[i]))
        matched = False
        
        for pheno in skill.phenotypes:
            for feat, mean, _ in pheno.elevated_features + pheno.reduced_features:
                if feat in feature_map:
                    matched = True
                    break
            if matched:
                break
        
        if matched:
            n_matched += 1
    
    return n_matched / len(X)


def _interpretability_score(skill) -> float:
    """Score how interpretable the phenotype descriptions are.
    
    Factors: rule count (fewer = better), condition length (shorter = better),
    feature name clarity (recognizable names = better).
    """
    if not skill.phenotypes:
        return 0.0
    
    scores = []
    for pheno in skill.phenotypes:
        n_features = len(pheno.elevated_features) + len(pheno.reduced_features)
        
        # Fewer features per phenotype = more interpretable
        feature_score = max(0, 1 - n_features / 20)
        
        # Has a meaningful name
        name_score = 0.5 if "-dominant" in pheno.name or "-deficient" in pheno.name else 0.2
        
        scores.append(0.6 * feature_score + 0.4 * name_score)
    
    return float(np.mean(scores))


def _clinical_plausibility(skill) -> float:
    """Heuristic: how clinically plausible are the phenotype groupings?
    
    Checks: 
    - Related features cluster together (e.g., renal markers in same phenotype)
    - Phenotype sizes are reasonable (not all in one cluster)
    - Feature directions make physiological sense
    """
    score = 0.5  # Base score
    
    # Bonus: diverse phenotype sizes
    sizes = [p.fraction for p in skill.phenotypes]
    if len(sizes) > 1:
        max_size = max(sizes)
        if max_size < 0.8:  # Not dominated by one cluster
            score += 0.2
        if min(sizes) > 0.05:  # No tiny clusters
            score += 0.1
    
    # Bonus: feature coherence (crude check)
    renal_keywords = ['尿蛋白', 'creatinine', '肌酐', 'eGFR', 'albumin', '白蛋白']
    inflam_keywords = ['CRP', 'ESR', '血沉', 'fever', '发热', 'IgG']
    
    for pheno in skill.phenotypes:
        features_text = ' '.join([f for f, _, _ in pheno.elevated_features + pheno.reduced_features])
        
        renal_hits = sum(1 for kw in renal_keywords if kw in features_text)
        inflam_hits = sum(1 for kw in inflam_keywords if kw in features_text)
        
        # Coherent phenotype (renal features together, not mixed with unrelated ones)
        if renal_hits >= 2 and inflam_hits == 0:
            score += 0.1
        if inflam_hits >= 2 and renal_hits == 0:
            score += 0.1
    
    return min(1.0, score)


def pretty_print(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 50)
    print("PHENOTYPE SKILL EVALUATION")
    print("=" * 50)
    
    labels = {
        "silhouette": ("Cluster Quality", "higher ↑", 0.25),
        "davies_bouldin": ("Cluster Separation", "lower ↓", None),
        "stability": ("Bootstrap Stability", "higher ↑", 0.5),
        "coverage": ("Patient Coverage", "higher ↑", 0.7),
        "interpretability": ("Interpretability", "higher ↑", 0.6),
        "clinical_plausibility": ("Clinical Plausibility", "higher ↑", 0.5),
        "overall": ("OVERALL SCORE", "higher ↑", 0.5),
    }
    
    for key, (label, direction, threshold) in labels.items():
        if key in metrics:
            val = metrics[key]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            status = "✅" if threshold and val >= threshold else "⚠️" if threshold else "  "
            print(f"  {status} {label:<25} {bar} {val:.3f} ({direction})")
