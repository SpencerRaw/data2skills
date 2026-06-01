"""Unsupervised data2skills: phenotype discovery from unlabeled data.

For one-class datasets (e.g., all SLE patients, no healthy controls),
we invert the skill paradigm:

    Supervised:  IF feature > threshold THEN predict CLASS
    Unsupervised: Patient subgroup A is characterized by elevated X, reduced Y

Skills become phenotype descriptions — structured, readable clinical knowledge
extracted purely from data patterns.

Methods:
    1. Cluster → Phenotype: k-means/GMM → per-cluster feature profiles → readable rules
    2. Contrastive: Compare against population priors (NHANES, lab reference ranges)
    3. Feature association: Discover co-occurring feature patterns (apriori-like)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import numpy as np


@dataclass
class Phenotype:
    """A discovered patient subgroup with interpretable description."""
    id: str
    name: str  # Human-readable name, e.g. "Renal-dominant SLE"
    size: int  # Number of patients
    fraction: float  # % of cohort
    
    # Characterizing features (what makes this subgroup distinct)
    elevated_features: list[tuple[str, float, float]] = field(default_factory=list)
    # (feature_name, subgroup_mean, global_mean)
    reduced_features: list[tuple[str, float, float]] = field(default_factory=list)
    
    # Quality
    silhouette: float = 0.0  # Cluster quality
    stability: float = 0.0    # Bootstrap stability
    
    @property
    def description(self) -> str:
        """Generate human-readable phenotype description."""
        lines = [f"## {self.name} (n={self.size}, {self.fraction:.0%} of cohort)"]
        
        if self.elevated_features:
            lines.append("\n**Elevated**:")
            for feat, mean, global_mean in self.elevated_features[:5]:
                fold = mean / max(global_mean, 1e-8)
                lines.append(f"  - {feat}: {mean:.1f} vs cohort avg {global_mean:.1f} ({fold:.1f}x)")
        
        if self.reduced_features:
            lines.append("\n**Reduced**:")
            for feat, mean, global_mean in self.reduced_features[:5]:
                lines.append(f"  - {feat}: {mean:.1f} vs cohort avg {global_mean:.1f}")
        
        return "\n".join(lines)


@dataclass
class UnsupervisedSkill:
    """A skill document extracted from unlabeled data.
    
    Contains discovered phenotypes + global feature importance.
    """
    name: str
    domain: str
    n_patients: int
    n_features: int
    phenotypes: list[Phenotype] = field(default_factory=list)
    global_insights: list[str] = field(default_factory=list)
    
    @property
    def text(self) -> str:
        """Render as readable markdown."""
        lines = [
            f"# {self.name}",
            f"Domain: {self.domain}",
            f"Patients: {self.n_patients} | Features: {self.n_features}",
            f"Phenotypes discovered: {len(self.phenotypes)}",
            "",
        ]
        
        if self.global_insights:
            lines.append("## Global Insights")
            for insight in self.global_insights:
                lines.append(f"- {insight}")
            lines.append("")
        
        lines.append("## Discovered Phenotypes")
        for pheno in self.phenotypes:
            lines.append(pheno.description)
            lines.append("")
        
        return "\n".join(lines)


def discover_phenotypes(
    X: np.ndarray,
    feature_names: list[str],
    n_clusters: int = None,
    method: str = "kmeans",
    random_state: int = 42,
) -> UnsupervisedSkill:
    """Discover patient phenotypes from unlabeled data.
    
    Args:
        X: (n_patients, n_features) array
        feature_names: list of feature names
        n_clusters: number of phenotypes to discover (auto if None)
        method: 'kmeans' or 'gmm'
    
    Returns:
        UnsupervisedSkill with discovered phenotypes
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    n_patients, n_features = X.shape
    
    # Auto-determine cluster count if not specified
    if n_clusters is None:
        n_clusters = _estimate_clusters(X, max_k=min(8, n_patients // 30))
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    if method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    labels = model.fit_predict(X_scaled)
    
    # Silhouette score
    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
    
    # Global means for fold-change computation
    global_means = X.mean(axis=0)
    global_stds = X.std(axis=0) + 1e-8
    
    # Extract phenotypes
    phenotypes = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_X = X[mask]
        cluster_mean = cluster_X.mean(axis=0)
        
        # Find distinguishing features (highest z-score deviation from global mean)
        deviations = np.abs(cluster_mean - global_means) / global_stds
        top_indices = np.argsort(deviations)[::-1]
        
        elevated = []
        reduced = []
        
        for idx in top_indices[:10]:
            feat_name = feature_names[idx]
            cm = cluster_mean[idx]
            gm = global_means[idx]
            
            if cm > gm + 0.5 * global_stds[idx]:
                elevated.append((feat_name, float(cm), float(gm)))
            elif cm < gm - 0.5 * global_stds[idx]:
                reduced.append((feat_name, float(cm), float(gm)))
        
        # Name the phenotype based on top features
        if elevated:
            name = f"Phenotype {cluster_id+1}: {elevated[0][0]}-dominant"
        elif reduced:
            name = f"Phenotype {cluster_id+1}: {reduced[0][0]}-deficient"
        else:
            name = f"Phenotype {cluster_id+1}"
        
        phenotypes.append(Phenotype(
            id=f"P{cluster_id+1}",
            name=name,
            size=int(mask.sum()),
            fraction=float(mask.sum()) / n_patients,
            elevated_features=elevated,
            reduced_features=reduced,
            silhouette=float(sil),
        ))
    
    # Sort by size (largest first)
    phenotypes.sort(key=lambda p: p.size, reverse=True)
    
    # Global insights
    insights = _generate_insights(X, feature_names, global_means, global_stds)
    
    return UnsupervisedSkill(
        name="SLE Phenotype Discovery",
        domain="medical_phenotyping",
        n_patients=n_patients,
        n_features=n_features,
        phenotypes=phenotypes,
        global_insights=insights,
    )


def discover_associations(
    X: np.ndarray,
    feature_names: list[str],
    min_support: float = 0.1,
    min_lift: float = 1.5,
    n_top: int = 20,
) -> list[dict]:
    """Discover co-occurring feature patterns.
    
    Finds pairs/triplets of features that co-occur at abnormal levels
    more often than expected by chance.
    
    Returns list of association rules with support, confidence, lift.
    """
    n_patients, n_features = X.shape
    global_medians = np.median(X, axis=0)
    
    # Binarize: above/below median
    binary = (X > global_medians).astype(int)
    
    rules = []
    
    # Pairwise associations
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Support: fraction where both are elevated
            support = np.mean(binary[:, i] & binary[:, j])
            if support < min_support:
                continue
            
            # Confidence: P(j high | i high)
            conf_i_given_j = np.sum(binary[:, i] & binary[:, j]) / max(np.sum(binary[:, i]), 1)
            
            # Lift: P(i,j) / (P(i) * P(j))
            p_i = np.mean(binary[:, i])
            p_j = np.mean(binary[:, j])
            lift = support / max(p_i * p_j, 1e-8)
            
            if lift >= min_lift and conf_i_given_j >= 0.5:
                rules.append({
                    "antecedent": feature_names[i],
                    "consequent": feature_names[j],
                    "support": float(support),
                    "confidence": float(conf_i_given_j),
                    "lift": float(lift),
                })
    
    # Sort by lift, take top
    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules[:n_top]


def _estimate_clusters(X: np.ndarray, max_k: int = 8) -> int:
    """Estimate optimal number of clusters using silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    if max_k < 2:
        return 2
    
    X_scaled = StandardScaler().fit_transform(X)
    
    best_k = 2
    best_score = -1
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k


def _generate_insights(
    X: np.ndarray,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> list[str]:
    """Generate global insights about the patient cohort."""
    insights = []
    
    # Most variable features
    cvs = stds / (np.abs(means) + 1e-8)
    top_variable = np.argsort(cvs)[::-1][:3]
    insights.append(
        f"Most variable features: {', '.join(feature_names[i] for i in top_variable)} "
        f"(CV: {', '.join(f'{cvs[i]:.1%}' for i in top_variable)})"
    )
    
    # Strongest pairwise correlations
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0)
    max_i, max_j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    insights.append(
        f"Strongest correlation: {feature_names[max_i]} ↔ {feature_names[max_j]} "
        f"(r={corr[max_i, max_j]:.2f})"
    )
    
    return insights
