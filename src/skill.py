"""Skill representation: structured text document with metadata and rules."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Rule:
    """A single diagnostic rule in a skill."""
    id: str
    condition: str  # Human-readable condition, e.g. "worst_radius > 17.5 AND worst_concave_points > 0.14"
    prediction: str  # The prediction when condition is met
    confidence: float  # 0-1, estimated accuracy of this rule
    support: tuple = (0, 0)  # (correct, total) from training data
    source: str = "optimized"  # "seed", "optimized", "manual"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "condition": self.condition,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "support": list(self.support),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(
            id=d["id"],
            condition=d["condition"],
            prediction=d["prediction"],
            confidence=d["confidence"],
            support=tuple(d.get("support", [0, 0])),
            source=d.get("source", "optimized"),
        )


@dataclass
class Skill:
    """A skill document — the core artifact of data2skills."""
    name: str
    domain: str
    features: list[str]
    target: str
    rules: list[Rule] = field(default_factory=list)
    
    # Metadata
    version: int = 1
    performance: dict = field(default_factory=dict)
    
    # Internal
    _feature_stats: Optional[dict] = None  # mean/std/min/max per feature
    
    @property
    def text(self) -> str:
        """Render the skill as a human-readable markdown document."""
        lines = [
            f"# {self.name}",
            f"Domain: {self.domain}  ",
            f"Features: {', '.join(self.features)}  ",
            f"Target: {self.target}  ",
            "",
        ]
        
        if self.performance:
            lines.append("## Performance")
            for k, v in self.performance.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
        
        lines.append("## Rules")
        for rule in self.rules:
            lines.append(f"### Rule {rule.id}")
            lines.append(f"- **IF** {rule.condition}")
            lines.append(f"- **THEN** predict `{rule.prediction}`")
            lines.append(f"- **Confidence**: {rule.confidence:.2f}")
            correct, total = rule.support
            if total > 0:
                lines.append(f"- **Support**: {correct}/{total} ({correct/total:.1%})")
            lines.append("")
        
        return "\n".join(lines)
    
    def add_rule(self, rule: Rule):
        """Add a rule with auto-incrementing ID."""
        if not rule.id:
            rule.id = f"R{len(self.rules) + 1}"
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found and removed."""
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.id != rule_id]
        return len(self.rules) < before
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        for r in self.rules:
            if r.id == rule_id:
                return r
        return None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "features": self.features,
            "target": self.target,
            "version": self.version,
            "performance": self.performance,
            "rules": [r.to_dict() for r in self.rules],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        skill = cls(
            name=d["name"],
            domain=d["domain"],
            features=d["features"],
            target=d["target"],
            version=d.get("version", 1),
            performance=d.get("performance", {}),
        )
        skill.rules = [Rule.from_dict(r) for r in d.get("rules", [])]
        return skill
    
    def save(self, path: str):
        """Save skill to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Skill":
        """Load skill from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def seed_skill_from_data(
    name: str,
    domain: str,
    features: list[str],
    target: str,
    X,  # numpy array
    y,  # numpy array
    feature_names: list[str],
    target_names: list[str],
    n_rules: int = 5,
) -> Skill:
    """Generate an initial skill from data statistics.
    
    Creates simple threshold rules based on feature means and class separations.
    This is the "initialization" step of the optimization loop.
    """
    import numpy as np
    
    skill = Skill(
        name=name,
        domain=domain,
        features=feature_names,
        target=target,
    )
    
    # Compute per-class statistics
    classes = np.unique(y)
    
    for cls in classes:
        mask = y == cls
        if mask.sum() == 0:
            continue
        
        cls_name = target_names[int(cls)] if target_names else str(cls)
        X_cls = X[mask]
        
        # Find the most discriminative features for this class
        # Simple heuristic: features where this class deviates most from global mean
        global_mean = X.mean(axis=0)
        global_std = X.std(axis=0) + 1e-8
        cls_mean = X_cls.mean(axis=0)
        
        deviations = np.abs(cls_mean - global_mean) / global_std
        top_features = np.argsort(deviations)[-3:][::-1]  # top 3 discriminative features
        
        for i, feat_idx in enumerate(top_features[:n_rules // len(classes)]):
            feat_name = feature_names[feat_idx]
            threshold = cls_mean[feat_idx]
            direction = ">" if cls_mean[feat_idx] > global_mean[feat_idx] else "<"
            
            rule = Rule(
                id=f"R{len(skill.rules) + 1}",
                condition=f"{feat_name} {direction} {threshold:.2f}",
                prediction=cls_name,
                confidence=0.6,  # initial low confidence
                support=(int(mask.sum()), int(mask.sum())),
                source="seed",
            )
            skill.rules.append(rule)
    
    skill._feature_stats = {
        "mean": {fn: float(X[:, i].mean()) for i, fn in enumerate(feature_names)},
        "std": {fn: float(X[:, i].std()) for i, fn in enumerate(feature_names)},
    }
    
    return skill
