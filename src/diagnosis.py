"""LLM-based failure diagnosis → text gradient generation.

This is the "backward pass" of text-gradient descent.
Given a skill and its failures, produces an edit plan (text gradient).

The rule-based diagnoser provides a baseline. For full performance,
an LLM-powered diagnoser (like SkillOpt/SkillGrad) should be used.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional
import numpy as np


class FailureDiagnoser:
    """Diagnoses skill failures and produces text gradients (edit plans).
    
    Two modes:
    1. Rule-based: Statistical threshold optimization (no LLM needed)
    2. LLM-powered: Uses an LLM for semantic diagnosis (higher quality)
    """
    
    def __init__(
        self,
        model: str = "deepseek-v4",
        feature_names: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.target_names = target_names or []
    
    def diagnose(
        self,
        skill,
        failures: list,  # (x, y_true, y_pred)
        momentum: dict,
    ) -> dict:
        """Generate a text gradient (edit plan) from failures.
        
        Uses statistical analysis of failures to propose:
        - New rules for uncovered patterns
        - Threshold tightening for inaccurate rules
        - Rule deletion for harmful rules
        """
        if not failures:
            return {"diagnosis": "No failures", "edits": [], "diagnosis_patterns": []}
        
        return self._statistical_diagnose(skill, failures, momentum)
    
    def meta_diagnose(self, skill, rejected_history: list, momentum: dict) -> dict:
        """Meta-review: rethink strategy when many edits are rejected."""
        top_patterns = sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "diagnosis": "Meta-review: consolidating rules",
            "diagnosis_patterns": [p for p, _ in top_patterns],
            "edits": [{
                "type": "meta_consolidate",
                "reason": f"Pattern '{p}' at {c:.1f} — merging overlapping rules",
                "target": "all_rules",
            } for p, c in top_patterns],
        }
    
    def _statistical_diagnose(self, skill, failures: list, momentum: dict) -> dict:
        """Statistical threshold optimization from failure patterns.
        
        For each failing class, finds features where failed examples
        deviate from the skill's existing decision boundaries, then
        proposes specific threshold adjustments.
        """
        edits = []
        patterns = []
        feature_names = self.feature_names
        n_features = len(feature_names)
        
        # Group failures by true class
        by_class = defaultdict(list)
        for x, y_true, y_pred in failures:
            by_class[int(y_true)].append(x)
        
        for cls, examples in by_class.items():
            if len(examples) < 2:
                continue
            
            X_fail = np.array(examples)
            cls_name = self.target_names[int(cls)] if self.target_names else str(cls)
            
            # Find features where failed examples concentrate
            for feat_idx in range(n_features):
                feat_vals = X_fail[:, feat_idx]
                feat_name = feature_names[feat_idx]
                
                feat_mean = feat_vals.mean()
                feat_std = feat_vals.std() + 1e-8
                
                # Check if existing rules cover this feature
                covered = False
                for rule in skill.rules:
                    if feat_name in rule.condition:
                        covered = True
                        break
                
                if not covered and len(examples) >= 3:
                    # Propose new rule for uncovered feature pattern
                    pattern = f"uncovered_feature_{feat_name}"
                    patterns.append(pattern)
                    
                    # Determine direction: are failures above or below global mean?
                    direction = ">" if feat_mean > 0 else "<"
                    threshold = abs(feat_mean)
                    
                    edits.append({
                        "type": "add",
                        "target": "",
                        "reason": f"{len(examples)} failures for class '{cls_name}' — "
                                 f"feature '{feat_name}' not covered by any rule",
                        "content": f"IF {feat_name} {direction} {threshold:.3f} "
                                  f"THEN predict {cls_name}",
                    })
                elif covered:
                    # Tighten existing threshold
                    pattern = f"class_{cls}_misclassification"
                    patterns.append(pattern)
                    edits.append({
                        "type": "modify",
                        "target": "",
                        "reason": f"Tighten threshold for class {cls_name}",
                        "suggestion": "tighten_threshold",
                    })
        
        # Find harmful rules (rules that cause more errors than correct predictions)
        for rule in skill.rules:
            if rule.confidence < 0.3 and rule.support[1] > 5:
                patterns.append(f"low_confidence_rule_{rule.id}")
                edits.append({
                    "type": "delete",
                    "target": rule.id,
                    "reason": f"Rule {rule.id} confidence too low ({rule.confidence:.2f})",
                })
        
        return {
            "diagnosis": f"Found {len(failures)} failures across {len(by_class)} classes",
            "diagnosis_patterns": list(set(patterns))[:10],
            "edits": edits[:5],
        }
