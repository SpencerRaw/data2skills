from __future__ import annotations
"""Skill evaluator: apply skill to data and compute metrics."""

from dataclasses import dataclass, field
from typing import Optional
import re
import operator


@dataclass
class EvalResult:
    """Result of evaluating a skill on a dataset."""
    accuracy: float
    f1_macro: float
    predictions: list = field(default_factory=list)
    per_class: dict = field(default_factory=dict)
    coverage: float = 0.0  # Fraction of samples matched by at least one rule
    fallback_rate: float = 0.0  # Fraction requiring LLM fallback
    n_rules_used: int = 0


class SkillEvaluator:
    """Evaluate a skill by applying its rules to data points.
    
    The evaluator does NOT call an LLM. It uses rule engines (numeric comparisons)
    to evaluate rule conditions against data points.
    """
    
    # Operators supported in rule conditions
    OPS = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    
    def evaluate(
        self,
        skill,
        X,  # numpy array
        y,  # numpy array
        feature_names: Optional[list[str]] = None,
        label_map: Optional[dict] = None,
    ) -> EvalResult:
        """Evaluate skill on a dataset. Returns EvalResult.
        
        Args:
            label_map: Optional dict mapping string predictions to numeric labels.
                       e.g. {'malignant': 0, 'benign': 1}
        """
        import numpy as np
        
        if feature_names is None:
            feature_names = skill.features
        
        predictions = []
        matched = []
        
        for i in range(len(X)):
            pred, did_match = self._apply_skill_to_row(skill, X[i], feature_names)
            predictions.append(pred)
            matched.append(did_match)
        
        # Map string predictions to numeric if label_map provided
        if label_map:
            predictions = [label_map.get(p, -1) for p in predictions]
        
        matched_arr = np.array(matched)
        pred_arr = np.array(predictions)
        
        # Accuracy
        accuracy = np.mean(pred_arr == y)
        
        # Coverage
        coverage = np.mean(matched_arr)
        
        # F1 (macro)
        classes = np.unique(y)
        f1s = []
        per_class = {}
        for cls in classes:
            tp = np.sum((pred_arr == cls) & (y == cls))
            fp = np.sum((pred_arr == cls) & (y != cls))
            fn = np.sum((pred_arr != cls) & (y == cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1s.append(f1)
            per_class[int(cls)] = {"precision": precision, "recall": recall, "f1": f1}
        
        f1_macro = np.mean(f1s) if f1s else 0
        
        return EvalResult(
            accuracy=float(accuracy),
            f1_macro=float(f1_macro),
            predictions=pred_arr.tolist(),
            per_class=per_class,
            coverage=float(coverage),
            fallback_rate=float(1 - coverage),
            n_rules_used=len(skill.rules),
        )
    
    def _apply_skill_to_row(self, skill, x, feature_names: list[str]):
        """Apply skill rules to a single data point.
        
        Returns (prediction, matched: bool).
        If no rule matches, returns (None, False) — caller handles fallback.
        """
        feature_map = {fn: x[i] for i, fn in enumerate(feature_names)}
        
        # Collect matching rules
        matches = []
        for rule in skill.rules:
            if self._evaluate_condition(rule.condition, feature_map):
                matches.append(rule)
        
        if not matches:
            return (None, False)
        
        # Weighted vote by confidence
        votes = {}
        for rule in matches:
            correct, total = rule.support
            weight = rule.confidence * (correct / max(total, 1))
            votes[rule.prediction] = votes.get(rule.prediction, 0) + weight
        
        best_prediction = max(votes, key=votes.get)
        return (best_prediction, True)
    
    def _evaluate_condition(self, condition: str, feature_map: dict) -> bool:
        """Evaluate a rule condition against a feature dictionary.
        
        Supports simple conditions like "worst_radius > 17.5"
        and compound conditions with AND/OR (basic support).
        """
        condition = condition.strip()
        
        # Handle AND
        if " AND " in condition:
            parts = condition.split(" AND ")
            return all(self._evaluate_simple(p.strip(), feature_map) for p in parts)
        
        # Handle OR
        if " OR " in condition:
            parts = condition.split(" OR ")
            return any(self._evaluate_simple(p.strip(), feature_map) for p in parts)
        
        return self._evaluate_simple(condition, feature_map)
    
    def _evaluate_simple(self, condition: str, feature_map: dict) -> bool:
        """Evaluate a simple condition like 'feature > value'."""
        # Try each operator
        for op_str in [">=", "<=", "!=", "==", ">", "<"]:
            if op_str in condition:
                parts = condition.split(op_str, 1)
                if len(parts) != 2:
                    continue
                
                feat_name = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                except ValueError:
                    continue
                
                if feat_name in feature_map:
                    feat_value = float(feature_map[feat_name])
                    return self.OPS[op_str](feat_value, value)
        
        return False
