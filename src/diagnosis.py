"""LLM-based failure diagnosis → text gradient generation.

This is the "backward pass" of text-gradient descent.
Given a skill and its failures, produces an edit plan (text gradient).
"""

from typing import Optional


class FailureDiagnoser:
    """Uses an LLM to diagnose skill failures and produce text gradients.
    
    The diagnoser is the key differentiator of data2skills vs traditional ML:
    instead of computing numerical gradients, it generates textual edit plans
    that explain WHAT went wrong and HOW to fix it.
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
        failures: list,  # list of (x, y_true, y_pred)
        momentum: dict,
    ) -> dict:
        """Generate a text gradient (edit plan) from failures.
        
        This is the core LLM call. It:
        1. Shows the LLM the current skill
        2. Shows representative failures
        3. Asks for a structured edit plan
        
        The edit plan is the "text gradient" — it indicates the direction
        and magnitude of needed changes.
        """
        # Build the diagnosis prompt
        prompt = self._build_diagnosis_prompt(skill, failures, momentum)
        
        # Call LLM (in production, this would use the actual LLM API)
        # For now, we provide a rule-based fallback that generates
        # simple threshold adjustments
        edit_plan = self._rule_based_diagnose(skill, failures)
        
        return edit_plan
    
    def meta_diagnose(
        self,
        skill,
        rejected_history: list,
        momentum: dict,
    ) -> dict:
        """Meta-review: rethink strategy when many edits are rejected.
        
        Called when the rejected buffer exceeds threshold.
        Generates a higher-level edit plan that may restructure the skill.
        """
        # Simplified: merge recurring momentum patterns into larger edits
        top_patterns = sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "diagnosis": "Meta-review: restructuring skill based on accumulated feedback",
            "diagnosis_patterns": [p for p, _ in top_patterns],
            "edits": [
                {
                    "type": "meta_consolidate",
                    "reason": f"Pattern '{p}' appeared {c:.1f} times — consolidating related rules",
                    "target": "all_rules",
                }
                for p, c in top_patterns
            ],
        }
    
    def _build_diagnosis_prompt(
        self, skill, failures: list, momentum: dict
    ) -> str:
        """Build the prompt for the LLM diagnosis call."""
        # Feature descriptions
        feature_desc = "\n".join(
            f"  - {fn}" for fn in self.feature_names
        )
        
        # Current skill rules
        rules_text = "\n".join(
            f"  Rule {r.id}: IF {r.condition} THEN predict {r.prediction} "
            f"(confidence={r.confidence:.2f}, support={r.support[0]}/{r.support[1]})"
            for r in skill.rules
        )
        
        # Failure examples
        failure_text = "\n".join(
            f"  Example {i}: features={dict(zip(self.feature_names, x))}, "
            f"true={true}, predicted={pred}"
            for i, (x, true, pred) in enumerate(failures[:8])
        )
        
        # Momentum
        momentum_text = "\n".join(
            f"  - '{p}': {c:.1f} occurrences"
            for p, c in sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:5]
        ) if momentum else "  (no momentum yet)"
        
        return f"""You are a skill optimizer. Analyze the following failures and propose precise edits.

FEATURES:
{feature_desc}

CURRENT SKILL RULES:
{rules_text}

RECENT FAILURES (true ≠ predicted):
{failure_text}

MOMENTUM (recurring failure patterns):
{momentum_text}

Propose an edit plan with add/delete/modify operations.
Each edit must be specific and bounded.
"""
    
    def _rule_based_diagnose(self, skill, failures: list) -> dict:
        """Fallback: rule-based diagnosis without LLM.
        
        Generates simple threshold adjustments based on failure statistics.
        In production, this would be replaced by an actual LLM call.
        """
        if not failures:
            return {"diagnosis": "No failures to diagnose", "edits": [], "diagnosis_patterns": []}
        
        import numpy as np
        
        # Analyze failures: which feature thresholds need adjustment?
        edits = []
        patterns = []
        
        # Group failures by true class
        from collections import defaultdict
        by_class = defaultdict(list)
        for x, true, pred in failures:
            by_class[int(true)].append(x)
        
        for cls, examples in by_class.items():
            if len(examples) < 2:
                continue
            
            X_fail = np.array(examples)
            
            # Find features where failed examples differ most from the skill's existing thresholds
            for rule in skill.rules:
                if str(cls) not in rule.prediction and rule.prediction not in str(cls):
                    continue
                
                pattern = f"class_{cls}_misclassification"
                patterns.append(pattern)
                
                # Propose threshold tightening
                edits.append({
                    "type": "modify",
                    "target": rule.id,
                    "reason": f"{len(examples)} failures for class {cls}",
                    "suggestion": "tighten_threshold",
                })
        
        return {
            "diagnosis": f"Analyzed {len(failures)} failures across {len(by_class)} classes",
            "diagnosis_patterns": list(set(patterns)),
            "edits": edits[:5],  # Limit edits per batch
        }
