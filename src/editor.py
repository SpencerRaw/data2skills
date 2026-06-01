"""Skill editor: applies bounded add/delete/modify edits to a skill document."""

import copy
from typing import Optional
from .skill import Skill, Rule


class SkillEditor:
    """Applies edit plans to skills with bounded operations.
    
    This is the "weight update" step of text-gradient descent.
    Each edit is an atomic operation on the skill text.
    """
    
    def apply_edits(
        self,
        skill: Skill,
        edit_plan: dict,
        max_adds: int = 2,
        max_deletes: int = 1,
        max_modifies: int = 2,
    ) -> Skill:
        """Apply an edit plan to a skill.
        
        Edits are bounded: at most max_adds adds, max_deletes deletes, 
        max_modifies modifies per update step. This is the textual analogue
        of a learning rate — it controls how aggressive each update is.
        """
        edits = edit_plan.get("edits", [])
        
        adds_done = 0
        deletes_done = 0
        modifies_done = 0
        
        for edit in edits:
            edit_type = edit.get("type", "")
            
            if edit_type == "add" and adds_done < max_adds:
                if self._apply_add(skill, edit):
                    adds_done += 1
            
            elif edit_type == "delete" and deletes_done < max_deletes:
                if self._apply_delete(skill, edit):
                    deletes_done += 1
            
            elif edit_type == "modify" and modifies_done < max_modifies:
                if self._apply_modify(skill, edit):
                    modifies_done += 1
            
            elif edit_type == "meta_consolidate":
                # Meta-review: consolidate overlapping rules
                self._apply_consolidate(skill)
        
        return skill
    
    def _apply_add(self, skill: Skill, edit: dict) -> bool:
        """Add a new rule."""
        content = edit.get("content", "")
        if not content:
            return False
        
        # Parse the new rule from content
        # Expected format: "IF condition THEN predict label"
        condition, prediction = self._parse_add_content(content)
        if not condition or not prediction:
            return False
        
        new_rule = Rule(
            id=f"R{len(skill.rules) + 1}",
            condition=condition,
            prediction=prediction,
            confidence=0.5,  # Initial low confidence, will be calibrated
            source="optimized",
        )
        
        # Insert after target rule if specified, otherwise append
        target = edit.get("target", "")
        if target:
            for i, rule in enumerate(skill.rules):
                if rule.id == target or target in rule.condition:
                    skill.rules.insert(i + 1, new_rule)
                    return True
        
        skill.rules.append(new_rule)
        return True
    
    def _apply_delete(self, skill: Skill, edit: dict) -> bool:
        """Remove a rule."""
        target = edit.get("target", "")
        if not target:
            return False
        
        return skill.remove_rule(target)
    
    def _apply_modify(self, skill: Skill, edit: dict) -> bool:
        """Modify an existing rule's condition or prediction."""
        target = edit.get("target", "")
        rule = skill.get_rule(target)
        if not rule:
            return False
        
        old = edit.get("old", "")
        new = edit.get("new", "")
        
        if old and new and old in rule.condition:
            rule.condition = rule.condition.replace(old, new)
            rule.confidence = max(0.3, rule.confidence - 0.05)  # Slight confidence decay on modification
            return True
        
        # If no old/new specified, apply suggestion-based modification
        suggestion = edit.get("suggestion", "")
        if suggestion == "tighten_threshold":
            # Adjust threshold values slightly
            rule.condition = self._tighten_thresholds(rule.condition)
            rule.confidence = max(0.3, rule.confidence - 0.02)
            return True
        
        return False
    
    def _apply_consolidate(self, skill: Skill):
        """Consolidate overlapping rules (meta-review operation)."""
        if len(skill.rules) < 2:
            return
        
        # Find rules with high condition overlap and same prediction
        to_remove = set()
        for i, r1 in enumerate(skill.rules):
            for j, r2 in enumerate(skill.rules):
                if j <= i or j in to_remove:
                    continue
                if r1.prediction != r2.prediction:
                    continue
                
                # Simple overlap: share a feature in condition
                r1_features = set(r1.condition.split(" AND ")[0].split()[0] for _ in [0])
                r2_features = set(r2.condition.split(" AND ")[0].split()[0] for _ in [0])
                
                # Broad overlap detection
                overlap = len(set(r1.condition.split()) & set(r2.condition.split())) / max(
                    len(set(r1.condition.split())), len(set(r2.condition.split())), 1
                )
                
                if overlap > 0.6:
                    # Merge: keep the higher-confidence rule, tighten the other's condition
                    if r1.confidence >= r2.confidence:
                        to_remove.add(r2.id)
                        r1.support = (r1.support[0] + r2.support[0], r1.support[1] + r2.support[1])
                    else:
                        to_remove.add(r1.id)
                        r2.support = (r1.support[0] + r2.support[0], r1.support[1] + r2.support[1])
        
        skill.rules = [r for r in skill.rules if r.id not in to_remove]
    
    def _parse_add_content(self, content: str) -> tuple:
        """Parse 'IF condition THEN predict label' into (condition, prediction)."""
        content = content.strip()
        
        # Try "IF ... THEN predict ..." format
        import re
        match = re.match(r"IF\s+(.+?)\s+THEN\s+predict\s+(.+)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        
        # Try "IF ... THEN ..." format
        match = re.match(r"IF\s+(.+?)\s+THEN\s+(.+)", content, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        
        return ("", "")
    
    def _tighten_thresholds(self, condition: str) -> str:
        """Slightly tighten threshold values in a condition."""
        import re
        
        def adjust(match):
            feat = match.group(1)
            op = match.group(2)
            val = float(match.group(3))
            
            # Tighten: for >, increase threshold by 5%; for <, decrease by 5%
            if op == ">":
                return f"{feat} > {val * 1.05:.2f}"
            elif op == "<":
                return f"{feat} < {val * 0.95:.2f}"
            elif op == ">=":
                return f"{feat} >= {val * 1.03:.2f}"
            elif op == "<=":
                return f"{feat} <= {val * 0.97:.2f}"
            return match.group(0)
        
        return re.sub(r"(\w+)\s*(>=|<=|>|<)\s*([\d.]+)", adjust, condition)
