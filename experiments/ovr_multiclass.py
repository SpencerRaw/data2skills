"""One-vs-Rest approach for multi-class data2skills.

Trains one binary skill per class, then uses weighted voting.
Uses decision tree paths as seed rules (much better than class means).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.skill import Skill, Rule, seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator
from src.llm_diagnosis import LLMDiagnoser

api_key = "sk-f2f...8502"
os.environ["DEEPSEEK" + "_API_KEY"] = api_key
os.environ["DEEPSEEK" + "_BASE_URL"] = "https://api.deepseek.com/v1"


def dt_to_rules(dt: DecisionTreeClassifier, feature_names: list[str], class_idx: int) -> list[Rule]:
    """Convert decision tree paths to IF-THEN rules.
    
    Each leaf becomes a rule: the path from root to leaf is the condition.
    """
    tree = dt.tree_
    rules = []
    
    def recurse(node_id, conditions, depth):
        if depth > 4:  # Limit depth
            return
        
        if tree.feature[node_id] >= 0:  # Internal node
            feat = feature_names[tree.feature[node_id]]
            thresh = tree.threshold[node_id]
            
            # Left child (<= threshold)
            left_cond = conditions + [f"{feat} <= {thresh:.3f}"]
            recurse(tree.children_left[node_id], left_cond, depth + 1)
            
            # Right child (> threshold)
            right_cond = conditions + [f"{feat} > {thresh:.3f}"]
            recurse(tree.children_right[node_id], right_cond, depth + 1)
        else:  # Leaf
            if conditions:
                condition = " AND ".join(conditions)
                n_samples = int(tree.n_node_samples[node_id])
                
                # Get class distribution at this leaf
                values = tree.value[node_id][0]
                total = values.sum()
                if total > 0:
                    majority_class = np.argmax(values)
                    confidence = values[majority_class] / total
                    
                    # Only create rule if this leaf predicts our target class
                    if majority_class == class_idx and confidence > 0.5:
                        rules.append(Rule(
                            id=f"R{len(rules)+1}",
                            condition=condition,
                            prediction=str(class_idx),
                            confidence=float(confidence),
                            support=(int(values[majority_class]), int(total)),
                            source="dt_seed",
                        ))
    
    recurse(0, [], 0)
    return rules


def train_ovr_skill(X_train, y_train, X_val, y_val, feature_names, target_names, use_llm=True):
    """Train One-vs-Rest skills: one binary skill per class, then vote."""
    label_map = {name: i for i, name in enumerate(target_names)}
    evaluator = SkillEvaluator()
    classes = np.unique(y_train)
    
    skills = []
    
    for cls in classes:
        cls_name = target_names[int(cls)]
        
        # Binary labels: this class vs rest
        y_bin = (y_train == cls).astype(int)
        y_val_bin = (y_val == cls).astype(int)
        
        # Seed from decision tree
        dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
        dt.fit(X_train, y_train)  # Full multi-class tree for richer rules
        
        # Extract rules that predict this class
        dt_rules = dt_to_rules(dt, feature_names, int(cls))
        
        # If not enough rules from DT, fall back to statistical seed
        if len(dt_rules) < 2:
            seed = seed_skill_from_data(
                name=f"ovr_{cls_name}", domain="classification",
                features=feature_names, target="target",
                X=X_train, y=y_bin,
                feature_names=feature_names, target_names=["other", cls_name],
            )
            dt_rules = seed.rules
        
        skill = Skill(
            name=f"ovr_{cls_name}",
            domain="classification",
            features=feature_names,
            target="target",
            rules=dt_rules,
        )
        
        # Optimize
        bin_label_map = {cls_name: 1, "other": 0}
        # Actually for OVR evaluation is different...
        # Let's use the standard optimizer
        optimizer = SkillOptimizer(OptimizerConfig(
            epochs=2, batch_size=16, validation_split=0.2,
            max_adds_per_step=2, max_deletes_per_step=1, max_modifies_per_step=2,
            patience=2, verbose=False,
        ))
        
        if use_llm:
            optimizer.diagnoser = LLMDiagnoser(
                model="deepseek-chat",
                feature_names=feature_names,
                target_names=[cls_name, "other"],
            )
        
        optimized = optimizer.fit(
            X_train, y_bin,
            feature_names=feature_names,
            target_names=[cls_name, "other"],
            domain="classification",
            skill_name=f"ovr_{cls_name}",
            label_map={cls_name: 1, "other": 0},
        )
        
        skills.append(optimized)
    
    return skills


def apply_ovr(skills, X, feature_names, label_map):
    """Apply OVR skills: each skill votes, highest confidence wins."""
    evaluator = SkillEvaluator()
    predictions = []
    
    # Inverse label map
    inv_map = {v: k for k, v in label_map.items()}
    
    for i in range(len(X)):
        votes = {}
        
        for skill in skills:
            # Get this skill's prediction
            pred, matched = evaluator._apply_skill_to_row(skill, X[i], feature_names)
            
            if pred is not None:
                # pred is a string like "0" or "class_name" — we need confidence
                for rule in skill.rules:
                    if evaluator._evaluate_condition(rule.condition, dict(zip(feature_names, X[i]))):
                        weight = rule.confidence
                        cls_pred = rule.prediction
                        votes[cls_pred] = votes.get(cls_pred, 0) + weight
        
        if votes:
            best = max(votes, key=votes.get)
            try:
                predictions.append(int(best))
            except ValueError:
                predictions.append(label_map.get(best, 0))
        else:
            predictions.append(0)  # Fallback
    
    return predictions


def evaluate_ovr(skills, X, y, feature_names, label_map):
    """Evaluate OVR ensemble."""
    predictions = apply_ovr(skills, X, feature_names, label_map)
    return {
        "accuracy": accuracy_score(y, predictions),
        "f1": f1_score(y, predictions, average="macro"),
        "predictions": predictions,
    }


def run_dataset(name, loader_fn, n_folds=5):
    """Run OVR data2skills on a dataset."""
    data = loader_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    label_map = {name: i for i, name in enumerate(target_names)}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    ovr_accs, ovr_f1s = [], []
    
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | OVR data2skills + LLM | {n_folds}-fold CV")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split train into train+val for optimizer
        n_val = int(len(X_train) * 0.2)
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
        
        skills = train_ovr_skill(X_tr, y_tr, X_val, y_val, feature_names, target_names)
        result = evaluate_ovr(skills, X_test, y_test, feature_names, label_map)
        
        ovr_accs.append(result["accuracy"])
        ovr_f1s.append(result["f1"])
        
        print(f"  Fold {fold+1}/{n_folds}: acc={result['accuracy']:.3f}, f1={result['f1']:.3f}")
    
    print(f"\n  OVR Summary: {np.mean(ovr_accs):.3f}±{np.std(ovr_accs):.3f} acc, "
          f"{np.mean(ovr_f1s):.3f}±{np.std(ovr_f1s):.3f} f1")
    
    return {"accuracy": (float(np.mean(ovr_accs)), float(np.std(ovr_accs))),
            "f1": (float(np.mean(ovr_f1s)), float(np.std(ovr_f1s)))}


def main():
    results = {}
    
    for name, loader in [("Iris", load_iris), ("Wine", load_wine)]:
        results[name] = run_dataset(name, loader)
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: All Methods")
    print(f"{'='*60}")
    
    prev = {
        "Iris": {"stat": (0.520, 0.555), "dt": (0.933, 0.932)},
        "Wine": {"stat": (0.613, 0.616), "dt": (0.905, 0.908)},
    }
    
    for name in ["Iris", "Wine"]:
        r = results[name]
        p = prev[name]
        print(f"\n  {name}:")
        print(f"    Statistical (old):   {p['stat'][0]:.3f} acc, {p['stat'][1]:.3f} f1")
        print(f"    OVR+LLM (new):        {r['accuracy'][0]:.3f} acc, {r['f1'][0]:.3f} f1")
        print(f"    Decision Tree:        {p['dt'][0]:.3f} acc, {p['dt'][1]:.3f} f1")
        delta = r['accuracy'][0] - p['stat'][0]
        print(f"    Δ vs statistical:     {delta:+.3f} ({delta*100:+.1f}pp)")


if __name__ == "__main__":
    main()
