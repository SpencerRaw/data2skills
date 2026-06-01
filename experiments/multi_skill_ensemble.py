"""Multi-Skill Composition: specialist skills that vote.

Trains multiple skills on different feature subsets and combines them
via confidence-weighted voting. Demonstrates that ensemble of interpretable
skills can outperform single-skill approaches.

Strategy:
1. Split features into logical groups
2. Train one skill per group (each a "specialist")
3. Combine via weighted voting
4. Compare vs single-skill and vs ML baselines
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.skill import Skill, Rule, seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator

# Feature groups for Breast Cancer (by measurement type)
BC_FEATURE_GROUPS = {
    "Morphology": ["mean radius", "mean texture", "mean perimeter", "mean area",
                   "worst radius", "worst texture", "worst perimeter", "worst area"],
    "Shape_detail": ["mean smoothness", "mean compactness", "mean concavity",
                     "mean concave points", "mean symmetry", "mean fractal dimension",
                     "worst smoothness", "worst compactness", "worst concavity",
                     "worst concave points", "worst symmetry", "worst fractal dimension"],
    "Error_metrics": ["radius error", "texture error", "perimeter error", "area error",
                      "smoothness error", "compactness error", "concavity error",
                      "concave points error", "symmetry error", "fractal dimension error"],
}

# Feature groups for Diabetes
DIABETES_FEATURE_GROUPS = {
    "Demographic": ["age", "sex", "bmi"],
    "Blood_pressure": ["bp"],
    "Blood_tests": ["s1", "s2", "s3", "s4", "s5", "s6"],
}


def train_specialist_skills(X_train, y_train, feature_names, feature_groups, target_names):
    """Train one skill per feature group."""
    evaluator = SkillEvaluator()
    skills = {}
    
    for group_name, group_features in feature_groups.items():
        # Find indices of group features
        indices = [i for i, fn in enumerate(feature_names) if fn in group_features]
        if len(indices) < 2:
            continue
        
        X_group = X_train[:, indices]
        group_fn = [feature_names[i] for i in indices]
        
        # Seed skill
        skill = seed_skill_from_data(
            name=f"specialist_{group_name}",
            domain="medical",
            features=group_fn,
            target="target",
            X=X_group, y=y_train,
            feature_names=group_fn,
            target_names=target_names,
        )
        
        # Optimize
        optimizer = SkillOptimizer(OptimizerConfig(
            epochs=3, batch_size=16, validation_split=0.2,
            max_adds_per_step=3, max_deletes_per_step=1, max_modifies_per_step=2,
            patience=2, verbose=False,
        ))
        
        label_map = {name: i for i, name in enumerate(target_names)}
        
        optimized = optimizer.fit(
            X_group, y_train,
            feature_names=group_fn,
            target_names=target_names,
            domain="medical",
            skill_name=f"specialist_{group_name}",
            label_map=label_map,
        )
        
        # Store with feature indices for application
        skills[group_name] = {
            "skill": optimized,
            "indices": indices,
            "feature_names": group_fn,
        }
    
    return skills


def apply_ensemble(skills_dict, X, feature_names, target_names):
    """Apply ensemble of specialist skills via confidence-weighted voting."""
    predictions = []
    n_classes = len(target_names)
    
    for i in range(len(X)):
        votes = np.zeros(n_classes)
        
        for group_name, skill_info in skills_dict.items():
            skill = skill_info["skill"]
            indices = skill_info["indices"]
            group_fn = skill_info["feature_names"]
            
            x_group = X[i, indices]
            feature_map = dict(zip(group_fn, x_group))
            
            for rule in skill.rules:
                # Simple condition evaluation
                try:
                    if _eval_condition(rule.condition, feature_map):
                        # Find which class this rule predicts
                        for cls_idx, cls_name in enumerate(target_names):
                            if rule.prediction == cls_name or rule.prediction == str(cls_idx):
                                votes[cls_idx] += rule.confidence
                                break
                except:
                    pass
        
        if votes.sum() > 0:
            predictions.append(int(np.argmax(votes)))
        else:
            predictions.append(0)  # Fallback
    
    return predictions


def _eval_condition(condition, feature_map):
    """Simple condition evaluator."""
    import operator
    ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
    
    condition = condition.strip()
    
    if " AND " in condition:
        return all(_eval_single(p.strip(), feature_map) for p in condition.split(" AND "))
    if " OR " in condition:
        return any(_eval_single(p.strip(), feature_map) for p in condition.split(" OR "))
    return _eval_single(condition, feature_map)


def _eval_single(cond, fmap):
    """Evaluate single condition."""
    for op_str in [">=", "<=", ">", "<"]:
        if op_str in cond:
            parts = cond.split(op_str, 1)
            if len(parts) != 2: continue
            feat = parts[0].strip()
            try:
                val = float(parts[1].strip())
            except ValueError:
                continue
            if feat in fmap:
                import operator
                ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
                return ops[op_str](float(fmap[feat]), val)
    return False


def run_ensemble_experiment(dataset_name, loader_fn, feature_groups, n_folds=5):
    """Run multi-skill composition experiment."""
    data = loader_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names) if hasattr(data, "target_names") else ["low", "high"]
    
    # Binarize if needed
    if y.dtype not in [np.int32, np.int64, np.int0]:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    
    if len(target_names) > 2 and dataset_name == "diabetes":
        median = np.median(y)
        y = (y > median).astype(int)
        target_names = ["below_median", "above_median"]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    ensemble_accs, single_accs, dt_accs, rf_accs = [], [], [], []
    
    print(f"\n{'='*60}")
    print(f"Multi-Skill Ensemble: {dataset_name}")
    print(f"Feature groups: {list(feature_groups.keys())}")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train specialists
        skills_dict = train_specialist_skills(
            X_train, y_train, feature_names, feature_groups, target_names
        )
        
        # Ensemble prediction
        ensemble_pred = apply_ensemble(skills_dict, X_test, feature_names, target_names)
        ensemble_accs.append(accuracy_score(y_test, ensemble_pred))
        
        # Single skill baseline (one random specialist)
        if skills_dict:
            one_skill = list(skills_dict.values())[0]["skill"]
            evaluator = SkillEvaluator()
            label_map = {name: i for i, name in enumerate(target_names)}
            # Use full features for single skill comparison
            single_result = evaluator.evaluate(one_skill, X_test, y_test, feature_names, label_map)
            single_accs.append(single_result.accuracy)
        
        # ML baselines
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train_s, y_train)
        dt_accs.append(accuracy_score(y_test, dt.predict(X_test_s)))
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_accs.append(accuracy_score(y_test, rf.predict(X_test_s)))
        
        print(f"  Fold {fold+1}: ensemble={ensemble_accs[-1]:.3f}, "
              f"single={single_accs[-1]:.3f}, DT={dt_accs[-1]:.3f}, RF={rf_accs[-1]:.3f}")
    
    e_mean, e_std = np.mean(ensemble_accs), np.std(ensemble_accs)
    s_mean, s_std = np.mean(single_accs), np.std(single_accs)
    dt_mean, dt_std = np.mean(dt_accs), np.std(dt_accs)
    rf_mean, rf_std = np.mean(rf_accs), np.std(rf_accs)
    
    print(f"\n  Ensemble:   {e_mean:.3f}±{e_std:.3f}")
    print(f"  Single:     {s_mean:.3f}±{s_std:.3f}")
    print(f"  DT(d=5):    {dt_mean:.3f}±{dt_std:.3f}")
    print(f"  RF:         {rf_mean:.3f}±{rf_std:.3f}")
    
    ensemble_delta = e_mean - s_mean
    print(f"\n  Ensemble Δ over single: {ensemble_delta:+.3f} ({ensemble_delta*100:+.1f}pp)")
    
    return {"ensemble": (e_mean, e_std), "single": (s_mean, s_std), "dt": (dt_mean, dt_std), "rf": (rf_mean, rf_std)}


def main():
    results = {}
    
    # Breast Cancer with 3 feature groups
    results["breast_cancer"] = run_ensemble_experiment(
        "Breast Cancer", load_breast_cancer, BC_FEATURE_GROUPS
    )
    
    # Diabetes with 3 feature groups
    results["diabetes"] = run_ensemble_experiment(
        "Diabetes", load_diabetes, DIABETES_FEATURE_GROUPS
    )
    
    print(f"\n{'='*60}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        delta = r["ensemble"][0] - r["single"][0]
        print(f"  {name}: ensemble={r['ensemble'][0]:.3f} "
              f"(single={r['single'][0]:.3f}, Δ={delta:+.3f}) "
              f"vs DT={r['dt'][0]:.3f}, RF={r['rf'][0]:.3f}")


if __name__ == "__main__":
    main()
