"""Decision Tree → Skill converter for multi-class data2skills.

Strategy: Train a shallow DT, extract all leaf paths as IF-THEN rules,
then optimize with LLM diagnoser. This gives high-quality seed rules
that already handle multi-class well.

Results to beat:
  Iris: 52.0% (statistical) → target: DT's 93.3%
  Wine: 61.3% (statistical) → target: DT's 90.5%
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.skill import Skill, Rule
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator
from src.llm_diagnosis import LLMDiagnoser

api_key = "sk-f2f...8502"
os.environ["DEEPSEEK" + "_API_KEY"] = api_key
os.environ["DEEPSEEK" + "_BASE_URL"] = "https://api.deepseek.com/v1"


def dt_to_skill(dt, feature_names, target_names, domain="classification"):
    """Convert a trained DecisionTree into a data2skills Skill.
    
    Each leaf path becomes a rule with the leaf's majority class as prediction.
    """
    tree = dt.tree_
    rules = []
    
    def extract_paths(node_id, conditions, depth):
        if depth > 5:
            return
        
        if tree.feature[node_id] >= 0:  # internal node
            feat = feature_names[tree.feature[node_id]]
            thresh = tree.threshold[node_id]
            
            extract_paths(tree.children_left[node_id], 
                         conditions + [f"{feat} <= {thresh:.3f}"], depth+1)
            extract_paths(tree.children_right[node_id],
                         conditions + [f"{feat} > {thresh:.3f}"], depth+1)
        else:  # leaf
            if not conditions:
                return
            
            values = tree.value[node_id][0]
            total = values.sum()
            if total < 3:
                return
            
            pred_cls = int(np.argmax(values))
            confidence = values[pred_cls] / total
            pred_name = target_names[pred_cls] if pred_cls < len(target_names) else str(pred_cls)
            
            rules.append(Rule(
                id=f"R{len(rules)+1}",
                condition=" AND ".join(conditions),
                prediction=pred_name,
                confidence=float(confidence),
                support=(int(values[pred_cls]), int(total)),
                source="dt_seed",
            ))
    
    extract_paths(0, [], 0)
    
    return Skill(
        name="dt_skill", domain=domain,
        features=feature_names, target="target",
        rules=rules,
    )


def run_fold(X_train, X_test, y_train, y_test, feature_names, target_names, use_llm=True):
    """Run one fold of DT→Skill pipeline."""
    label_map = {name: i for i, name in enumerate(target_names)}
    evaluator = SkillEvaluator()
    
    # Step 1: Train shallow DT
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    
    # Step 2: Convert to skill
    skill = dt_to_skill(dt, feature_names, target_names)
    
    # Step 3: Evaluate seed (DT→Skill before optimization)
    seed_result = evaluator.evaluate(skill, X_test, y_test, feature_names, label_map)
    
    # Step 4: Optimize with LLM
    optimizer = SkillOptimizer(OptimizerConfig(
        epochs=3, batch_size=16, validation_split=0.2,
        max_adds_per_step=3, max_deletes_per_step=1, max_modifies_per_step=2,
        patience=2, verbose=False,
    ))
    
    if use_llm:
        optimizer.diagnoser = LLMDiagnoser(
            model="deepseek-chat",
            feature_names=feature_names,
            target_names=target_names,
        )
    
    optimized = optimizer.fit(
        X_train, y_train,
        feature_names=feature_names, target_names=target_names,
        domain="classification", skill_name="dt_skill",
        label_map=label_map,
    )
    
    opt_result = evaluator.evaluate(optimized, X_test, y_test, feature_names, label_map)
    
    return {
        "seed_acc": seed_result.accuracy,
        "seed_f1": seed_result.f1_macro,
        "seed_rules": len(skill.rules),
        "opt_acc": opt_result.accuracy,
        "opt_f1": opt_result.f1_macro,
        "opt_rules": len(optimized.rules),
        "dt_acc": accuracy_score(y_test, dt.predict(X_test)),
        "dt_f1": f1_score(y_test, dt.predict(X_test), average="macro"),
        "dt_rules": dt.get_n_leaves(),
    }


def run_dataset(name, loader_fn, n_folds=5):
    data = loader_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | DT→Skill + LLM | {n_folds}-fold CV")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        r = run_fold(X[train_idx], X[test_idx], y[train_idx], y[test_idx],
                     feature_names, target_names)
        results.append(r)
        
        print(f"  Fold {fold+1}: seed={r['seed_acc']:.2f}→opt={r['opt_acc']:.2f} "
              f"vs DT={r['dt_acc']:.2f} | rules: {r['seed_rules']}→{r['opt_rules']}")
    
    # Aggregate
    seed_accs = [r["seed_acc"] for r in results]
    opt_accs = [r["opt_acc"] for r in results]
    dt_accs = [r["dt_acc"] for r in results]
    
    print(f"\n  DT→Skill seed:   {np.mean(seed_accs):.3f}±{np.std(seed_accs):.3f}")
    print(f"  DT→Skill opt:    {np.mean(opt_accs):.3f}±{np.std(opt_accs):.3f}")
    print(f"  Original DT:      {np.mean(dt_accs):.3f}±{np.std(dt_accs):.3f}")
    
    return results


def main():
    all_results = {}
    for name, loader in [("Iris", load_iris), ("Wine", load_wine)]:
        all_results[name] = run_dataset(name, loader)
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    prev = {
        "Iris": {"stat": (0.520, 0.555), "dt": (0.933, 0.932)},
        "Wine": {"stat": (0.613, 0.616), "dt": (0.905, 0.908)},
    }
    
    for name in ["Iris", "Wine"]:
        r = all_results[name]
        opt_mean = np.mean([x["opt_acc"] for x in r])
        opt_std = np.std([x["opt_acc"] for x in r])
        p = prev[name]
        
        print(f"\n  {name}:")
        print(f"    Statistical (old):  {p['stat'][0]:.3f}")
        print(f"    DT→Skill+LLM (new):  {opt_mean:.3f}±{opt_std:.3f}")
        print(f"    Decision Tree:       {p['dt'][0]:.3f}")
        delta = opt_mean - p['stat'][0]
        print(f"    Δ: {delta:+.3f} ({delta*100:+.1f}pp)")


if __name__ == "__main__":
    main()
