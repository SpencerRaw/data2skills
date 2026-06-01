"""Multi-class LLM diagnoser experiment: Iris + Wine.

The statistical diagnoser performs poorly on multi-class tasks (52% Iris, 61% Wine)
because threshold-based rules don't capture class interactions well.
LLM-powered diagnosis should understand semantic feature relationships.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.skill import seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator
from src.llm_diagnosis import LLMDiagnoser

# API key
api_key = "sk-f2f...8502"
os.environ["DEEPSEEK" + "_API_KEY"] = api_key
os.environ["DEEPSEEK" + "_BASE_URL"] = "https://api.deepseek.com/v1"


def run_dataset(name, loader_fn, n_folds=5):
    """Run LLM diagnoser on a dataset with k-fold CV."""
    data = loader_fn()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    label_map = {name: i for i, name in enumerate(target_names)}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    evaluator = SkillEvaluator()
    
    llm_accs, llm_f1s, llm_rules = [], [], []
    
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Classes: {target_names} | {n_folds}-fold CV | LLM diagnoser")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        seed = seed_skill_from_data(
            name=f"{name}_fold{fold}", domain="classification",
            features=feature_names, target="target",
            X=X_train, y=y_train,
            feature_names=feature_names, target_names=target_names,
        )
        
        optimizer = SkillOptimizer(OptimizerConfig(
            epochs=3, batch_size=16, validation_split=0.2,
            max_adds_per_step=3, max_deletes_per_step=1, max_modifies_per_step=2,
            patience=2, verbose=False,
        ))
        
        optimizer.diagnoser = LLMDiagnoser(
            model="deepseek-chat",
            feature_names=feature_names,
            target_names=target_names,
        )
        
        optimized = optimizer.fit(
            X_train, y_train,
            feature_names=feature_names, target_names=target_names,
            domain="classification", skill_name=f"{name}_fold{fold}",
            label_map=label_map,
        )
        
        result = evaluator.evaluate(optimized, X_test, y_test, feature_names, label_map)
        llm_accs.append(result.accuracy)
        llm_f1s.append(result.f1_macro)
        llm_rules.append(len(optimized.rules))
        
        print(f"  Fold {fold+1}/{n_folds}: acc={result.accuracy:.3f}, "
              f"f1={result.f1_macro:.3f}, rules={len(optimized.rules)}")
    
    print(f"\n  Summary: {np.mean(llm_accs):.3f}±{np.std(llm_accs):.3f} acc, "
          f"{np.mean(llm_f1s):.3f}±{np.std(llm_f1s):.3f} f1, "
          f"{int(np.mean(llm_rules))} rules")
    
    return {
        "accuracy": (float(np.mean(llm_accs)), float(np.std(llm_accs))),
        "f1": (float(np.mean(llm_f1s)), float(np.std(llm_f1s))),
        "rules": int(np.mean(llm_rules)),
    }


def main():
    results = {}
    
    # Iris
    results["iris_llm"] = run_dataset("Iris", load_iris, n_folds=5)
    
    # Wine
    results["wine_llm"] = run_dataset("Wine", load_wine, n_folds=5)
    
    # Compare with previous statistical results
    print(f"\n{'='*60}")
    print(f"COMPARISON: Statistical vs LLM Diagnoser")
    print(f"{'='*60}")
    
    # Previous results (from run_all.py 10-fold CV)
    prev = {
        "Iris": {"stat": (0.520, 0.555), "rules": 4},
        "Wine": {"stat": (0.613, 0.616), "rules": 6},
    }
    
    for name, llm_key in [("Iris", "iris_llm"), ("Wine", "wine_llm")]:
        llm = results[llm_key]
        stat = prev[name]
        print(f"\n  {name}:")
        print(f"    Statistical:  acc={stat['stat'][0]:.3f}, f1={stat['stat'][1]:.3f}, rules={stat['rules']}")
        print(f"    LLM (DeepSeek): acc={llm['accuracy'][0]:.3f}, f1={llm['f1'][0]:.3f}, rules={llm['rules']}")
        delta = llm['accuracy'][0] - stat['stat'][0]
        print(f"    Δ: {delta:+.3f} ({delta*100:+.1f}pp)")
    
    import json
    with open(os.path.join(os.path.dirname(__file__), "..", "multiclass_llm_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
