"""Robust experiment runner: k-fold CV, multiple datasets, statistical tests.

Usage:
    python experiments/run_all.py                    # all datasets
    python experiments/run_all.py --dataset iris     # single dataset
    python experiments/run_all.py --kfold 10          # 10-fold CV
"""

from __future__ import annotations
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

from src.skill import seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator


DATASETS = {
    "iris": {
        "loader": load_iris,
        "domain": "botany",
    },
    "wine": {
        "loader": load_wine,
        "domain": "chemistry",
    },
    "breast_cancer": {
        "loader": load_breast_cancer,
        "domain": "medical_diagnostics",
    },
    "diabetes": {
        "loader": load_diabetes,
        "domain": "medical_diagnostics",
        "binarize": True,  # Regression → binary classification
    },
}

BASELINES = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (d=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def load_dataset(name: str):
    """Load a dataset by name. Returns X, y, feature_names, target_names."""
    cfg = DATASETS[name]
    data = cfg["loader"]()
    
    X = data.data
    y = data.target
    
    if hasattr(data, "feature_names"):
        feature_names = list(data.feature_names)
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if hasattr(data, "target_names"):
        target_names = list(data.target_names)
    else:
        target_names = [str(c) for c in np.unique(y)]
    
    # Binarize regression targets
    if cfg.get("binarize"):
        median = np.median(y)
        y = (y > median).astype(int)
        target_names = ["below_median", "above_median"]
    
    # Ensure labels are 0, 1, 2, ...
    if y.dtype not in [np.int32, np.int64]:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y, feature_names, target_names


def run_baselines(X_train, X_test, y_train, y_test):
    """Run all baseline models."""
    results = {}
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    for name, clf in BASELINES.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
        }
        if "Decision Tree" in name:
            results[name]["n_rules"] = clf.get_n_leaves()
    
    return results


def run_data2skills(X_train, X_test, y_train, y_test, feature_names, target_names):
    """Run data2skills pipeline: seed → optimize."""
    label_map = {name: i for i, name in enumerate(target_names)}
    evaluator = SkillEvaluator()
    
    # Seed
    seed = seed_skill_from_data(
        name="skill", domain="classification",
        features=feature_names, target="target",
        X=X_train, y=y_train,
        feature_names=feature_names, target_names=target_names,
    )
    seed_result = evaluator.evaluate(seed, X_test, y_test, feature_names, label_map)
    
    # Optimize (use reduced epochs for speed)
    optimizer = SkillOptimizer(OptimizerConfig(
        epochs=3, batch_size=32, validation_split=0.2,
        max_adds_per_step=3, max_deletes_per_step=1, max_modifies_per_step=2,
        patience=2, verbose=False,
    ))
    
    optimized = optimizer.fit(
        X_train, y_train,
        feature_names=feature_names, target_names=target_names,
        domain="classification", skill_name="optimized",
        label_map=label_map,
    )
    opt_result = evaluator.evaluate(optimized, X_test, y_test, feature_names, label_map)
    
    return {
        "seed_accuracy": seed_result.accuracy,
        "seed_f1": seed_result.f1_macro,
        "seed_rules": len(seed.rules),
        "optimized_accuracy": opt_result.accuracy,
        "optimized_f1": opt_result.f1_macro,
        "optimized_rules": len(optimized.rules),
        "coverage": opt_result.coverage,
    }


def run_kfold(dataset_name: str, n_splits: int = 5):
    """Run k-fold cross-validation for all methods."""
    X, y, feature_names, target_names = load_dataset(dataset_name)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} | Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Classes: {target_names} | {n_splits}-fold CV")
    print(f"{'='*70}")
    
    # Accumulate results
    baseline_scores = {name: [] for name in BASELINES}
    d2s_seed = []
    d2s_opt = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Baselines
        bl = run_baselines(X_train, X_test, y_train, y_test)
        for name, scores in bl.items():
            baseline_scores[name].append(scores)
        
        # data2skills
        d2s = run_data2skills(X_train, X_test, y_train, y_test, feature_names, target_names)
        d2s_seed.append(d2s)
        d2s_opt.append(d2s)
        
        print(f"  Fold {fold+1}/{n_splits}: ", end="")
        print(f"d2s_seed={d2s['seed_accuracy']:.2f}, ", end="")
        print(f"d2s_opt={d2s['optimized_accuracy']:.2f}, ", end="")
        print(f"RF={bl['Random Forest']['accuracy']:.2f}")
    
    # Aggregate
    print(f"\n{'─'*70}")
    print(f"{'Method':<25} {'Accuracy':>10} {'F1':>10} {'Rules':>8}")
    print(f"{'─'*70}")
    
    results = {}
    
    for name in BASELINES:
        accs = [s["accuracy"] for s in baseline_scores[name]]
        f1s = [s["f1"] for s in baseline_scores[name]]
        mu_a, std_a = np.mean(accs), np.std(accs)
        mu_f, std_f = np.mean(f1s), np.std(f1s)
        rules = baseline_scores[name][0].get("n_rules", "N/A")
        print(f"  {name:<25} {mu_a:.3f}±{std_a:.3f} {mu_f:.3f}±{std_f:.3f} {rules:>8}")
        results[name] = {"accuracy": (mu_a, std_a), "f1": (mu_f, std_f), "rules": rules}
    
    # data2skills seed
    accs = [s["seed_accuracy"] for s in d2s_seed]
    f1s = [s["seed_f1"] for s in d2s_seed]
    rules = int(np.mean([s["seed_rules"] for s in d2s_seed]))
    print(f"  {'d2s (seed)':<25} {np.mean(accs):.3f}±{np.std(accs):.3f} {np.mean(f1s):.3f}±{np.std(f1s):.3f} {rules:>8}")
    results["d2s_seed"] = {"accuracy": (np.mean(accs), np.std(accs)), "f1": (np.mean(f1s), np.std(f1s)), "rules": rules}
    
    # data2skills optimized
    accs = [s["optimized_accuracy"] for s in d2s_opt]
    f1s = [s["optimized_f1"] for s in d2s_opt]
    rules = int(np.mean([s["optimized_rules"] for s in d2s_opt]))
    print(f"  {'d2s (optimized)':<25} {np.mean(accs):.3f}±{np.std(accs):.3f} {np.mean(f1s):.3f}±{np.std(f1s):.3f} {rules:>8}")
    results["d2s_optimized"] = {"accuracy": (np.mean(accs), np.std(accs)), "f1": (np.mean(f1s), np.std(f1s)), "rules": rules}
    
    # Statistical test: d2s_opt vs Decision Tree
    dt_accs = [s["accuracy"] for s in baseline_scores["Decision Tree (d=5)"]]
    d2s_accs = [s["optimized_accuracy"] for s in d2s_opt]
    t_stat, p_val = stats.ttest_rel(d2s_accs, dt_accs)
    print(f"\n  d2s(opt) vs Decision Tree: t={t_stat:.3f}, p={p_val:.4f}")
    
    results["_test"] = {"t_stat": t_stat, "p_val": p_val}
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--kfold", type=int, default=5)
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_kfold(ds, n_splits=args.kfold)
    
    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "results.json")
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
