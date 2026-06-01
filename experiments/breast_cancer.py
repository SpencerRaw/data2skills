"""Experiment: data2skills on UCI Breast Cancer Wisconsin dataset."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.skill import seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator


def load_data():
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


def train_baselines(X_train, X_test, y_train, y_test):
    results = {}
    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=2000)),
        ("Decision Tree (depth=5)", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
        }
        if "Decision Tree" in name:
            results[name]["n_rules"] = clf.get_n_leaves()
    return results


def run_experiment():
    print("=" * 60)
    print("data2skills — Breast Cancer Wisconsin")
    print("=" * 60)

    X, y, feature_names, target_names = load_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {target_names}")

    # ML baselines (use scaled data)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\n--- ML Baselines ---")
    ml_results = train_baselines(X_train_s, X_test_s, y_train, y_test)
    for name, m in ml_results.items():
        extra = f", rules={m.get('n_rules', 'N/A')}" if "n_rules" in m else ""
        print(f"  {name}: acc={m['accuracy']:.3f}, f1={m['f1']:.3f}{extra}")

    # Seed skill (raw values)
    print("\n--- Seed Skill ---")
    seed = seed_skill_from_data(
        name="breast_cancer_diagnostic",
        domain="medical_diagnostics",
        features=feature_names,
        target="diagnosis",
        X=X_train, y=y_train,
        feature_names=feature_names,
        target_names=target_names,
    )

    evaluator = SkillEvaluator()
    # Map string labels to ints for evaluation
    label_map = {name: i for i, name in enumerate(target_names)}
    
    seed_result = evaluator.evaluate(seed, X_test, y_test, feature_names, label_map)
    print(f"  Seed: acc={seed_result.accuracy:.3f}, f1={seed_result.f1_macro:.3f}, "
          f"rules={len(seed.rules)}, coverage={seed_result.coverage:.2%}")

    for rule in seed.rules:
        print(f"    {rule.id}: IF {rule.condition} THEN {rule.prediction}")

    # Optimize
    print("\n--- Optimizing ---")
    optimizer = SkillOptimizer(OptimizerConfig(
        epochs=5, batch_size=32, validation_split=0.2,
        max_adds_per_step=3, max_deletes_per_step=1, max_modifies_per_step=2,
        patience=3, verbose=True,
    ))
    
    optimized = optimizer.fit(
        X_train, y_train,
        feature_names=feature_names,
        target_names=target_names,
        domain="medical_diagnostics",
        skill_name="breast_cancer_diagnostic_optimized",
        label_map=label_map,
    )

    opt_result = evaluator.evaluate(optimized, X_test, y_test, feature_names, label_map)

    # Summary
    print(f"\n--- Comparison ---")
    print(f"  {'Model':<25} {'Accuracy':>8} {'F1':>8} {'Interpretable':>14}")
    print(f"  {'-'*55}")
    for name, m in ml_results.items():
        interp = "Yes" if "Decision Tree" in name else "No"
        print(f"  {name:<25} {m['accuracy']:>8.3f} {m['f1']:>8.3f} {interp:>14}")
    print(f"  {'data2skills (seed)':<25} {seed_result.accuracy:>8.3f} {seed_result.f1_macro:>8.3f} {'Yes (text)':>14}")
    print(f"  {'data2skills (optimized)':<25} {opt_result.accuracy:>8.3f} {opt_result.f1_macro:>8.3f} {'Yes (text)':>14}")

    # Save
    optimized.save(os.path.join(os.path.dirname(__file__), "..", "skills", "breast_cancer_skill.json"))
    with open(os.path.join(os.path.dirname(__file__), "..", "skills", "breast_cancer_skill.md"), "w") as f:
        f.write(optimized.text)
    print(f"\nSkills saved to skills/breast_cancer_skill.{{json,md}}")
    return optimized, ml_results


if __name__ == "__main__":
    run_experiment()
