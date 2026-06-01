"""Experiment: data2skills on UCI Breast Cancer Wisconsin dataset.

Compares skill-based classification against traditional ML baselines.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.skill import seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator


def load_data():
    """Load and prepare the Breast Cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_names, target_names


def train_baselines(X_train, X_test, y_train, y_test):
    """Train and evaluate traditional ML baselines."""
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }
    
    # Decision Tree (interpretable baseline)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    results["Decision Tree (depth=5)"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "n_rules": dt.get_n_leaves(),
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }
    
    return results


def run_experiment():
    """Run the full data2skills experiment."""
    print("=" * 60)
    print("data2skills — Breast Cancer Wisconsin Experiment")
    print("=" * 60)
    
    # Load data
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Classes: {target_names}")
    
    # Baseline ML models
    print("\n--- Training ML Baselines ---")
    ml_results = train_baselines(X_train, X_test, y_train, y_test)
    for name, metrics in ml_results.items():
        extra = f", rules={metrics.get('n_rules', 'N/A')}" if "n_rules" in metrics else ""
        print(f"  {name}: acc={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}{extra}")
    
    # data2skills — seed skill (no optimization)
    print("\n--- Seed Skill (before optimization) ---")
    seed = seed_skill_from_data(
        name="breast_cancer_diagnostic",
        domain="medical_diagnostics",
        features=feature_names,
        target="diagnosis",
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        target_names=target_names,
    )
    
    evaluator = SkillEvaluator()
    seed_result = evaluator.evaluate(seed, X_test, y_test, feature_names)
    print(f"  Seed skill: acc={seed_result.accuracy:.3f}, "
          f"f1={seed_result.f1_macro:.3f}, "
          f"rules={len(seed.rules)}, "
          f"coverage={seed_result.coverage:.2%}")
    
    print("\n  Seed skill rules:")
    for rule in seed.rules:
        print(f"    {rule.id}: IF {rule.condition} THEN {rule.prediction}")
    
    # data2skills — optimize
    print("\n--- Optimizing Skill (text-gradient descent) ---")
    config = OptimizerConfig(
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        max_adds_per_step=2,
        max_deletes_per_step=1,
        max_modifies_per_step=2,
        patience=3,
        verbose=True,
    )
    
    optimizer = SkillOptimizer(config)
    optimized_skill = optimizer.fit(
        X_train, y_train,
        feature_names=feature_names,
        target_names=target_names,
        domain="medical_diagnostics",
        skill_name="breast_cancer_diagnostic_optimized",
    )
    
    # Evaluate optimized skill
    opt_result = evaluator.evaluate(optimized_skill, X_test, y_test, feature_names)
    
    print(f"\n--- Final Results ---")
    print(f"  Optimized skill: acc={opt_result.accuracy:.3f}, "
          f"f1={opt_result.f1_macro:.3f}, "
          f"rules={len(optimized_skill.rules)}, "
          f"coverage={opt_result.coverage:.2%}")
    
    print(f"\n  Optimized skill rules:")
    for rule in optimized_skill.rules:
        print(f"    {rule.id}: IF {rule.condition} THEN {rule.prediction} "
              f"(conf={rule.confidence:.2f})")
    
    # Comparison summary
    print(f"\n--- Comparison ---")
    print(f"  {'Model':<25} {'Accuracy':>8} {'F1':>8} {'Interpretable':>14}")
    print(f"  {'-'*55}")
    for name, metrics in ml_results.items():
        interp = "Yes" if "Decision Tree" in name else "No"
        print(f"  {name:<25} {metrics['accuracy']:>8.3f} {metrics['f1']:>8.3f} {interp:>14}")
    print(f"  {'data2skills (seed)':<25} {seed_result.accuracy:>8.3f} {seed_result.f1_macro:>8.3f} {'Yes (text)':>14}")
    print(f"  {'data2skills (optimized)':<25} {opt_result.accuracy:>8.3f} {opt_result.f1_macro:>8.3f} {'Yes (text)':>14}")
    
    # Save optimized skill
    out_path = os.path.join(os.path.dirname(__file__), "..", "skills", "breast_cancer_skill.json")
    optimized_skill.save(out_path)
    print(f"\nOptimized skill saved to: {out_path}")
    
    # Also save as readable markdown
    md_path = os.path.join(os.path.dirname(__file__), "..", "skills", "breast_cancer_skill.md")
    with open(md_path, "w") as f:
        f.write(optimized_skill.text)
    print(f"Readable skill saved to: {md_path}")
    
    return optimized_skill, ml_results


if __name__ == "__main__":
    run_experiment()
