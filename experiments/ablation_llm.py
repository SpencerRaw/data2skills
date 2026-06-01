"""Ablation experiment: Statistical vs LLM-powered diagnoser.

Runs data2skills on Breast Cancer with both diagnoser types and reports results.
Fills the TBD in the paper's ablation table.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.skill import seed_skill_from_data
from src.optimizer import SkillOptimizer, OptimizerConfig
from src.evaluator import SkillEvaluator
from src.diagnosis import FailureDiagnoser
from src.llm_diagnosis import LLMDiagnoser

# Set API key
api_key = "sk-f2fb40dcbe5240a4b2ce111aef8c8502"
os.environ["DEEPSEEK" + "_API_KEY"] = api_key
os.environ["DEEPSEEK" + "_BASE_URL"] = "https://api.deepseek.com/v1"


def run_ablation(diagnoser_type: str, X_train, X_test, y_train, y_test, feature_names, target_names):
    """Run data2skills with a specific diagnoser type."""
    label_map = {name: i for i, name in enumerate(target_names)}
    evaluator = SkillEvaluator()
    
    seed = seed_skill_from_data(
        name="ablation_skill", domain="medical",
        features=feature_names, target="target",
        X=X_train, y=y_train,
        feature_names=feature_names, target_names=target_names,
    )
    
    config = OptimizerConfig(
        epochs=3, batch_size=32, validation_split=0.2,
        max_adds_per_step=3, max_deletes_per_step=1, max_modifies_per_step=2,
        patience=2, verbose=True,
    )
    
    optimizer = SkillOptimizer(config)
    
    # Override diagnoser
    if diagnoser_type == "llm":
        optimizer.diagnoser = LLMDiagnoser(
            model="deepseek-chat",
            feature_names=feature_names,
            target_names=target_names,
        )
    else:
        optimizer.diagnoser = FailureDiagnoser(
            feature_names=feature_names,
            target_names=target_names,
        )
    
    print(f"\n{'='*50}")
    print(f"Diagnoser: {diagnoser_type}")
    print(f"{'='*50}")
    
    optimized = optimizer.fit(
        X_train, y_train,
        feature_names=feature_names,
        target_names=target_names,
        domain="medical", skill_name=f"ablation_{diagnoser_type}",
        label_map=label_map,
    )
    
    result = evaluator.evaluate(optimized, X_test, y_test, feature_names, label_map)
    
    print(f"\n  Result: acc={result.accuracy:.3f}, f1={result.f1_macro:.3f}, rules={len(optimized.rules)}")
    
    return {
        "diagnoser": diagnoser_type,
        "accuracy": result.accuracy,
        "f1": result.f1_macro,
        "n_rules": len(optimized.rules),
        "coverage": result.coverage,
        "skill_text": optimized.text[:500],
    }


def main():
    print("=" * 60)
    print("LLM Diagnoser Ablation — Breast Cancer")
    print("=" * 60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Seed only (no optimization)
    print("\n--- Seed Only ---")
    seed = seed_skill_from_data(
        name="seed", domain="medical", features=feature_names, target="target",
        X=X_train, y=y_train, feature_names=feature_names, target_names=target_names,
    )
    label_map = {name: i for i, name in enumerate(target_names)}
    evaluator = SkillEvaluator()
    seed_result = evaluator.evaluate(seed, X_test, y_test, feature_names, label_map)
    results["seed"] = {"accuracy": seed_result.accuracy, "n_rules": len(seed.rules)}
    print(f"  acc={seed_result.accuracy:.3f}, rules={len(seed.rules)}")
    
    # Statistical diagnoser
    stat_result = run_ablation("statistical", X_train, X_test, y_train, y_test, feature_names, target_names)
    results["statistical"] = stat_result
    
    # LLM diagnoser
    llm_result = run_ablation("llm", X_train, X_test, y_train, y_test, feature_names, target_names)
    results["llm"] = llm_result
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Diagnoser':<15} {'Accuracy':>8} {'Rules':>6}")
    print(f"  {'-'*30}")
    print(f"  {'None (seed)':<15} {results['seed']['accuracy']:>8.3f} {results['seed']['n_rules']:>6}")
    print(f"  {'Statistical':<15} {results['statistical']['accuracy']:>8.3f} {results['statistical']['n_rules']:>6}")
    print(f"  {'LLM (DeepSeek)':<15} {results['llm']['accuracy']:>8.3f} {results['llm']['n_rules']:>6}")
    
    import json
    out_path = os.path.join(os.path.dirname(__file__), "..", "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
