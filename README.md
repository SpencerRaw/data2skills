# data2skills — Gradient-Optimized Expert Knowledge from Data

> Traditional ML: Data → Model (weights) → Predictions (black box)  
> **data2skills**: Data → Skills (text knowledge) → Reasoning (interpretable)  
> *Inspired by SkillOpt (Microsoft, 2026) and SkillGrad*

---

## The Core Idea

Machine learning turns data into **numbers** (weights in a neural network).  
data2skills turns data into **text** (interpretable expert skills).

Instead of training a black-box classifier on 1000 patient records, we extract and optimize a set of natural-language diagnostic skills — like a doctor's clinical reasoning, but discovered and refined by gradient-based optimization over data.

### Why?

| | Traditional ML | data2skills |
|---|---|---|
| **Output** | Weight matrix (unreadable) | Expert skill document (readable) |
| **Interpretability** | Post-hoc (SHAP/LIME) | Built-in (the skill IS the explanation) |
| **Doctor's trust** | "The model says..." | "The skill says: check for X when Y..." |
| **Update** | Retrain whole model | Edit one skill rule |
| **Transfer** | Fine-tune on new data | Skills transfer with zero retraining |
| **Audit** | Hard | Read the skill document |

---

## How It Works

```
┌──────────┐    ┌──────────────────┐    ┌───────────────────┐
│  DATA    │───→│  SKILL OPTIMIZER │───→│  EXPERT SKILLS.md │
│ (tabular, │    │  (text-gradient  │    │  (interpretable   │
│  text,    │    │   descent)       │    │   knowledge)      │
│  images)  │    └──────────────────┘    └───────────────────┘
└──────────┘             │                        │
                         ▼                        ▼
                  ┌──────────────┐         ┌──────────────┐
                  │  Validation  │         │  EVALUATION   │
                  │  score gate  │         │  vs ML models │
                  └──────────────┘         └──────────────┘
```

### The Skill Optimization Loop

1. **Initialize**: Generate a seed skill from data statistics (mean, std, correlations)
2. **Rollout**: Apply the skill to a batch of training examples → get predictions
3. **Score**: Compare predictions to ground truth → compute loss
4. **Diagnose**: LLM analyzes failures and produces a **text gradient** (what to fix)
5. **Update**: Apply bounded edits (add/delete/replace rules) ← only if validation score improves
6. **Repeat**: Like gradient descent, but in text space

```
for epoch in range(N):
    for batch in dataloader:
        predictions = apply_skill(skill, batch)
        loss = compute_loss(predictions, labels)
        text_gradient = diagnose(skill, batch, loss)  # ← LLM call
        candidate = apply_edits(skill, text_gradient)
        if validate(candidate) > validate(skill):
            skill = candidate  # ← accept only if better
```

### Analogy to Gradient Descent

| Gradient Descent | data2skills |
|------------------|-------------|
| Parameters θ | Skill text S |
| Forward pass f(x; θ) | apply_skill(S, x) |
| Loss L(ŷ, y) | compute_loss(predictions, labels) |
| Gradient ∇L | text_gradient(S, failures) |
| θ = θ - α∇L | S = apply_edits(S, text_gradient) |
| Validation loss | Validation accuracy gate |

---

## Quick Start

```bash
git clone https://github.com/SpencerRaw/data2skills.git
cd data2skills
uv pip install -r requirements.txt
```

### Run on a classic dataset

```python
from data2skills import SkillOptimizer
from data2skills.data import load_breast_cancer

# Load data
X_train, X_test, y_train, y_test = load_breast_cancer()

# Initialize optimizer
optimizer = SkillOptimizer(
    model="deepseek-v4",        # LLM for text-gradient generation
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)

# Train — this produces optimized SKILL.md
skill = optimizer.fit(X_train, y_train)

# Evaluate
accuracy = skill.evaluate(X_test, y_test)
print(f"Skill accuracy: {accuracy:.2%}")

# Read the skill — it's human-readable!
print(skill.text)
```

**Expected skill output** (excerpt):
```markdown
# Breast Cancer Diagnostic Skill

## Rule 1: Radius threshold
IF worst_radius > 17.5 AND worst_concave_points > 0.14
THEN predict MALIGNANT (confidence: 0.92)

## Rule 2: Texture pattern
IF mean_texture > 20.0 AND worst_area > 900
THEN predict MALIGNANT (confidence: 0.88)

## Rule 3: Smoothness exclusion
IF worst_smoothness < 0.10 AND mean_concavity < 0.05
THEN predict BENIGN (confidence: 0.95)
```

---

## Project Structure

```
data2skills/
├── src/
│   ├── __init__.py
│   ├── skill.py           # Skill representation (text + metadata)
│   ├── optimizer.py       # Text-gradient descent loop
│   ├── evaluator.py       # Apply skill to data, compute metrics
│   ├── diagnosis.py       # LLM-based failure diagnosis → text gradient
│   ├── llm_diagnosis.py   # DeepSeek-powered text gradient generator
│   └── editor.py          # Bounded add/delete/replace edits
├── experiments/
│   ├── run_all.py         # Multi-dataset k-fold CV benchmark
│   └── breast_cancer.py   # Single-dataset experiment
├── paper/
│   └── main.tex           # Preprint (LaTeX)
├── skills/                # Optimized skill artifacts
├── docs/
│   └── design.md          # Technical design document
└── README.md
```

---

## Results (5-fold CV, Statistical Diagnoser)

| Dataset | Method | Accuracy | Rules |
|---------|--------|----------|-------|
| **Breast Cancer** | d2s (seed) | 64.3% ± 6.7 | 4 |
| (569 samples) | d2s (optimized) | **83.5% ± 4.1** | 8 |
| | Decision Tree (d=5) | 92.8% ± 2.3 | 15 |
| | Random Forest | 95.6% ± 1.2 | -- |
| **Iris** | d2s (optimized) | **94.7% ± 2.9** | 6 |
| (150 samples) | Decision Tree | 95.3% ± 3.1 | 7 |
| **Wine** | d2s (optimized) | **91.0% ± 3.6** | 7 |
| (178 samples) | Decision Tree | 89.3% ± 4.8 | 8 |

> With statistical diagnoser only. LLM-powered diagnoser (DeepSeek/Claude) is expected to close the remaining gap.

---

## Roadmap

- [x] Project design + architecture
- [x] Core optimizer loop (text-gradient descent)
- [x] Classic dataset experiments (Breast Cancer, Iris, Wine)
- [x] Head-to-head comparison vs 5 ML baselines with k-fold CV
- [x] LLM-powered diagnoser (DeepSeek integration)
- [x] Preprint draft (LaTeX)
- [ ] Ablation study: statistical vs LLM diagnoser
- [ ] Medical dataset integration (SLE patient records)
- [ ] Submit to arXiv
- [ ] Multi-skill composition (specialist skills that vote)

---

## References

- **SkillOpt**: Yang et al., "SkillOpt: Executive Strategy for Self-Evolving Agent Skills", arXiv 2605.23904, 2026
- **SkillGrad**: Wang et al., "SkillGrad: Optimizing Agent Skills Like Gradient Descent", arXiv 2605.27760, 2026
- **TextGrad**: Yuksekgonul et al., "TextGrad: Automatic Differentiation via Text", arXiv 2406.07496, 2024

## License

MIT
