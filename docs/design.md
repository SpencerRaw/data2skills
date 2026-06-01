# data2skills — Technical Design

## Problem Formulation

Given a dataset D = {(x_i, y_i)}, where x_i are features and y_i are labels, 
traditional ML learns a function f_θ: x → y parameterized by weights θ.

data2skills learns a text skill document S (a structured natural-language artifact) 
that can be applied by an LLM to predict y from x:

```
S* = argmax_S  E[(apply_skill(S, x) == y)]  for (x,y) in D_validation
```

The optimization is performed via text-gradient descent, inspired by SkillOpt and SkillGrad.

## Core Components

### 1. Skill Representation

A skill is a structured text document with:

```yaml
# SKILL.md
name: breast_cancer_diagnostic
version: 1.0
domain: medical_diagnostics
features: [worst_radius, worst_texture, worst_perimeter, ...]
target: diagnosis (M/B)
performance:
  accuracy: 0.94
  f1: 0.93
  trained_on: breast_cancer_wisconsin
  validation_split: 0.2

rules:
  - id: R1
    condition: worst_radius > 17.5 AND worst_concave_points > 0.14
    prediction: MALIGNANT
    confidence: 0.92
    support: 127/138 correct
    
  - id: R2
    condition: worst_smoothness < 0.10 AND mean_concavity < 0.05
    prediction: BENIGN
    confidence: 0.95
    support: 89/94 correct
```

### 2. Skill Application (Forward Pass)

To apply a skill to a data point x:

1. **Rule matching**: For each rule in the skill, evaluate its condition against x
2. **Conflict resolution**: If multiple rules match, use confidence-weighted voting
3. **Fallback**: If no rule matches, use nearest-neighbor reasoning over matched partial conditions
4. **Confidence calibration**: Scale confidence by rule support rate

```
apply_skill(S, x):
    matches = []
    for rule in S.rules:
        if evaluate(rule.condition, x):
            matches.append(rule)
    
    if matches:
        return weighted_vote(matches)  # by confidence × support_rate
    else:
        return fallback_reasoning(S, x)  # LLM call with skill context
```

### 3. Loss Function

```
compute_loss(predictions, labels, skill):
    accuracy_loss = 1 - accuracy(predictions, labels)
    
    # Regularization penalties
    complexity_penalty = α * len(skill.rules)  # prefer simpler skills
    overlap_penalty = β * rule_overlap(skill)  # prefer non-redundant rules
    coverage_penalty = γ * (1 - coverage(skill))  # prefer skills that cover all cases
    
    return accuracy_loss + complexity_penalty + overlap_penalty + coverage_penalty
```

### 4. Text-Gradient Generation (Backward Pass)

Given failures F = {(x, y_true, y_pred)} where y_pred ≠ y_true, an LLM generates a text-gradient:

```
text_gradient(S, F) → "Edit Plan"

Input to LLM:
- Current skill S
- Failure examples F (up to k examples)
- Feature descriptions

LLM outputs structured edit plan:
{
    "diagnosis": "Rules R1 and R3 are conflicting on high-radius benign cases...",
    "edits": [
        {"type": "add", "target": "after R1", "content": "IF worst_radius > 17.5 AND worst_concave_points > 0.14 AND worst_texture < 15 THEN predict MALIGNANT"},
        {"type": "modify", "target": "R3", "old": "worst_smoothness < 0.10", "new": "worst_smoothness < 0.10 AND worst_concavity < 0.08"},
        {"type": "delete", "target": "R5", "reason": "covered by modified R3"}
    ]
}
```

### 5. Bounded Edit Application

Edits are applied atomically. Only add/delete/replace operations:

- **add**: Insert a new rule (max budget: L_add per epoch)
- **delete**: Remove a rule (max budget: L_delete)
- **modify**: Replace condition/prediction in existing rule (max budget: L_modify)

Learning rate analogy: L_total = L_add + L_delete + L_modify controls how aggressive each update is.

### 6. Validation Gate

```python
def update_skill(skill, edits):
    candidate = apply_edits(skill, edits)
    if validate(candidate, D_val) > validate(skill, D_val):
        return candidate  # accept
    else:
        rejected_buffer.append(edits)  # save for momentum
        return skill  # reject
```

Rejected edits accumulate in a buffer. After N consecutive rejections of similar edits, trigger a "meta-review" — the LLM rethinks the optimization strategy.

### 7. Momentum (from SkillGrad)

Recurring diagnostic patterns across epochs are accumulated:

```
momentum[pattern] += 1 if pattern appears in this epoch's diagnosis

After k epochs, momentum-weighted patterns get priority in edit proposals.
```

## Optimization Loop Pseudocode

```python
def optimize_skill(D_train, D_val, epochs=10, batch_size=32):
    skill = initialize_skill(D_train)  # seed from data statistics
    rejected_buffer = []
    momentum = {}
    
    for epoch in range(epochs):
        # Shuffle and batch
        for batch in batch(D_train, batch_size):
            # Forward
            predictions = [apply_skill(skill, x) for x in batch]
            loss = compute_loss(predictions, batch_labels, skill)
            
            # Diagnose failures
            failures = [(x, y_true, y_pred) for (x, y_true, y_pred) 
                       in zip(batch, batch_labels, predictions) if y_true != y_pred]
            
            if not failures:
                continue
            
            # Text gradient (LLM call)
            text_grad = llm_diagnose(skill, failures, momentum)
            
            # Apply edits with validation gate
            edits = parse_edits(text_grad)
            candidate = apply_edits(skill, edits)
            
            if evaluate(candidate, D_val) > evaluate(skill, D_val):
                skill = candidate
                update_momentum(momentum, text_grad.diagnosis)
            else:
                rejected_buffer.append(edits)
        
        # Slow update (meta-review)
        if len(rejected_buffer) > threshold:
            skill = meta_review(skill, rejected_buffer, momentum)
            rejected_buffer.clear()
        
        # Log
        print(f"Epoch {epoch}: val_acc={evaluate(skill, D_val):.3f}, "
              f"n_rules={len(skill.rules)}, momentum_dims={len(momentum)}")
    
    return skill
```

## Baseline Comparisons

For each dataset, compare data2skills against:

| Model | Type | Interpretable? |
|-------|------|---------------|
| Logistic Regression | Linear | Partially (coefficients) |
| Decision Tree (max_depth=5) | Rule-based | Yes (but brittle) |
| Random Forest | Ensemble | Post-hoc only |
| XGBoost | Gradient boosting | Post-hoc only |
| **data2skills** | **Text skill** | **Yes (the skill IS the explanation)** |
| LLM zero-shot | Prompt only | Yes (but unoptimized) |

## Datasets

### Phase 1: Classic Benchmarks
- Iris (150 samples, 3 classes)
- Wine (178 samples, 3 classes)
- Breast Cancer Wisconsin (569 samples, 2 classes)
- Diabetes (442 samples, regression → binarized)

### Phase 2: Medical
- SLE patient records (1000 samples, private)
- MIMIC-III subset (if accessible)

## Evaluation Metrics

- **Accuracy**: Standard classification accuracy
- **F1 Score**: Per-class and macro-averaged
- **Rule Simplicity**: Number of rules in the skill (fewer = more interpretable)
- **Coverage**: % of test samples matched by at least one rule
- **Confidence Calibration**: Brier score between rule confidence and actual accuracy
