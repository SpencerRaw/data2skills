"""Text-gradient descent optimizer for skills.

The core optimization loop:
1. Forward: apply skill to batch → predictions
2. Loss: compare predictions to labels
3. Diagnose: LLM analyzes failures → text gradient (edit plan)
4. Update: apply edits only if validation score improves
5. Momentum: accumulate recurring diagnostic patterns
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Optional, Callable

from .skill import Skill, Rule
from .evaluator import SkillEvaluator
from .diagnosis import FailureDiagnoser
from .editor import SkillEditor


@dataclass
class OptimizerConfig:
    """Configuration for the skill optimizer."""
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    
    # Learning rate analogues
    max_adds_per_step: int = 2       # Max new rules per update
    max_deletes_per_step: int = 1    # Max rule deletions per update
    max_modifies_per_step: int = 2   # Max rule modifications per update
    
    # Momentum
    momentum_decay: float = 0.9      # Decay factor for momentum accumulation
    meta_review_threshold: int = 5   # Rejected edits before meta-review
    
    # LLM
    model: str = "deepseek-v4"       # Model for text-gradient generation
    max_failures_per_diagnosis: int = 8  # Max failure examples to show LLM
    
    # Early stopping
    patience: int = 3                # Epochs without improvement before stopping
    min_delta: float = 0.005         # Minimum improvement to count
    
    verbose: bool = True


@dataclass
class OptimizationState:
    """Mutable state tracked across the optimization loop."""
    skill: Skill
    epoch: int = 0
    best_val_score: float = 0.0
    best_skill: Optional[Skill] = None
    rejected_buffer: list = field(default_factory=list)
    momentum: dict = field(default_factory=dict)  # pattern → count
    patience_counter: int = 0
    history: list[dict] = field(default_factory=list)  # per-epoch metrics


class SkillOptimizer:
    """Text-gradient descent optimizer for skills.
    
    Usage:
        optimizer = SkillOptimizer(config)
        skill = optimizer.fit(X_train, y_train, feature_names, target_names)
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None, **kwargs):
        self.config = config or OptimizerConfig(**kwargs)
        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        self.evaluator = SkillEvaluator()
        self.diagnoser: Optional[FailureDiagnoser] = None
        self.editor = SkillEditor()
    
    def fit(
        self,
        X,
        y,
        feature_names: list[str],
        target_names: list[str],
        domain: str = "classification",
        skill_name: str = "optimized_skill",
        progress_callback: Optional[Callable] = None,
        label_map: Optional[dict] = None,
    ) -> Skill:
        """Run the full optimization loop.
        
        Returns the best skill found (by validation score).
        """
        import numpy as np
        
        # Split data
        n_val = int(len(X) * self.config.validation_split)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Initialize skill from data statistics
        from .skill import seed_skill_from_data
        skill = seed_skill_from_data(
            name=skill_name,
            domain=domain,
            features=feature_names,
            target="target",
            X=X_train,
            y=y_train,
            feature_names=feature_names,
            target_names=target_names,
        )
        
        # Initialize diagnoser (lazy, needs LLM)
        self.diagnoser = FailureDiagnoser(
            model=self.config.model,
            feature_names=feature_names,
            target_names=target_names,
        )
        
        state = OptimizationState(
            skill=skill,
            best_val_score=self.evaluator.evaluate(skill, X_val, y_val, feature_names, label_map).accuracy,
            best_skill=copy.deepcopy(skill),
        )
        
        if self.config.verbose:
            print(f"Seed skill: {len(skill.rules)} rules, "
                  f"val_acc={state.best_val_score:.3f}")
        
        # Optimization loop
        for epoch in range(self.config.epochs):
            state.epoch = epoch
            
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]
            
            epoch_losses = []
            
            # Mini-batch training
            for batch_start in range(0, len(X_train), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(X_train))
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                
                # Forward pass
                result = self.evaluator.evaluate(state.skill, X_batch, y_batch, feature_names, label_map)
                epoch_losses.append(1 - result.accuracy)
                
                # Find failures
                failures = [
                    (X_batch[i], y_batch[i], pred)
                    for i, (true, pred) in enumerate(zip(y_batch, result.predictions))
                    if true != pred
                ]
                
                if len(failures) == 0:
                    continue
                
                # Text gradient (LLM call)
                text_gradient = self.diagnoser.diagnose(
                    skill=state.skill,
                    failures=failures[:self.config.max_failures_per_diagnosis],
                    momentum=state.momentum,
                )
                
                # Apply edits
                candidate = self.editor.apply_edits(
                    skill=copy.deepcopy(state.skill),
                    edit_plan=text_gradient,
                    max_adds=self.config.max_adds_per_step,
                    max_deletes=self.config.max_deletes_per_step,
                    max_modifies=self.config.max_modifies_per_step,
                )
                
                # Validation gate
                candidate_score = self.evaluator.evaluate(candidate, X_val, y_val, feature_names, label_map).accuracy
                
                if candidate_score > state.best_val_score + self.config.min_delta:
                    state.skill = candidate
                    state.best_val_score = candidate_score
                    state.best_skill = copy.deepcopy(candidate)
                    state.patience_counter = 0
                    
                    # Update momentum
                    for pattern in text_gradient.get("diagnosis_patterns", []):
                        state.momentum[pattern] = (
                            state.momentum.get(pattern, 0) * self.config.momentum_decay + 1
                        )
                else:
                    state.rejected_buffer.append(text_gradient)
                    state.patience_counter += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            val_score = self.evaluator.evaluate(state.skill, X_val, y_val, feature_names, label_map).accuracy
            
            state.history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_accuracy": val_score,
                "n_rules": len(state.skill.rules),
                "momentum_dims": len(state.momentum),
            })
            
            if self.config.verbose:
                print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, "
                      f"val_acc={val_score:.3f}, "
                      f"rules={len(state.skill.rules)}, "
                      f"rejected={len(state.rejected_buffer)}, "
                      f"patience={state.patience_counter}")
            
            if progress_callback:
                progress_callback(state)
            
            # Meta-review trigger
            if len(state.rejected_buffer) >= self.config.meta_review_threshold:
                if self.config.verbose:
                    print(f"  → Triggering meta-review ({len(state.rejected_buffer)} rejected edits)")
                state.skill = self._meta_review(state)
                state.rejected_buffer.clear()
            
            # Early stopping
            if state.patience_counter >= self.config.patience:
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch} (patience={self.config.patience})")
                break
        
        # Return best skill
        result = state.best_skill or state.skill
        result.performance = {
            "accuracy": state.best_val_score,
            "n_rules": len(result.rules),
            "epochs_trained": state.epoch,
        }
        
        return result
    
    def _meta_review(self, state: OptimizationState) -> Skill:
        """Meta-review: rethink optimization strategy when many edits are rejected."""
        if self.diagnoser is None:
            return state.skill
        
        meta_gradient = self.diagnoser.meta_diagnose(
            skill=state.skill,
            rejected_history=state.rejected_buffer[-5:],  # last 5 rejected plans
            momentum=state.momentum,
        )
        
        return self.editor.apply_edits(
            skill=copy.deepcopy(state.skill),
            edit_plan=meta_gradient,
            max_adds=self.config.max_adds_per_step,
            max_deletes=self.config.max_deletes_per_step,
            max_modifies=self.config.max_modifies_per_step,
        )
