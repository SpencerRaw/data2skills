"""data2skills — Gradient-Optimized Expert Knowledge from Data."""

from .skill import Skill, Rule, seed_skill_from_data
from .optimizer import SkillOptimizer, OptimizerConfig, OptimizationState
from .evaluator import SkillEvaluator, EvalResult
from .diagnosis import FailureDiagnoser
from .editor import SkillEditor

__version__ = "0.1.0"
