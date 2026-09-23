"""Reusable peak-memory planner for dense tensor contractions."""

from .core import (
    MAX_TENSORS,
    ContractionPlan,
    ContractionPlannerError,
    InputTensor,
    InvalidProblemError,
    InfeasiblePlanError,
    OperandRef,
    PeakEvidence,
    PlanStep,
    Resident,
    SearchBudgetExhaustedError,
    Tensor,
    plan_contraction,
)
from .evaluator import DenseArray, execute_plan

__all__ = [
    "MAX_TENSORS",
    "Tensor",
    "InputTensor",
    "Resident",
    "OperandRef",
    "PlanStep",
    "PeakEvidence",
    "ContractionPlan",
    "ContractionPlannerError",
    "InvalidProblemError",
    "InfeasiblePlanError",
    "SearchBudgetExhaustedError",
    "plan_contraction",
    "DenseArray",
    "execute_plan",
]
