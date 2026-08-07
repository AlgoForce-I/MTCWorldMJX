"""Continual World CL metrics (average performance, forgetting, FT, BT).

Formulas follow Wołczyk et al., "Continual World" (NeurIPS 2021), §4.1 and
Appendix E.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CLScores:
    """Aggregate and per-task Continual World scores.

    Attributes:
        average_performance: ``P(T)``, mean final success over tasks.
        forgetting: mean ``F_i`` over tasks.
        backward_transfer: mean ``B_i`` over tasks (Appendix E.1).
        forward_transfer: mean ``FT_i`` if baselines were provided, else ``None``.
        per_task_forgetting: length-``N`` array of ``F_i``.
        per_task_backward_transfer: length-``N`` array of ``B_i``.
        per_task_forward_transfer: length-``N`` array of ``FT_i``, or ``None``.
        end_of_training_success: ``p_i(i·Δ)`` used as input.
        final_success: ``p_i(T)`` used as input.
    """

    average_performance: float
    forgetting: float
    backward_transfer: float
    forward_transfer: float | None
    per_task_forgetting: np.ndarray
    per_task_backward_transfer: np.ndarray
    per_task_forward_transfer: np.ndarray | None
    end_of_training_success: np.ndarray
    final_success: np.ndarray


def _as_1d_float(name: str, values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def auc(success_curve: ArrayLike) -> float:
    """Discrete AUC: mean success over a task's training window."""
    curve = _as_1d_float("success_curve", success_curve)
    return float(np.mean(curve))


def compute_cl_scores(
    end_of_training_success: ArrayLike,
    final_success: ArrayLike,
    *,
    train_auc: ArrayLike | None = None,
    baseline_auc: ArrayLike | None = None,
) -> CLScores:
    """Compute Continual World CL scores from per-task success rates.

    Inputs
    ------
    end_of_training_success:
        Length-``N`` array. Entry ``i`` is ``p_i(i·Δ)``: success rate on task
        ``i`` evaluated when training of task ``i`` finishes (after ``Δ`` steps
        on that task). Typical collection: after training task ``i``, evaluate
        task ``i`` (or take the diagonal of a full eval matrix).
    final_success:
        Length-``N`` array. Entry ``i`` is ``p_i(T)``: success rate on task
        ``i`` at the end of the full sequence (``T = N·Δ``). Typical
        collection: after the last task, evaluate all tasks ``0..N-1``.
    train_auc:
        Optional length-``N`` array of ``AUC_i``: mean success on task ``i``
        during its own training window ``[(i-1)·Δ, i·Δ]``. Required together
        with ``baseline_auc`` to compute forward transfer.
    baseline_auc:
        Optional length-``N`` array of ``AUC_i^b`` from separate single-task
        runs of each task (same budget ``Δ``). Required together with
        ``train_auc`` for forward transfer.

    Outputs
    -------
    CLScores with:

    - ``average_performance`` = ``P(T) = mean_i final_success[i]``
    - ``per_task_forgetting[i]`` = ``F_i = end_of_training_success[i] - final_success[i]``
    - ``forgetting`` = ``mean_i F_i``
    - ``per_task_backward_transfer[i]`` = ``B_i = max(0, final_success[i] - end_of_training_success[i])``
    - ``backward_transfer`` = ``mean_i B_i``
    - ``per_task_forward_transfer[i]`` = ``FT_i = (AUC_i - AUC_i^b) / (1 - AUC_i^b)``
      when both AUC arrays are given; else ``None``
    - ``forward_transfer`` = ``mean_i FT_i`` when FT is computed; else ``None``

    Success values are expected in ``[0, 1]``. Forward transfer is undefined
    for a task when ``baseline_auc[i]`` is (numerically) 1; those entries become
    ``nan`` and are ignored in the mean.
    """
    eot = _as_1d_float("end_of_training_success", end_of_training_success)
    final = _as_1d_float("final_success", final_success)
    if eot.shape != final.shape:
        raise ValueError(
            "end_of_training_success and final_success must have the same shape, "
            f"got {eot.shape} vs {final.shape}"
        )

    per_task_f = eot - final
    per_task_b = np.maximum(0.0, final - eot)
    average_performance = float(np.mean(final))
    forgetting = float(np.mean(per_task_f))
    backward_transfer = float(np.mean(per_task_b))

    per_task_ft: np.ndarray | None = None
    forward_transfer: float | None = None
    if train_auc is not None or baseline_auc is not None:
        if train_auc is None or baseline_auc is None:
            raise ValueError("train_auc and baseline_auc must both be provided for FT")
        train = _as_1d_float("train_auc", train_auc)
        baseline = _as_1d_float("baseline_auc", baseline_auc)
        if train.shape != eot.shape or baseline.shape != eot.shape:
            raise ValueError(
                "train_auc and baseline_auc must match end_of_training_success shape "
                f"{eot.shape}, got {train.shape} and {baseline.shape}"
            )
        denom = 1.0 - baseline
        with np.errstate(divide="ignore", invalid="ignore"):
            per_task_ft = (train - baseline) / denom
        per_task_ft = np.where(np.abs(denom) < 1e-12, np.nan, per_task_ft)
        forward_transfer = float(np.nanmean(per_task_ft))

    return CLScores(
        average_performance=average_performance,
        forgetting=forgetting,
        backward_transfer=backward_transfer,
        forward_transfer=forward_transfer,
        per_task_forgetting=per_task_f,
        per_task_backward_transfer=per_task_b,
        per_task_forward_transfer=per_task_ft,
        end_of_training_success=eot,
        final_success=final,
    )


def scores_from_eval_matrix(
    success_matrix: ArrayLike,
    *,
    train_auc: ArrayLike | None = None,
    baseline_auc: ArrayLike | None = None,
) -> CLScores:
    """Compute CL scores from a square eval matrix.

    ``success_matrix[t, i]`` is the success rate on task ``i`` evaluated after
    finishing training of task ``t`` (0-based). Uses the diagonal as
    ``p_i(i·Δ)`` and the last row as ``p_i(T)``.
    """
    matrix = np.asarray(success_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"success_matrix must be square 2-D, got shape {matrix.shape}"
        )
    if matrix.size == 0:
        raise ValueError("success_matrix must be non-empty")
    return compute_cl_scores(
        np.diag(matrix),
        matrix[-1],
        train_auc=train_auc,
        baseline_auc=baseline_auc,
    )
