"""Unit tests for Continual World CL metrics."""

from __future__ import annotations

import numpy as np
import pytest

from MTCWorldMJX.cw_metrics import auc, compute_cl_scores, scores_from_eval_matrix


def test_perfect_retention() -> None:
    eot = np.array([0.8, 0.7, 0.9])
    final = eot.copy()
    scores = compute_cl_scores(eot, final)
    assert scores.average_performance == pytest.approx(0.8)
    assert scores.forgetting == pytest.approx(0.0)
    assert scores.backward_transfer == pytest.approx(0.0)
    assert scores.forward_transfer is None


def test_catastrophic_forgetting() -> None:
    eot = np.array([1.0, 1.0, 1.0])
    final = np.array([0.0, 0.0, 1.0])
    scores = compute_cl_scores(eot, final)
    assert scores.average_performance == pytest.approx(1.0 / 3.0)
    assert scores.forgetting == pytest.approx(2.0 / 3.0)
    assert scores.backward_transfer == pytest.approx(0.0)
    np.testing.assert_allclose(scores.per_task_forgetting, [1.0, 1.0, 0.0])


def test_backward_transfer_floored_at_zero() -> None:
    eot = np.array([0.5, 0.4])
    final = np.array([0.7, 0.3])  # task 0 improved, task 1 forgot
    scores = compute_cl_scores(eot, final)
    np.testing.assert_allclose(scores.per_task_forgetting, [-0.2, 0.1])
    np.testing.assert_allclose(scores.per_task_backward_transfer, [0.2, 0.0])
    assert scores.backward_transfer == pytest.approx(0.1)


def test_forward_transfer() -> None:
    eot = np.array([0.5, 0.5])
    final = np.array([0.5, 0.5])
    train_auc = np.array([0.6, 0.4])
    baseline_auc = np.array([0.5, 0.5])
    scores = compute_cl_scores(
        eot, final, train_auc=train_auc, baseline_auc=baseline_auc
    )
    # FT = (0.6-0.5)/(1-0.5)=0.2, (0.4-0.5)/(1-0.5)=-0.2 → mean 0
    assert scores.forward_transfer == pytest.approx(0.0)
    np.testing.assert_allclose(scores.per_task_forward_transfer, [0.2, -0.2])


def test_auc_mean() -> None:
    assert auc([0.0, 0.5, 1.0]) == pytest.approx(0.5)


def test_scores_from_eval_matrix() -> None:
    # rows = after finishing task t; cols = eval on task i
    matrix = np.array(
        [
            [0.8, 0.0, 0.0],
            [0.5, 0.7, 0.0],
            [0.2, 0.1, 0.9],
        ]
    )
    scores = scores_from_eval_matrix(matrix)
    np.testing.assert_allclose(scores.end_of_training_success, [0.8, 0.7, 0.9])
    np.testing.assert_allclose(scores.final_success, [0.2, 0.1, 0.9])
    assert scores.average_performance == pytest.approx((0.2 + 0.1 + 0.9) / 3)
    assert scores.forgetting == pytest.approx((0.6 + 0.6 + 0.0) / 3)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_cl_scores([0.1, 0.2], [0.1])
