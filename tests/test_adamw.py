import pytest
import numpy as np
from src.parameter import Parameter
from src.optim.adamw import AdamW

def test_adamw_reference_numerical_update():
    p = Parameter([1.0, -2.0])
    p.grad = np.array([0.1, -0.2])

    opt = AdamW([p], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    opt.step()

    # Expected reference calculation:
    # m1 = 0.1 * g = [0.01, -0.02] -> m_hat1 = [0.1, -0.2]
    # v1 = 0.001 * g^2 = [0.00001, 0.00004] -> v_hat1 = [0.01, 0.04]
    # Decoupled WD: p = [1.0, -2.0] * (1 - 0.01 * 0.1) = [0.999, -1.998]
    # Update: p = [0.999, -1.998] - 0.01 * [0.1 / sqrt(0.01), -0.2 / sqrt(0.04)]
    #        = [0.999, -1.998] - 0.01 * [1.0, -1.0] = [0.989, -1.988]
    expected_data = np.array([0.989, -1.988])
    np.testing.assert_allclose(p.data, expected_data, atol=1e-6)

def test_adamw_parameter_groups_decay_vs_no_decay():
    p_decay = Parameter([1.0, 2.0])
    p_decay.grad = np.array([0.0, 0.0]) # zero gradient

    p_no_decay = Parameter([1.0, 2.0])
    p_no_decay.grad = np.array([0.0, 0.0]) # zero gradient

    opt = AdamW([
        {"params": [p_decay], "weight_decay": 0.1},
        {"params": [p_no_decay], "weight_decay": 0.0}
    ], lr=0.01)

    opt.step()

    # p_decay should decrease due to weight decay even with zero gradient: 1.0 * (1 - 0.01 * 0.1) = 0.999
    np.testing.assert_allclose(p_decay.data, [0.999, 1.998], atol=1e-6)

    # p_no_decay should remain unchanged with zero gradient and zero weight decay
    np.testing.assert_allclose(p_no_decay.data, [1.0, 2.0], atol=1e-6)
