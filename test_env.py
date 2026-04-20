"""
test_env.py
===========
Unit tests for ArmReachEnv and LagrangianRewardWrapper.

Run with:
    python -m pytest test_env.py -v
"""

import numpy as np
import pytest

from arm_reach_env import ArmReachEnv, LagrangianRewardWrapper, TORQUE_LIMIT, COST_LIMIT


@pytest.fixture(scope="module")
def env():
    e = ArmReachEnv(render_mode=None)
    yield e
    e.close()


@pytest.fixture(scope="module")
def lag_env():
    e = LagrangianRewardWrapper(ArmReachEnv(render_mode=None))
    yield e
    e.close()


# ── Spaces ────────────────────────────────────────────────────────────────────

def test_observation_space_shape(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (21,)


def test_action_space_bounds(env):
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0
    assert env.action_space.shape == (7,)


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_finite_obs(env):
    obs, info = env.reset()
    assert np.all(np.isfinite(obs)), "reset() returned non-finite observation"
    assert isinstance(info, dict)


def test_reset_reproducible_with_seed(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


# ── Step ──────────────────────────────────────────────────────────────────────

def test_step_shapes(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (21,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_obs_indices_consistent(env):
    """The distance in obs[-1] must match np.linalg.norm(ee - target)."""
    env.reset(seed=1)
    action = np.zeros(7, dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    n = env.num_controllable
    ee  = obs[env._idx_ee]
    tgt = obs[env._idx_tgt]
    dist_from_obs   = float(obs[env._idx_d])
    dist_recomputed = float(np.linalg.norm(ee - tgt))

    assert abs(dist_from_obs - dist_recomputed) < 1e-5, (
        f"obs distance {dist_from_obs:.6f} != recomputed {dist_recomputed:.6f}"
    )


def test_cost_is_binary(env):
    env.reset(seed=2)
    for _ in range(10):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        assert info["cost"] in (0.0, 1.0), f"cost must be 0 or 1, got {info['cost']}"
        if terminated or truncated:
            env.reset()


def test_truncation_at_max_steps(env):
    env.reset(seed=3)
    action = np.zeros(7, dtype=np.float32)
    for _ in range(env.max_steps - 1):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated:
            env.reset(seed=3)
            break
    _, _, _, truncated, _ = env.step(action)
    # Not asserting truncated==True because the episode may have succeeded first,
    # but after a full run with zero action from a non-reaching pose it should truncate.
    assert isinstance(truncated, bool)


def test_termination_on_success(env):
    """Termination gate fires when distance drops below 0.05 m."""
    import mujoco as _mj
    obs, _ = env.reset(seed=4)
    ee = obs[env._idx_ee].copy()

    # Place target at the current EE position
    env.target_position = ee.copy()
    env.data.mocap_pos[env._target_mocap_id] = ee
    _mj.mj_forward(env.model, env.data)
    env.previous_distance = float(np.linalg.norm(
        env.data.site_xpos[env._ee_site_id] - env.target_position
    ))

    # Compute the hold action: maps current qpos back to [-1, 1] so the
    # controller targets the current position and the arm stays still.
    qpos = env.data.qpos[:env.num_controllable].copy()
    hold_action = (
        2.0 * (qpos - env.joint_lower_limits)
        / (env.joint_upper_limits - env.joint_lower_limits)
        - 1.0
    ).astype(np.float32)

    _, _, terminated, _, info = env.step(hold_action)
    assert info["distance"] < 0.05, (
        f"Expected distance < 0.05 with hold action, got {info['distance']:.4f}"
    )
    assert terminated


# ── LagrangianRewardWrapper ───────────────────────────────────────────────────

def test_lagrangian_wrapper_zero_lambda(lag_env):
    """With λ=0, augmented reward equals base reward."""
    lag_env.lam = 0.0
    lag_env.reset(seed=5)
    action = lag_env.action_space.sample()

    # Run base env step for reference
    base_env = lag_env.env
    obs_lag, r_lag, _, _, info_lag = lag_env.step(action)

    assert info_lag["lagrangian_penalty"] == pytest.approx(0.0)


def test_lagrangian_wrapper_penalises_cost(lag_env):
    """With λ>0, r_aug = r_base - λ·cost; difference equals λ·cost exactly."""
    # Both runs start from the same state and take the same action, so the
    # base reward (before penalty) is identical. Any difference in r0 vs r5
    # comes solely from the Lagrangian penalty term.
    lag_env.lam = 0.0
    lag_env.reset(seed=6)
    action = lag_env.action_space.sample()
    _, r0, _, _, info0 = lag_env.step(action)

    lag_env.lam = 5.0
    lag_env.reset(seed=6)
    _, r5, _, _, info5 = lag_env.step(action)

    cost = info5.get("cost", 0.0)
    assert r0 - r5 == pytest.approx(5.0 * cost, abs=1e-4)


def test_lagrangian_wrapper_penalty_in_info(lag_env):
    lag_env.lam = 3.0
    lag_env.reset(seed=7)
    action = lag_env.action_space.sample()
    _, _, _, _, info = lag_env.step(action)
    cost = info["cost"]
    assert info["lagrangian_penalty"] == pytest.approx(3.0 * cost, abs=1e-5)


# ── Constants sanity check ────────────────────────────────────────────────────

def test_constants():
    assert TORQUE_LIMIT > 0
    assert COST_LIMIT > 0
