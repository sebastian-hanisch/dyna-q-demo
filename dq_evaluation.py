"""Auswertung: Dyna-Q gegen die Value-Iteration-Referenz, dazu zwei Experimente - Stichprobeneffizienz (Wert-Abstand ueber Trainingsdauer, verschiedene
Planungsschritte n) und das Blocking-Maze-Experiment (Sutton & Barto, Kapitel 8.3: die Umgebung verschlechtert sich mitten im Training)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dq_agent as A
import dq_constants as C
import dq_grid as G
import dq_reference as R


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA
    alpha: float = C.DEFAULT_ALPHA
    epsilon_start: float = C.DEFAULT_EPSILON_START
    epsilon_decay: float = C.DEFAULT_EPSILON_DECAY
    n_planning: int = C.DEFAULT_N_PLANNING
    episodes: int = C.DEFAULT_EPISODES
    seed: int = 0

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _reference(rows, cols, slip, gamma):
    grid = G.Grid(rows, cols, slip, gamma)
    P, Rw = G.build_model(grid)
    V_star, Q_star, pi_star = R.value_iteration(P, Rw, gamma)
    return grid, P, Rw, V_star, Q_star, pi_star


@dataclass
class Analysis:
    settings: Settings
    grid: G.Grid
    Q: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    falls: np.ndarray
    policy: np.ndarray
    V_star: np.ndarray
    pi_star: np.ndarray
    V_pi: np.ndarray
    gap: float
    env_steps: int
    snapshots: dict


def analyse(s, checkpoints=()):
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    Q, returns, lengths, falls, snapshots = A.train(grid, s.alpha, s.epsilon_start, s.epsilon_decay, s.n_planning, s.episodes, s.seed, checkpoints=checkpoints)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, grid.gamma)
    start_s = grid.state_of(grid.start)
    gap = float(V_star[start_s] - V_pi[start_s])
    return Analysis(s, grid, Q, returns, lengths, falls, policy, V_star, pi_star, V_pi, gap, int(lengths.sum()), snapshots)


def policy_gap(rows, cols, slip, gamma, alpha, epsilon_start, epsilon_decay, n_planning, episodes, seed):
    grid, P, Rw, V_star, _, _ = _reference(rows, cols, slip, gamma)
    Q, _, lengths, _, _ = A.train(grid, alpha, epsilon_start, epsilon_decay, n_planning, episodes, seed)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, gamma)
    start_s = grid.state_of(grid.start)
    return float(V_star[start_s] - V_pi[start_s]), int(lengths.sum())


def _summary(gaps):
    gaps = np.asarray(gaps, dtype=float)
    se = float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
    return {"mean": float(gaps.mean()), "se": se, "frac_near_optimal": float(np.mean(gaps < C.NEAR_OPTIMAL_GAP))}


# --- Experiment 1: Stichprobeneffizienz - Wert-Abstand ueber die Trainingsdauer, verschiedene n -------------------------------------------------------

def sample_efficiency_experiment(n_levels=None, checkpoints=None, seeds=None, base=None):
    n_levels = C.EXP_N_LEVELS if n_levels is None else n_levels
    checkpoints = C.EXP_EPISODE_CHECKPOINTS if checkpoints is None else checkpoints
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for n in n_levels:
        row = {"n_planning": n}
        for episodes in checkpoints:
            gaps, env_steps = [], []
            for sd in seeds:
                gap, steps = policy_gap(base.rows, base.cols, base.slip, base.gamma, base.alpha, base.epsilon_start, base.epsilon_decay, n, episodes, sd)
                gaps.append(gap)
                env_steps.append(steps)
            row[episodes] = {**_summary(gaps), "env_steps_mean": float(np.mean(env_steps))}
        rows.append(row)
    return {"n_seeds": len(list(seeds)), "n_levels": tuple(n_levels), "checkpoints": tuple(checkpoints), "rows": rows}


# --- Experiment 2: Blocking-Maze - die Umgebung verschlechtert sich mitten im Training -----------------------------------------------------------------

def blocking_experiment(n_levels=None, seeds=None, episodes_before=None, episodes_after=None, base=None):
    n_levels = C.EXP_N_LEVELS if n_levels is None else n_levels
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    episodes_before = C.BLOCK_EPISODES_BEFORE if episodes_before is None else episodes_before
    episodes_after = C.BLOCK_EPISODES_AFTER if episodes_after is None else episodes_after
    # Konstantes, kleines Epsilon statt des GLIE-Zerfalls der Settings-Vorgabe: das Blocking-Maze-Experiment misst den ECHTEN Ertrag waehrend des
    # Trainings (Sutton & Bartos Vergleichsgroesse) - mit Start-Epsilon 1,0 waere der Ertrag noch lange von reinem Explorationsrauschen dominiert,
    # nicht von der gelernten Policy.
    base = Settings(slip=0.0, epsilon_start=0.1, epsilon_decay=0.0) if base is None else base
    row_above_cliff = base.rows - 2
    blocked = frozenset((row_above_cliff, c) for c in C.BLOCK_COLS)
    grid_before = G.Grid(base.rows, base.cols, 0.0, base.gamma)
    grid_after = G.Grid(base.rows, base.cols, 0.0, base.gamma, blocked=blocked)
    rows = []
    for n in n_levels:
        all_returns = []
        for sd in seeds:
            _, returns, _, _ = A.train_switching(grid_before, grid_after, episodes_before, episodes_after, base.alpha, base.epsilon_start, base.epsilon_decay, n, sd)
            all_returns.append(returns)
        arr = np.array(all_returns)
        rows.append({
            "n_planning": n,
            "before_last": float(arr[:, max(0, episodes_before - 10):episodes_before].mean()),
            "after_first": float(arr[:, episodes_before:episodes_before + 10].mean()),
            "after_last": float(arr[:, -10:].mean()),
            "mean_curve": arr.mean(axis=0).tolist(),
        })
    return {"n_seeds": len(list(seeds)), "n_levels": tuple(n_levels), "episodes_before": episodes_before, "episodes_after": episodes_after, "rows": rows}
