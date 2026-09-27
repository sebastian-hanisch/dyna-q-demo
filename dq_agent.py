"""Dyna-Q (Sutton 1990): Q-Learning UND ein selbst gelerntes Modell, das zwischen den echten Schritten zusaetzliche "Planungs"-Updates liefert.
Das Modell speichert fuer jedes real besuchte (State, Action) nur den ZULETZT beobachteten Ausgang (Reward, Folgezustand) - eine bewusst einfache
Tabellen-Annahme (Sutton & Barto, Kapitel 8.2), die bei n_planning=0 EXAKT auf reines Q-Learning reduziert (kein Modell wird je benutzt)."""

import numpy as np

import dq_constants as C
from dq_grid import ACTIONS, step

N_ACTIONS = len(ACTIONS)


def epsilon_by_episode(episode, start, decay, eps_min=C.EPSILON_MIN):
    if decay <= 0.0:
        return start
    return max(eps_min, start / (1.0 + decay * episode))


def choose_action(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    row = Q[state]
    best = np.flatnonzero(row == row.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


def q_update(Q, s, a, r, s_next, done, alpha, gamma):
    target = r if done else r + gamma * Q[s_next].max()
    Q[s, a] += alpha * (target - Q[s, a])


def run_episode(grid, Q, model, n_planning, epsilon, alpha, rng, max_steps=C.MAX_STEPS_PER_EPISODE):
    """Ein Real-Update pro Schritt, danach `n_planning` zusaetzliche Updates aus dem gelernten Modell (zufaellig gewaehlte, schon einmal real
    beobachtete State-Action-Paare). Rueckgabe: Ertrag, Schrittzahl, Zahl der Klippen-/Gefahren-Abstuerze."""
    gamma = grid.gamma
    s = grid.state_of(grid.start)
    total_reward, steps, falls = 0.0, 0, 0
    for _ in range(max_steps):
        a = choose_action(Q, s, epsilon, rng)
        s_next, r, done = step(grid, s, a, rng)
        if r == C.CLIFF_PENALTY:
            falls += 1
        q_update(Q, s, a, r, s_next, done, alpha, gamma)
        model[(s, a)] = (r, s_next, done)
        if n_planning > 0 and model:
            keys = list(model.keys())
            n = len(keys)
            for _ in range(n_planning):
                ks, ka = keys[int(rng.integers(n))]
                kr, ks_next, kdone = model[(ks, ka)]
                q_update(Q, ks, ka, kr, ks_next, kdone, alpha, gamma)
        total_reward += r
        steps += 1
        if done:
            break
        s = s_next
    return total_reward, steps, falls


def train(grid, alpha, epsilon_start, epsilon_decay, n_planning, episodes, seed, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=()):
    """Trainiert `episodes` Episoden auf EINEM festen Raster."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((grid.n_states, N_ACTIONS))
    model = {}
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    for e in range(episodes):
        eps = epsilon_by_episode(e, epsilon_start, epsilon_decay)
        total_reward, steps, n_falls = run_episode(grid, Q, model, n_planning, eps, alpha, rng, max_steps)
        returns[e] = total_reward
        lengths[e] = steps
        falls[e] = n_falls
        if (e + 1) in checkpoint_set:
            snapshots[e + 1] = Q.copy()
    return Q, returns, lengths, falls, snapshots


def train_switching(grid_before, grid_after, episodes_before, episodes_after, alpha, epsilon_start, epsilon_decay, n_planning, seed, max_steps=C.MAX_STEPS_PER_EPISODE):
    """Blocking-Maze-Experiment (Sutton & Barto, Kapitel 8.3): `episodes_before` Episoden auf `grid_before`, danach wird OHNE das gelernte Q/Modell
    zurueckzusetzen auf `grid_after` umgeschaltet (die Umgebung "verschlechtert sich unter den Fuessen" des Agenten) - fuer weitere `episodes_after`
    Episoden. Rueckgabe: Q, Ertraege und Faelle ueber BEIDE Phasen zusammenhaengend (Index 0 = erste Episode vor dem Wechsel)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((grid_before.n_states, N_ACTIONS))
    model = {}
    total = episodes_before + episodes_after
    returns = np.zeros(total)
    lengths = np.zeros(total, dtype=int)
    falls = np.zeros(total, dtype=int)
    for e in range(total):
        grid = grid_before if e < episodes_before else grid_after
        eps = epsilon_by_episode(e, epsilon_start, epsilon_decay)
        total_reward, steps, n_falls = run_episode(grid, Q, model, n_planning, eps, alpha, rng, max_steps)
        returns[e] = total_reward
        lengths[e] = steps
        falls[e] = n_falls
    return Q, returns, lengths, falls
