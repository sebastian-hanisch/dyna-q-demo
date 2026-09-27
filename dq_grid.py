"""Das Vehikel: derselbe Lagerroboter auf demselben Raster wie in `q-learning-demo`/`sarsa-demo` (Cliff-Walking-Vorlage, Sutton & Barto 2018). Neu hier:
ein optionales Feld `blocked` - zusaetzliche Gefahrenzellen ueber die feste Klippe hinaus, die genauso wirken (Reward -100, Reset an den Start). Damit
laesst sich dieselbe Umgebung mitten im Training "verschlechtern" (Sutton & Bartos Blocking-Maze-Motiv, Kapitel 8.3): eine zuvor sichere Zelle wird
zur Gefahr, waehrend der Agent noch sein altes (jetzt veraltetes) Modell davon im Kopf hat."""

from dataclasses import dataclass, field

import numpy as np

import dq_constants as C

NORTH, SOUTH, EAST, WEST = range(4)
ACTIONS = (NORTH, SOUTH, EAST, WEST)
ACTION_NAMES = {NORTH: "Norden", SOUTH: "Süden", EAST: "Osten", WEST: "Westen"}
ACTION_ARROWS = {NORTH: "↑", SOUTH: "↓", EAST: "→", WEST: "←"}
_DELTA = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1)}
_ORTHOGONAL = {NORTH: (EAST, WEST), SOUTH: (EAST, WEST), EAST: (NORTH, SOUTH), WEST: (NORTH, SOUTH)}


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    slip: float
    gamma: float
    blocked: frozenset = field(default_factory=frozenset)

    @property
    def start(self):
        return (self.rows - 1, 0)

    @property
    def goal(self):
        return (self.rows - 1, self.cols - 1)

    @property
    def cliff(self):
        return frozenset((self.rows - 1, c) for c in range(1, self.cols - 1))

    @property
    def hazard(self):
        """Alle Gefahrenzellen: die feste Klippe PLUS die zusaetzlich blockierten Zellen (leer im Normalfall)."""
        return self.cliff | self.blocked

    @property
    def n_states(self):
        return self.rows * self.cols

    def state_of(self, rc):
        r, c = rc
        return r * self.cols + c

    def rc_of(self, s):
        return divmod(s, self.cols)

    def cell_kind(self, rc):
        if rc == self.goal:
            return "goal"
        if rc == self.start:
            return "start"
        if rc in self.blocked:
            return "blocked"
        if rc in self.cliff:
            return "cliff"
        return "free"


def _resolve(grid, rc, direction):
    r, c = rc
    dr, dc = _DELTA[direction]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
        nr, nc = r, c
    return (nr, nc)


def step(grid, state, action, rng):
    """Ein einzelner gesampelter Uebergang - das einzige, was der lernende Agent von der ECHTEN Umgebung sieht (nie das gelernte Modell)."""
    rc = grid.rc_of(state)
    if rc == grid.goal:
        return state, 0.0, True
    u = rng.random()
    direction = action if u < 1.0 - grid.slip else _ORTHOGONAL[action][0] if u < 1.0 - grid.slip / 2.0 else _ORTHOGONAL[action][1]
    nrc = _resolve(grid, rc, direction)
    if nrc == grid.goal:
        return grid.state_of(nrc), C.GOAL_REWARD, True
    if nrc in grid.hazard:
        return grid.state_of(grid.start), C.CLIFF_PENALTY, False
    return grid.state_of(nrc), C.STEP_COST, False


def build_model(grid):
    """Modell (P, R): NUR fuer die Referenzloesung/Gegenprobe, niemals fuer den lernenden Agenten selbst."""
    S, A = grid.n_states, len(ACTIONS)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    goal_s = grid.state_of(grid.goal)
    start_s = grid.state_of(grid.start)
    hazard_states = {grid.state_of(rc) for rc in grid.hazard}
    for s in range(S):
        r, c = grid.rc_of(s)
        if s == goal_s:
            for a in ACTIONS:
                P[s, a, s] = 1.0
            continue
        for a in ACTIONS:
            outcomes = [(1.0 - grid.slip, a)] + [(grid.slip / 2.0, oa) for oa in _ORTHOGONAL[a]]
            for prob, direction in outcomes:
                nrc = _resolve(grid, (r, c), direction)
                ns = grid.state_of(nrc)
                if ns == goal_s:
                    reward, target = C.GOAL_REWARD, goal_s
                elif ns in hazard_states:
                    reward, target = C.CLIFF_PENALTY, start_s
                else:
                    reward, target = C.STEP_COST, ns
                P[s, a, target] += prob
                R[s, a] += prob * reward
    return P, R
