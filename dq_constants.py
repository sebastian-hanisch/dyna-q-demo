"""Konstanten der Demo "Dyna-Q" (Stück 5 der Reinforcement-Learning-Linie): dasselbe Raster wie `q-learning-demo`/`sarsa-demo`, dazu die
Lernparameter UND die Konfiguration des "Blocking-Maze"-Experiments (Sutton & Barto 2018, Kapitel 8.3: die Umgebung verschlechtert sich mitten im
Training, das gelernte Modell ist dann kurzzeitig veraltet)."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster (wie q-learning-demo/sarsa-demo, Cliff-Walking-Vorlage) -----------------------------------------------------------------------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 4
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 8
# DEFAULT_SLIP=0 weicht bewusst von value-iteration-demo/q-learning-demo (0,10) ab, wie schon bei sarsa-demo: gemessen (15 Seeds, 50 Episoden, n=10),
# dass bei Rutschen 0,10 die Stichprobeneffizienz von Dyna-Q INS GEGENTEIL verkehrt ist (Wert-Abstand 12,73 gegen 1,18 ohne Planung; bei 0,02 noch
# kein Nachteil: 2,90 gegen 3,38) - das gelernte Modell merkt sich nur den zuletzt beobachteten Ausgang je Zustand-Aktion-Paar, eine unter Rutschen
# verzerrte Stichprobe (siehe README/Grenzen-Tabelle, eigener Befund dieser Demo).
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.0
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

# --- Referenzlösung (Value Iteration, nur zur Gegenprobe) -------------------------------------------------------------------------------------------
VI_TOL = 1e-8
VI_MAX_ITER = 5000

# --- Lernparameter (Dyna-Q: Q-Learning + n Planungsschritte je echtem Schritt mit einem selbst gelernten Modell) -------------------------------------
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.05, 0.50, 0.05, 0.10
EPSILON_START_MIN, EPSILON_START_MAX, EPSILON_START_STEP, DEFAULT_EPSILON_START = 0.20, 1.00, 0.05, 1.00
EPSILON_DECAY_MIN, EPSILON_DECAY_MAX, EPSILON_DECAY_STEP, DEFAULT_EPSILON_DECAY = 0.0, 0.02, 0.001, 0.005
EPSILON_MIN = 0.01

N_PLANNING_MIN, N_PLANNING_MAX, N_PLANNING_STEP, DEFAULT_N_PLANNING = 0, 50, 5, 10

# EPISODES-Bereich bewusst klein: Vormessung zeigt, dass sich der ganze Stichproben-Unterschied zwischen 2 und ~20 Episoden abspielt - reines
# Q-Learning (n=0) braucht dort noch 10+ Punkte Wert-Abstand, jedes n>0 ist praktisch schon am Ziel (siehe README).
EPISODES_MIN, EPISODES_MAX, EPISODES_STEP, DEFAULT_EPISODES = 5, 100, 1, 12
MAX_STEPS_PER_EPISODE = 400

# --- Experimente (feste Konfigurationen) --------------------------------------------------------------------------------------------------------------
# EXP_SEEDS bewusst kleiner als bei den Geschwistern, n=50 aus dem Experiment-Sweep entfernt: ein einzelner Dyna-Q-Lauf ist bis zu 51x teurer als
# reines Q-Learning (n zusaetzliche Planungs-Updates je echtem Schritt), 30 Seeds x n=50 waeren fuer "Dauer bis zu einer Minute" zu langsam (gemessen: 192s).
EXP_SEEDS = 15
EXP_N_LEVELS = (0, 5, 10, 25)
EXP_EPISODE_CHECKPOINTS = (2, 4, 8, 12, 20)

NEAR_OPTIMAL_GAP = 0.5
CLIFF_GAP_THRESHOLD = 50.0

# --- Blocking-Maze-Experiment (Umgebung verschlechtert sich nach BLOCK_EPISODES_BEFORE; slip=0, damit die Route eindeutig ist) -----------------------
BLOCK_EPISODES_BEFORE = 30
BLOCK_EPISODES_AFTER = 30
BLOCK_COLS = (2, 3, 4)        # drei Zellen der Reihe direkt ueber der Klippe (rows-2) werden ab dem Wechsel zusaetzlich zur Gefahr
