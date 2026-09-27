"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dq_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "epsilon_start_slider": SettingSpec("eps0", float, C.DEFAULT_EPSILON_START, C.EPSILON_START_MIN, C.EPSILON_START_MAX),
    "epsilon_decay_slider": SettingSpec("decay", float, C.DEFAULT_EPSILON_DECAY, C.EPSILON_DECAY_MIN, C.EPSILON_DECAY_MAX),
    "n_planning_slider": SettingSpec("n", int, C.DEFAULT_N_PLANNING, C.N_PLANNING_MIN, C.N_PLANNING_MAX),
    "episodes_slider": SettingSpec("episodes", int, C.DEFAULT_EPISODES, C.EPISODES_MIN, C.EPISODES_MAX),
}
PRESET_KEYS = {
    "rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider",
    "alpha": "alpha_slider", "epsilon_start": "epsilon_start_slider", "epsilon_decay": "epsilon_decay_slider",
    "n_planning": "n_planning_slider", "episodes": "episodes_slider",
}
STEPS = {
    "slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP, "alpha_slider": C.ALPHA_STEP,
    "epsilon_start_slider": C.EPSILON_START_STEP, "epsilon_decay_slider": C.EPSILON_DECAY_STEP,
    "n_planning_slider": C.N_PLANNING_STEP, "episodes_slider": C.EPISODES_STEP,
}


def _p(**kw):
    base = {
        "rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA,
        "alpha": C.DEFAULT_ALPHA, "epsilon_start": C.DEFAULT_EPSILON_START, "epsilon_decay": C.DEFAULT_EPSILON_DECAY,
        "n_planning": C.DEFAULT_N_PLANNING, "episodes": C.DEFAULT_EPISODES,
    }
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Reines Q-Learning (n=0)": _p(n_planning=0),
    "Viel Planung": _p(n_planning=50),
    "Mit Rutschen": _p(slip=0.10, episodes=50),
    "Ausreichend trainiert": _p(episodes=20),
    "Großes Raster": _p(rows=6, cols=12, episodes=40),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall": "Nur 12 Episoden, ohne Rutschen: reines Q-Learning (n=0) hat davon noch kaum etwas gelernt (Wert-Abstand zweistellig), Dyna-Q mit n=10 Planungsschritten je echtem Schritt ist schon fast am Ziel - ohne einen einzigen zusätzlichen echten Schritt.",
    "Reines Q-Learning (n=0)": "n=0: das gelernte Modell wird nie zur Planung benutzt (nur geschrieben) - das ist exakt Q-Learning, der Vergleichsmaßstab.",
    "Viel Planung": "n=50: bei deterministischer Umgebung (ohne Rutschen) bringt mehr Planung ab n≈5-10 kaum noch etwas dazu (abnehmender Grenzertrag) - bei Rutschen dagegen kann sie sogar schaden (siehe Preset \"Mit Rutschen\").",
    "Mit Rutschen": "Rutschen 0,10, 50 Episoden: das gelernte Modell speichert nur den ZULETZT beobachteten Ausgang je Zustand-Aktion-Paar - bei einer stochastischen Umgebung ist das eine verzerrte Stichprobe. Planung mit einem verzerrten Modell schadet hier mehr, als sie nützt.",
    "Ausreichend trainiert": "20 Episoden: der Stichproben-Vorsprung von Dyna-Q verschwindet, sobald auch reines Q-Learning (n=0) genug Zeit hatte - der Vorteil gilt nur in der frühen Trainingsphase.",
    "Großes Raster": "6×12-Raster (72 States, 40 Episoden): derselbe Vorteil auf einem größeren Problem mit mehr zu lernenden Zustand-Aktion-Paaren.",
}
