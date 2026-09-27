"""Dyna-Q - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Reinforcement-Learning-Linie der "Konzepte"-Reihe: derselbe Lagerroboter wie bei Q-Learning (Stück 3), aber Dyna-Q (Sutton 1990)
lernt zusätzlich zur Q-Tabelle ein eigenes Modell der Umgebung und "plant" damit zwischen den echten Schritten - zusätzliche Q-Updates, ohne dafür
neue echte Erfahrung zu brauchen.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import dq_constants as C
from dq_evaluation import Settings, analyse, blocking_experiment, sample_efficiency_experiment
from dq_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from dq_visualization import build_blocking_bars, build_blocking_curves, build_env_steps, build_grid, build_sample_efficiency

st.set_page_config(page_title="Dyna-Q – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _analyse(settings, checkpoints):
    return analyse(settings, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _analyse_baseline(settings):
    """Dieselben Einstellungen, aber n_planning=0 - der Vergleichsmassstab (reines Q-Learning)."""
    return analyse(Settings(settings.rows, settings.cols, settings.slip, settings.gamma, settings.alpha, settings.epsilon_start, settings.epsilon_decay, 0, settings.episodes, settings.seed))


@st.cache_data(show_spinner=False)
def _sample_efficiency_exp(base):
    return sample_efficiency_experiment(base=base)


@st.cache_data(show_spinner=False)
def _blocking_exp(base):
    return blocking_experiment(base=base)


st.title("🗺️ Dyna-Q")
st.markdown(
    """
Derselbe Lagerroboter wie bei **Q-Learning** (Stück 3). **Dyna-Q** (Sutton 1990) lernt zusätzlich zur Q-Tabelle ein eigenes **Modell** der Umgebung
(für jedes real besuchte Zustand-Aktion-Paar: der zuletzt beobachtete Reward und Folgezustand) - und "plant" damit zwischen den echten Schritten:
$n$ zusätzliche Q-Updates aus zufällig ausgewählten, schon einmal real erlebten Situationen, ganz ohne dafür neue echte Erfahrung zu brauchen.
Bei $n=0$ wird das Modell nie gelesen - Dyna-Q ist dann exakt Q-Learning. Alle Daten sind erzeugt.
"""
)
st.caption(
    "Fünftes Stück der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe: der Nachfolger von Q-Learning (Stück 3) - dieselbe Lernregel, "
    "zusätzlich ein selbst gelerntes Modell. **Bezug:** anders als Value Iteration (Stück 2, KENNT das Modell) lernt Dyna-Q sein Modell selbst aus "
    "Erfahrung - und dieses gelernte Modell kann falsch oder veraltet sein (siehe Experiment 2)."
)

with st.expander("So funktioniert Dyna-Q", expanded=True):
    st.markdown(
        r"""
1. **Echter Schritt.** Wie Q-Learning: Aktion epsilon-gierig wählen, Übergang beobachten, Q-Tabelle per TD-Update aktualisieren.
2. **Modell merken.** Das Tripel (State, Action) → (Reward, Folgezustand) wird gespeichert - einfach der zuletzt beobachtete Ausgang, keine Verteilung.
3. **Planen.** Danach werden $n$ zusätzliche Updates aus dem Modell gezogen: ein zufälliges, schon einmal real besuchtes (State, Action) auswählen, den gespeicherten (Reward, Folgezustand) einsetzen, genauso ein TD-Update rechnen - als hätte man diesen Schritt gerade noch einmal wirklich erlebt.
4. **Der Haken:** das Modell speichert nur den ZULETZT beobachteten Ausgang. In einer **stochastischen** Umgebung (Rutschen) ist das eine verzerrte Stichprobe - Planung mit einem verzerrten Modell kann dann mehr schaden als nützen (Experiment im Abschnitt "Wo die Annahmen enden").
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Raster**")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f", help="0 = deterministisch: das gelernte Modell ist dann exakt.")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f")
    st.markdown("**Das Lernen**")
    alpha = st.slider("Lernrate α", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP, format="%.2f")
    eps0 = st.slider("Start-Epsilon", *bounds("epsilon_start_slider"), key="epsilon_start_slider", step=C.EPSILON_START_STEP, format="%.2f")
    decay = st.slider("Epsilon-Zerfall", *bounds("epsilon_decay_slider"), key="epsilon_decay_slider", step=C.EPSILON_DECAY_STEP, format="%.3f")
    n_planning = st.slider("Planungsschritte n", *bounds("n_planning_slider"), key="n_planning_slider", step=C.N_PLANNING_STEP, help="0 = reines Q-Learning (das Modell wird nie gelesen).")
    episodes = st.slider("Trainingsepisoden", *bounds("episodes_slider"), key="episodes_slider", step=C.EPISODES_STEP)

sync_query_params({
    "rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3),
    "alpha_slider": round(float(alpha), 3), "epsilon_start_slider": round(float(eps0), 3), "epsilon_decay_slider": round(float(decay), 4),
    "n_planning_slider": int(n_planning), "episodes_slider": int(episodes),
})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(alpha), 3), round(float(eps0), 3), round(float(decay), 4), int(n_planning), int(episodes), seed=0)
frames = tuple(sorted({0} | {int(round(x)) for x in np.linspace(1, settings.episodes, min(settings.episodes, 11))}))
with st.spinner("Dyna-Q trainiert ..."):
    a = _analyse(settings, frames)
grid = a.grid
s0 = grid.state_of(grid.start)

# --- Episode für Episode -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Episode für Episode zur gelernten Policy")
if "dq_frame_idx" not in st.session_state or st.session_state.get("dq_frame_owner") != settings:
    st.session_state["dq_frame_idx"] = len(frames) - 1
    st.session_state["dq_frame_owner"] = settings
idx = st.slider("Trainingsstand", 0, len(frames) - 1, key="dq_frame_idx", help="0 = noch untrainiert (Q überall 0).")
ep = frames[idx]
Q_snap = a.snapshots.get(ep, np.zeros((grid.n_states, 4)))
head = "Vor dem Training (Q überall 0)" if ep == 0 else f"Nach {ep} von {settings.episodes} Episoden"
c1, c2 = st.columns([3, 2])
c1.markdown(f"**{head}**")
c1.plotly_chart(build_grid(grid, Q_snap.max(axis=1), Q_snap.argmax(axis=1) if ep > 0 else None), width="stretch", key=f"dq_grid_{ep}")
c2.markdown("**Ertrag je Episode (bis hierhin)**")
y = a.returns[:ep] if ep > 0 else np.array([0.0])
import plotly.graph_objects as go
fig_ret = go.Figure()
fig_ret.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=y, mode="lines", line=dict(color="#1f77b4", width=2)))
fig_ret.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
fig_ret.update_xaxes(fixedrange=True, title_text="Episode")
fig_ret.update_yaxes(fixedrange=True, title_text="Ertrag")
c2.plotly_chart(fig_ret, width="stretch", key=f"dq_curve_{ep}")

st.markdown("---")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wie viel weniger echte Erfahrung braucht Dyna-Q gegenüber reinem Q-Learning?")
with st.spinner("Vergleichslauf ohne Planung (n=0) ..."):
    baseline = _analyse_baseline(settings)
mcols = st.columns(4)
mcols[0].metric(f"Dyna-Q (n={settings.n_planning}): Wert-Abstand", de(a.gap, 2))
mcols[1].metric("Reines Q-Learning (n=0): Wert-Abstand", de(baseline.gap, 2))
mcols[2].metric(f"Dyna-Q: Umgebungsschritte", f"{a.env_steps:,}".replace(",", "."))
mcols[3].metric("Q-Learning: Umgebungsschritte", f"{baseline.env_steps:,}".replace(",", "."), delta=int(baseline.env_steps - a.env_steps), delta_color="off" if settings.n_planning == 0 else "normal")
if settings.n_planning == 0:
    st.info("ℹ️ n=0: Dyna-Q liest sein Modell nie - dieser Lauf IST reines Q-Learning (siehe Formel unten für den strukturellen Beweis).")
elif settings.slip == 0.0 and a.gap <= baseline.gap + 0.5:
    st.success(f"✅ Bei derselben Episodenzahl erreicht Dyna-Q einen kleineren oder gleich guten Wert-Abstand ({de(a.gap,2)} gegen {de(baseline.gap,2)}) - die zusätzlichen Planungs-Updates ersetzen einen Teil der fehlenden echten Erfahrung.")
else:
    st.warning(f"⚠️ Hier hilft Planung nicht (Abstand {de(a.gap,2)} gegen {de(baseline.gap,2)} ohne Planung) - bei Rutschen > 0 ist das gelernte Modell (nur der zuletzt beobachtete Ausgang) eine verzerrte Stichprobe der echten, stochastischen Übergänge (siehe Grenzen-Tabelle).")
g1, g2 = st.columns(2)
with g1:
    st.markdown(f"##### Dyna-Q (n={settings.n_planning}): gelernte Policy")
    st.plotly_chart(build_grid(grid, a.Q.max(axis=1), a.policy), width="stretch", key="dq_final_grid")
with g2:
    st.markdown("##### Reines Q-Learning (n=0): gelernte Policy")
    st.plotly_chart(build_grid(grid, baseline.Q.max(axis=1), baseline.policy), width="stretch", key="dq_baseline_grid")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hilft Planung - und mit abnehmendem Grenzertrag?")
st.caption(f"Standardraster (ohne Rutschen), α={de(C.DEFAULT_ALPHA,2)}, {C.EXP_SEEDS} Seeds je Kombination. Gezeigt: Wert-Abstand und Umgebungsschritte über die Trainingsdauer, für n={', '.join(str(n) for n in C.EXP_N_LEVELS)}. Dauer bis zu einer Minute.")
if st.button("Planungsstufen durchrechnen", key="sample_start"):
    st.session_state["sample_on"] = True
if st.session_state.get("sample_on"):
    se = _sample_efficiency_exp(Settings(slip=0.0))
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_sample_efficiency(se), width="stretch", key="sample_chart")
    c2.plotly_chart(build_env_steps(se), width="stretch", key="sample_steps_chart")
    r0, r_best = se["rows"][0], max(se["rows"][1:], key=lambda r: -r[se["checkpoints"][0]]["mean"])
    ep0 = se["checkpoints"][0]
    st.warning(
        f"**Befund:** Bei nur {ep0} Episoden erreicht reines Q-Learning (n=0) einen Wert-Abstand von {de(r0[ep0]['mean'],2)} - mit Planung (n={r_best['n_planning']}) sind es {de(r_best[ep0]['mean'],2)}, bei GLEICHER echter Erfahrung. "
        "Der Effekt wird mit wachsendem n kleiner (abnehmender Grenzertrag): doppelt so viel Planung bringt nicht doppelt so viel."
    )

st.markdown("---")

st.subheader("🔬 Was passiert, wenn sich die Umgebung mitten im Training verschlechtert?")
st.caption(f"Blocking-Maze-Experiment (Sutton & Barto, Kapitel 8.3): {C.BLOCK_EPISODES_BEFORE} Episoden auf dem Standardraster (ohne Rutschen), dann werden drei Zellen der Reihe über der Klippe zusätzlich zur Gefahr - für weitere {C.BLOCK_EPISODES_AFTER} Episoden, OHNE die gelernte Q-Tabelle/das Modell zurückzusetzen. {C.EXP_SEEDS} Seeds. Dauer bis zu einer Minute.")
if st.button("Umgebungswechsel durchrechnen", key="block_start"):
    st.session_state["block_on"] = True
if st.session_state.get("block_on"):
    be = _blocking_exp(Settings(slip=0.0, epsilon_start=0.1, epsilon_decay=0.0))
    st.plotly_chart(build_blocking_curves(be), width="stretch", key="block_curve_chart")
    st.plotly_chart(build_blocking_bars(be), width="stretch", key="block_bar_chart")
    rows_by_n = {r["n_planning"]: r for r in be["rows"]}
    r0, r10 = rows_by_n[0], rows_by_n.get(10, be["rows"][-1])
    st.warning(
        f"**Befund:** Direkt nach dem Wechsel fällt der Ertrag bei allen n spürbar (n=0: {de(r0['before_last'],1)} → {de(r0['after_first'],1)}; n={list(rows_by_n)[-1]}: {de(be['rows'][-1]['before_last'],1)} → {de(be['rows'][-1]['after_first'],1)}) - "
        "das gelernte Modell empfiehlt kurzzeitig weiter die jetzt gefährliche alte Route. Das Modell korrigiert sich aber, sobald der reale Sturz einmal beobachtet wurde (unser Modell speichert nur den ZULETZT beobachteten Ausgang) - danach erholt sich die Policy wieder."
    )

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Umgebung ist (nahezu) deterministisch** | Unser Modell merkt sich nur den ZULETZT beobachteten Ausgang je Zustand-Aktion-Paar - bei Rutschen > 0 ist das eine verzerrte Stichprobe. Planung mit diesem verzerrten Modell kann dann MEHR schaden als nützen (gemessen: bei Rutschen 0,10 schneidet Dyna-Q mit Planung schlechter ab als reines Q-Learning, Preset "Mit Rutschen"). | Ein Modell, das ganze Verteilungen statt einzelner Ausgänge lernt |
| **Das Modell wird schnell genug neu erlebt** | Ändert sich die Umgebung (Blocking-Maze-Experiment), führt das veraltete Modell die Planung kurzzeitig in die Irre, bis der reale Absturz das Modell korrigiert. | Dyna-Q+ (Exploration-Bonus für lange nicht besuchte Zustand-Aktion-Paare) |
| **Endlich viele States und Actions (Tabelle)** | Ein sehr großes oder stetiges Raster macht sowohl die Q-Tabelle als auch das Modell unhandlich. | Funktionsapproximation / DQN (Stück 6) |
"""
)
st.caption("Die Linie: Bandit → Value Iteration und Policy Iteration → Q-Learning → SARSA → **Dyna-Q** (dieses Stück) → Funktionsapproximation (DQN) / Policy Gradient → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Modell** ist identisch zu `q-learning-demo`. Zusätzlich lernt der Agent ein eigenes Modell $\widehat{P}, \widehat{R}$: fuer jedes real besuchte $(s,a)$ wird der zuletzt beobachtete Ausgang $(r, s')$ gespeichert.

**Echter Schritt** (wie Q-Learning): $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big)$, danach $\text{Modell}(s,a) \leftarrow (r, s')$.

**Planungsschritt** ($n$-mal wiederholt): ein zufaelliges, schon real besuchtes $(\bar s, \bar a)$ auswaehlen, $(\bar r, \bar s') \leftarrow \text{Modell}(\bar s, \bar a)$, dasselbe Update: $Q(\bar s, \bar a) \leftarrow Q(\bar s, \bar a) + \alpha\big(\bar r + \gamma \max_{a'} Q(\bar s',a') - Q(\bar s, \bar a)\big)$.

**Reduktion:** bei $n=0$ wird kein Planungsschritt ausgefuehrt - das Modell wird geschrieben, aber nie gelesen. Dyna-Q ist dann byte-gleich zu Q-Learning (struktureller Regressionstest).

Implementiert in `dq_grid.py` (Vehikel, inkl. `blocked` fuer das Blocking-Maze-Experiment), `dq_agent.py` (Dyna-Q: echter Schritt + Planung), `dq_reference.py` (Value Iteration, nur zur Gegenprobe), `dq_evaluation.py` (Analyse, zwei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
