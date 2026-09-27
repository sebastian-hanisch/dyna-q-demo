"""Plotly-Abbildungen der Demo "Dyna-Q". Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import dq_constants as C
import dq_grid as G

CLIFF_COLOR = "#3a3a3a"
BLOCKED_COLOR = "#8B0000"
START_COLOR = "#8c6bb1"
TEXT_LIGHT = "#ffffff"
TEXT_DARK = "#14233B"
N_COLORS = {0: "#7f7f7f", 5: "#1f77b4", 10: "#2e7d32", 25: "#d62728", 50: "#8c6bb1"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_grid(grid, V, policy=None):
    R, Cc = grid.rows, grid.cols
    z = np.full((R, Cc), np.nan)
    text = [["" for _ in range(Cc)] for _ in range(R)]
    for r in range(R):
        for c in range(Cc):
            s = grid.state_of((r, c))
            kind = grid.cell_kind((r, c))
            z[r, c] = np.nan if kind in ("cliff", "blocked") else V[s]
            if kind == "goal":
                text[r][c] = "Ziel"
            elif kind in ("cliff", "blocked"):
                text[r][c] = ""
            elif policy is not None:
                text[r][c] = G.ACTION_ARROWS[policy[s]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, colorscale="RdYlGn", zmid=0, showscale=True, text=[[de(v, 1) if not np.isnan(v) else "" for v in row] for row in z],
                              hovertemplate="Zeile %{y}, Spalte %{x}: Q=%{z:.2f}<extra></extra>", colorbar=dict(title="max Q(s,·)", thickness=14)))
    cliff_x = [c for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    cliff_y = [r for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    if cliff_x:
        fig.add_trace(go.Scatter(x=cliff_x, y=cliff_y, mode="markers", marker=dict(symbol="square", size=34, color=CLIFF_COLOR), showlegend=False, hovertemplate="Klippe<extra></extra>"))
    blocked_x = [c for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "blocked"]
    blocked_y = [r for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "blocked"]
    if blocked_x:
        fig.add_trace(go.Scatter(x=blocked_x, y=blocked_y, mode="markers", marker=dict(symbol="square", size=34, color=BLOCKED_COLOR), showlegend=False, hovertemplate="Neu blockiert<extra></extra>"))
    for r in range(R):
        for c in range(Cc):
            kind = grid.cell_kind((r, c))
            if text[r][c]:
                color = TEXT_LIGHT if kind == "goal" else TEXT_DARK
                fig.add_annotation(x=c, y=r, text=text[r][c], showarrow=False, font=dict(size=18, color=color))
    sr, sc = grid.start
    fig.add_shape(type="rect", x0=sc - 0.45, x1=sc + 0.45, y0=sr - 0.45, y1=sr + 0.45, line=dict(color=START_COLOR, width=3))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 90 * grid.rows + 60)


def build_sample_efficiency(exp):
    fig = go.Figure()
    for row in exp["rows"]:
        n = row["n_planning"]
        x = list(exp["checkpoints"])
        y = [row[ep]["mean"] for ep in x]
        se = [row[ep]["se"] for ep in x]
        fig.add_trace(go.Scatter(x=x, y=y, error_y=dict(type="data", array=se), mode="lines+markers", line=dict(color=N_COLORS.get(n, "#333"), width=2), marker=dict(size=6), name=f"n={n}"))
    fig.update_xaxes(title_text="Trainingsepisoden", type="log")
    fig.update_yaxes(title_text="Wert-Abstand zu V*(Start)", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_env_steps(exp):
    fig = go.Figure()
    for row in exp["rows"]:
        n = row["n_planning"]
        x = list(exp["checkpoints"])
        y = [row[ep]["env_steps_mean"] for ep in x]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=N_COLORS.get(n, "#333"), width=2), marker=dict(size=6), name=f"n={n}"))
    fig.update_xaxes(title_text="Trainingsepisoden", type="log")
    fig.update_yaxes(title_text="Umgebungsschritte (echte Erfahrung)", type="log")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_blocking_curves(exp, window=10):
    fig = go.Figure()
    for row in exp["rows"]:
        n = row["n_planning"]
        y = np.asarray(row["mean_curve"], dtype=float)
        if len(y) >= window:
            smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            x = np.arange(window, len(y) + 1)
        else:
            smooth, x = y, np.arange(1, len(y) + 1)
        fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", line=dict(color=N_COLORS.get(n, "#333"), width=2), name=f"n={n}"))
    fig.add_vline(x=exp["episodes_before"], line=dict(color="#8B0000", width=2, dash="dash"), annotation_text="Umgebung verschlechtert sich", annotation_position="top")
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text=f"Ertrag (gleitender Durchschnitt, {window})")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_blocking_bars(exp):
    levels = [r["n_planning"] for r in exp["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(n) for n in levels], y=[r["before_last"] for r in exp["rows"]], name="Vor dem Wechsel (letzte 10)", marker=dict(color="#7f7f7f")))
    fig.add_trace(go.Bar(x=[str(n) for n in levels], y=[r["after_first"] for r in exp["rows"]], name="Direkt nach dem Wechsel (erste 10)", marker=dict(color="#d62728")))
    fig.add_trace(go.Bar(x=[str(n) for n in levels], y=[r["after_last"] for r in exp["rows"]], name="Nach der Erholung (letzte 10)", marker=dict(color="#2e7d32")))
    fig.update_layout(barmode="group", yaxis=dict(title="Ertrag"))
    fig.update_xaxes(title_text="Planungsschritte n", type="category", categoryorder="array", categoryarray=[str(n) for n in levels])
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))
