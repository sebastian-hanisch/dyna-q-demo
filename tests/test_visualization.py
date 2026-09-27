"""Regressionstest fuer den in `value-iteration-demo`/`q-learning-demo` gefundenen Plotly-Bug (deutsch formatierte Dezimal-Strings werden als
US-tausendergruppierte Ganzzahl geparst, wenn die x-Achse nicht explizit kategorial ist). Die Balken-Charts hier nutzen ganzzahlige n-Werte als
Kategorie-Labels - unkritisch fuer den Bug, aber `type="category"` schuetzt trotzdem vor einer numerischen Fehlinterpretation der Reihenfolge."""

import dq_evaluation as E
from dq_visualization import build_blocking_bars


def test_blocking_bars_x_axis_is_categorical_and_ordered():
    exp = E.blocking_experiment(n_levels=(0, 10, 50), seeds=range(3), episodes_before=10, episodes_after=10)
    fig = build_blocking_bars(exp)
    assert fig.layout.xaxis.type == "category"
    assert list(fig.layout.xaxis.categoryarray) == ["0", "10", "50"]
