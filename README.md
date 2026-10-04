# 🗺️ Dyna-Q

**[→ Demo live ausprobieren](https://sebastianhanisch-dyna-q-demo.streamlit.app/)**

Fünftes Stück der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Derselbe Lagerroboter wie bei [Q-Learning](https://github.com/sebastian-hanisch/q-learning-demo) (Stück 3). **Dyna-Q** (Sutton 1990) lernt zusätzlich zur Q-Tabelle ein eigenes **Modell** der Umgebung – und "plant" damit zwischen den echten Schritten: zusätzliche Q-Updates aus zufällig ausgewählten, schon einmal real erlebten Situationen, ganz ohne dafür neue echte Erfahrung zu brauchen.

## Kernfrage

**Wie viel weniger echte Erfahrung braucht Dyna-Q gegenüber reinem Q-Learning – und was, wenn sich die Umgebung mitten im Training verschlechtert?** Bei $n=0$ Planungsschritten wird das gelernte Modell nie gelesen: Dyna-Q ist dann exakt Q-Learning.

## Modell

- **Vehikel** (`dq_grid.py`): dasselbe Raster wie `q-learning-demo`, zusätzlich ein Feld `blocked` – optionale weitere Gefahrenzellen für das Blocking-Maze-Experiment.
- **Echter Schritt** (wie Q-Learning): $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big)$, danach wird der beobachtete Ausgang im Modell gespeichert: $\widehat{P}, \widehat{R}$ merkt sich für jedes real besuchte $(s,a)$ nur den **zuletzt** beobachteten $(r, s')$.
- **Planungsschritt** ($n$-mal je echtem Schritt): ein zufälliges, schon real besuchtes $(\bar s, \bar a)$ auswählen, denselben TD-Update mit dem gespeicherten Ausgang rechnen – als hätte man den Schritt gerade noch einmal erlebt.
- **Reduktion:** bei $n=0$ wird kein Planungsschritt ausgeführt – Dyna-Q ist dann byte-gleich zu Q-Learning (struktureller Test).

**Abweichung vom Rest der Linie:** `DEFAULT_SLIP=0` (wie schon bei `sarsa-demo`) – gemessen, dass schon leichtes Rutschen (0,02) die Stichprobeneffizienz von Dyna-Q **umkehrt**: das Modell merkt sich nur den zuletzt beobachteten Ausgang, unter Rutschen eine verzerrte Stichprobe (siehe Grenzen-Tabelle).

## Methodik

Zwei Experimente: **Stichprobeneffizienz** (Wert-Abstand der gelernten Policy zu $V^*$ über die Trainingsdauer, für $n \in \{0,5,10,25\}$) und das **Blocking-Maze-Experiment** (Sutton & Barto 2018, Kapitel 8.3): nach einer festen Zahl Episoden werden drei bisher sichere Zellen zusätzlich zur Gefahr, ohne die gelernte Q-Tabelle/das Modell zurückzusetzen – gemessen wird der reale Trainingsertrag (konstantes kleines Epsilon, damit er nicht von Explorationsrauschen dominiert wird).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (12 Episoden, ohne Rutschen, Seed 0) | n=10: Wert-Abstand 0,00 nach 3.617 Umgebungsschritten. n=0 (reines Q-Learning): Wert-Abstand 19,90 nach 4.308 Umgebungsschritten – Dyna-Q lernt hier **mit weniger echter Erfahrung eine bessere Policy**. | `test_standard_case` |
| **Reduktion bei n=0** | Dyna-Q mit n=0 erzeugt bei gleichem Seed eine byte-gleiche Q-Tabelle wie eine eigenständige Q-Learning-Nachrechnung – das Modell wird geschrieben, aber nie gelesen. | `test_zero_planning_steps_reduce_dyna_q_exactly_to_plain_q_learning` |
| **Wie schnell zeigt sich der Stichproben-Vorteil?** (2 bis 20 Episoden, 15 Seeds) | Bei 8 Episoden: n=0 noch Wert-Abstand 17,25 – mit Planung (n≥5) schon 0,00 bis 1,46. Bei 20 Episoden haben auch n=0 fast konvergiert (1,33) – der Vorteil gilt vor allem in der frühen Trainingsphase, mit abnehmendem Grenzertrag oberhalb n≈5. | `test_sample_efficiency_experiment` |
| **Was passiert bei einer Umgebungsverschlechterung?** (Blocking-Maze, 30+30 Episoden, konstant ε=0,10, 15 Seeds) | Der Ertrag fällt bei ALLEN n spürbar, sobald drei bisher sichere Zellen zur Gefahr werden (z. B. n=0: −46,0 → −94,4; n=25: −22,9 → −59,7) – das veraltete Modell empfiehlt kurzzeitig die jetzt gefährliche Route weiter. Die Erholung ist danach messbar, aber unterschiedlich vollständig (nicht immer bis zum Vorwechsel-Niveau). | `test_blocking_experiment` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Umgebung ist (nahezu) deterministisch** | Unser Modell merkt sich nur den ZULETZT beobachteten Ausgang je Zustand-Aktion-Paar – bei Rutschen > 0 ist das eine verzerrte Stichprobe. Planung mit diesem verzerrten Modell kann dann MEHR schaden als nützen (gemessen: schon bei Rutschen 0,02 kehrt sich der Stichproben-Vorteil um). | Ein Modell, das ganze Verteilungen statt einzelner Ausgänge lernt |
| **Das Modell wird schnell genug neu erlebt** | Ändert sich die Umgebung (Blocking-Maze-Experiment), führt das veraltete Modell die Planung kurzzeitig in die Irre, bis der reale Absturz das Modell korrigiert. | Dyna-Q+ (Exploration-Bonus für lange nicht besuchte Zustand-Aktion-Paare) |
| **Endlich viele States und Actions (Tabelle)** | Ein sehr großes oder stetiges Raster macht sowohl die Q-Tabelle als auch das Modell unhandlich. | Funktionsapproximation / DQN (Stück 6) |

## Tests

`tests/` prüft das Vehikel (`dq_grid.py`: dasselbe Übergangsmodell wie `q-learning-demo`, zusätzlich `blocked` für das Blocking-Maze-Experiment), die Referenzlösung (`dq_reference.py`: Bellman-Formel von Hand, entarteter Fall mit geschlossener Lösung), Dyna-Q (`dq_agent.py`: TD-Update von Hand, eine unabhängige Schritt-für-Schritt-Nachrechnung mit Planung, ein struktureller Kürzungstest [Snapshot = kürzerer Lauf], `train_switching`s Vor-Wechsel-Phase gegen einen eigenständigen `train()`-Lauf UND die zentrale Korrektheits-Kette: bei n=0 ist Dyna-Q byte-gleich zu Q-Learning), die Auswertung und beide Experimente, die Presets und Permalinks, die Plotly-Achsen und jede Zahl dieses READMEs (`test_claims.py`). Dyna-Q ist stochastisch – Einzelläufe sind exakt (fester Seed), Mehr-Seed-Aussagen tragen großzügige Bänder. 55 Tests, Laufzeit knapp eine Minute (`test_claims.py` misst mit vollen Seed-Zahlen); die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Episode-für-Episode-Ansicht, Kernfrage, zwei Experimente auf Abruf, Grenzen, Formeln |
| `dq_constants.py` | Regler-Grenzen, feste Rewards, Lernparameter, Experiment- und Blocking-Maze-Konstanten |
| `dq_grid.py` | Das Vehikel: Raster, `step` (Einzelübergang), `build_model` (nur für die Referenz), `blocked` |
| `dq_agent.py` | Dyna-Q: echter Schritt + n Planungsschritte, `train_switching` fürs Blocking-Maze-Experiment |
| `dq_reference.py` | Value Iteration und Policy Evaluation – nur zur Gegenprobe |
| `dq_evaluation.py` | Analyse, zwei Experimente |
| `dq_visualization.py` | Plotly-Abbildungen |
| `dq_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Ein Modell, das Verteilungen statt einzelner Ausgänge lernt** – würde den Nachteil unter Rutschen beheben, ist aber nicht mehr das klassische tabellarische Dyna-Q (Sutton 1990).
- **Dyna-Q+** (Exploration-Bonus für lange nicht besuchte Zustand-Aktion-Paare) – würde das Blocking-Maze-Experiment robuster machen, ist aber ein eigenes Verfahren.
- **Funktionsapproximation** – die Q-Tabelle und das Modell bleiben hier dicht und klein genug, um sie vollständig zu speichern (Stück 6).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Reinforcement Learning: Bandit bis Actor-Critic](https://sebastianhanisch.net/konzepte-reinforcement-learning.html).
