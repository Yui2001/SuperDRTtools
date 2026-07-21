# SuperDRTtools architecture

SuperDRTtools separates application startup, GUI workflows, reusable UI code,
state services, persistence, parallel execution, and numerical algorithms. The
refactor preserves the existing public entry points and user-visible behavior.

## Package map

```text
pyDRTtools/
|-- app/
|   |-- bootstrap.py          Qt application startup
|   |-- cli.py                command-line entry point
|   |-- main_window.py        main-window composition root
|   |-- window_manager.py     independent window creation and lifetime
|   `-- window_setup.py       initial state and signal wiring
|-- algorithms/
|   |-- runs.py               DRT workflow entry points and EIS object
|   |-- basics.py             numerical matrix and optimization utilities
|   |-- BHT.py, HMC.py        BHT and sampling algorithms
|   |-- fGP.py                Gaussian-process calculations
|   |-- parameter_selection.py
|   |-- peak_analysis.py
|   `-- nearest_PD.py
|-- controllers/
|   |-- importing.py          file import and drag/drop
|   |-- masking.py            manual/automatic mask workflows
|   |-- file_list.py          selection, rename, reorder, and removal
|   |-- kramers_kronig.py     Lin-KK GUI workflow
|   |-- fitting_workflow.py   Fit One/Fit All coordination
|   |-- peak_analysis.py      Peak One/All, naming, results, and comparison
|   |-- plotting.py           shared plot switching and canvas hosting
|   |-- drt_comparison.py     comparison interaction and settings
|   |-- drt_map.py            map interaction and settings
|   |-- drt_export.py         DRT CSV export
|   |-- eis_export.py         configurable EIS CSV export
|   |-- figure_export.py      figure saving
|   `-- project.py            project state and save/open workflow
|-- services/
|   |-- eis_state.py          raw/visible arrays, mask state, and Lin-KK state
|   |-- peak_results.py       peak-count estimation and serializable metrics
|   `-- project_storage.py    safe ZIP/JSON/NumPy project storage
|-- infrastructure/
|   |-- parallel_fitting.py   adaptive multiprocessing Fit All backend
|   `-- peak_fitting.py       process-backed Peak One/Peak All execution
|-- ui/
|   |-- layout.py             generated Qt widget tree
|   |-- runtime_layout.py     responsive runtime layout adaptation
|   |-- canvas.py             Matplotlib canvas and plot primitives
|   |-- widgets.py            reusable Qt delegates and event filters
|   `-- theme.py              global Qt theme
`-- __init__.py               lazy package-level algorithm aliases
```

Application code imports the organized modules directly:

```python
from pyDRTtools.app import SuperDRTMainWindow
from pyDRTtools.app.bootstrap import launch_gui
from pyDRTtools.ui.canvas import Figure_Canvas
from pyDRTtools.ui.layout import Ui_MainWindow
```

## Dependency direction

```text
launch.py
  -> app/bootstrap.py
      -> app/main_window.py
          -> controllers
              -> services / ui / infrastructure
                  -> algorithms
```

- `app` composes the application and does not implement analysis behavior.
- `controllers` own one GUI feature each and communicate through the main-window
  state for backward compatibility.
- `services` provide non-widget state helpers shared by multiple controllers.
- `ui` contains presentation code only.
- `infrastructure/parallel_fitting.py` owns process management and delegates calculations to the
  established numerical workflow functions. Its worker count is bounded by physical CPU capacity
  and an adaptive commit-memory reserve, then increased one process at a time when CPU and memory
  headroom permit it.
- `infrastructure/peak_fitting.py` reuses that scheduler for non-blocking peak
  analysis while `services/peak_results.py` derives user-facing metrics from
  the unchanged fitted peak components.
- Numerical modules do not depend on GUI controllers.

## Numerical algorithm boundary

The GUI refactor does not change equations, solver calls, regularization,
Bayesian sampling, BHT calculations, peak analysis, or fitting parameters. The
following modules form the protected numerical layer:

- `algorithms/runs.py`
- `algorithms/basics.py`
- `algorithms/BHT.py`
- `algorithms/HMC.py`
- `algorithms/fGP.py`
- `algorithms/nearest_PD.py`
- `algorithms/parameter_selection.py`
- `algorithms/peak_analysis.py`

Changes to these modules should be reviewed as algorithm changes rather than UI
or architecture changes.

## Adding a GUI feature

1. Add one controller module under `controllers/`.
2. Keep Qt widgets and rendering primitives under `ui/`.
3. Put shared non-widget state transformations under `services/`.
4. Add the controller mixin to `SuperDRTMainWindow` in `app/main_window.py`.
5. Connect its signals in `app/window_setup.py`.
6. Preserve compatibility facades when moving an existing public symbol.
7. Run syntax, import, and offscreen main-window smoke tests before committing.

## Comments and naming

- Module docstrings describe ownership and responsibility.
- Controller names end in `ControllerMixin` or `WorkflowMixin`.
- Private event handlers use the existing leading-underscore convention.
- Comments explain intent, state invariants, and compatibility constraints; they
  should not restate obvious Python operations.
