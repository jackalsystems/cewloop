# cewloop — Design Document

A JAX research library for building cognitive-EW control loops out of interchangeable pieces.
Working name `cewloop`; rename freely.

---

## 1. Purpose and non-goals

**Purpose.** Let one researcher assemble a sense → (estimate) → decide → act loop against an adaptive
emitter from library pieces, run it at scale (`vmap` over seeds and scenario parameters), sweep sensing
constraints as dials, and produce the standard plots — without rewriting the scan/PRNG/logging plumbing
every time the research focus moves.

**Non-goals.**
- Not production software. No plugin system, no CLI, no config validation beyond what prevents silent bugs.
- Not a simulator of any specific radar or radio. Emitters and channels are deliberately simple and
  parameterized; fidelity is added by swapping pieces, not by growing a monolith.
- Not a framework that owns your loop. The library provides stages and a runner; the user writes the
  wiring for one step in plain Python so every call is visible.

---

## 2. Design principles

1. **One stage, one job.** Every library call does exactly one thing. A delay wrapper only delays; a
   channelizer only channelizes; an HMM filter only updates a belief. Nothing is fused "for convenience."
2. **Pure functions, explicit state.** Every stage is `init(key) -> state` and
   `step(state, inputs, key) -> (state, outputs)`. No hidden mutable state anywhere. This is what makes
   `jit`, `vmap`, and `scan` work without special cases.
3. **The user writes the wiring.** The library never decides what feeds what. The canonical loop is
   provided as *an example wiring*, not as a baked-in structure. If the focus is control, wire an
   estimator in; if the focus is sensing, wire raw observations straight into the controller.
4. **Ground truth is free.** The world state lives in the scan carry and is logged by default. The
   "referee" is just the log.
5. **Dials are wrappers.** Sensing constraints (noise, sparsity, latency, compute, nonstationarity) are
   functions that take a stage and return a stage. They compose, and each is off until applied.
6. **Same Blue code on hardware.** The I/O boundary is a single stage (the channel). Replacing it with a
   hardware transport changes the runner backend from `scan` to a Python loop and nothing else.
7. **Heavy comments.** Every module explains its role in the loop, its state fields with shapes, what it
   consumes, what it emits, and what it deliberately does *not* do.

---

## 3. Core abstractions

### 3.1 `Stage`

```python
# cewloop/core/stage.py

class Stage(NamedTuple):
    """A pure, stateful computation step. The only abstraction in the library.

    init : (key: PRNGKey) -> State
        Builds the initial state. State is any pytree of jnp arrays (NamedTuple preferred).
        Static configuration is closed over at construction time, never stored in State.

    step : (state: State, inputs: Inputs, key: PRNGKey) -> (State, Outputs)
        One step of work. Inputs/Outputs are pytrees. Must be jit-traceable:
        no Python control flow on traced values, fixed shapes, explicit PRNG.

    Conventions
    -----------
    - Every step receives a fresh PRNGKey even if it does not use randomness.
      The runner splits keys; stages never carry keys in state.
    - A stage never reads another stage's state. Cross-stage data flows only
      through inputs/outputs in the user's wiring function.
    """
    init: Callable[[PRNGKey], Any]
    step: Callable[[Any, Any, PRNGKey], tuple[Any, Any]]
```

A stage is constructed by a factory that takes static config and returns a `Stage`:

```python
def fh_emitter(n_channels: int, dwell: int, react_threshold_db: float, ...) -> Stage:
    ...
    return Stage(init=init, step=step)
```

Static config lives in the closure. This keeps State a flat pytree of arrays and makes `vmap` over
scenario parameters explicit: to sweep a parameter, make it a traced input instead of a closure value
(see §7).

### 3.2 `Loop`

```python
# cewloop/core/loop.py

class Loop(NamedTuple):
    """A named collection of stages plus a user-written wiring function.

    stages : dict[str, Stage]
        Named stages. Names are chosen by the user and appear in the carry and the log.

    wire : (stages, states, key) -> (states, log)
        One step. The user calls stage.step(...) explicitly, in whatever order,
        with whatever data flow. Returns the updated states dict and a log pytree.
        Anything in `log` is stacked over time by the runner.

    placement : dict[str, Placement]
        Where each named stage runs. Defaults to Placement.SIM for every stage.
        See §3.5. The wiring function is written once; placement decides whether
        a given stage.step is a pure JAX function or a call out to a device.

    The library does NOT inspect `wire`. It only packs/unpacks the states dict
    and splits keys. This is the transparency guarantee: every stage call is in
    user code and visible in the notebook.
    """
    stages: dict[str, Stage]
    wire: Callable
    placement: dict[str, Placement] = {}
```

Helpers (each one thing):

- `loop.init(key) -> dict[str, State]` — calls every stage's `init` with a split key.
- `split_for(stages, key) -> dict[str, PRNGKey]` — one key per stage, by name.
- `loop.resolve() -> Loop` — applies placement: wraps every non-SIM stage in `remote_stage` (§3.5)
  and returns a Loop whose stages all present the plain `Stage` interface again.

### 3.3 Runner

```python
# cewloop/core/runner.py

def run_once(loop: Loop, key: PRNGKey, n_steps: int, *, init_states=None, backend=None) -> tuple[states, Log]:
    """One run: n_steps consecutive steps of the loop under one key.

    init_states      : starting states. None -> loop.init(key), a fresh start. Pass the
                       final states of a previous run to continue it without reset
                       (e.g. a long nonstationary study, or Blue carrying what it learned).
                       What gets reset between runs is therefore the caller's choice,
                       not the runner's.
    backend=None     : chosen from placement. All stages SIM -> "scan"; any stage on a
                       device -> "python". Explicit backend overrides (e.g. "python" on an
                       all-SIM loop for pdb).
    backend="scan"   : lax.scan over steps. jit-compiled. Log is a pytree of [T, ...] arrays.
                       Refuses to run if any stage is placed on a device.
    backend="python" : plain for-loop. Same return structure, built by stacking with np.stack.
                       Also records per-stage wall time into log["_timing"] for latency_budget.
    """

def run_batch(loop: Loop, key: PRNGKey, n_steps: int, n_seeds: int) -> Log:
    """vmap run_once over seeds. Log becomes [S, T, ...]. SIM-only; raises otherwise."""
```

That is the entire runner. Scenario sweeps are a separate utility (§7) built on `run_batch`.

### 3.5 `Placement` — where each role runs

```python
# cewloop/core/placement.py

class Placement(NamedTuple):
    """Where a named stage executes. Attached per stage in Loop.placement.

    host   : "sim" | hostname or IP of a device running a cewloop worker
             ("pi-red.local", "jetson.local", "localhost" for a laptop-attached Pluto).
    device : what the stage drives on that host, if anything.
             None       -> pure compute on that host (e.g. a controller on the Jetson).
             "pluto:0"  -> the stage owns that host's Pluto (tx, rx, or both).
             "sa"       -> spectrum analyzer (observer only).
    role   : optional tag ("red", "blue", "observer") used only for grouping in logs and plots.
    """
    host: str = "sim"
    device: str | None = None
    role: str | None = None

SIM = Placement()
```

**What placement does, and only this:** `loop.resolve()` replaces each stage whose `host != "sim"`
with `remote_stage(stage, placement)`. A remote stage has the same `Stage` interface. Its `init`
ships the stage's factory config to the worker on that host and gets back a handle; its `step` sends
`(inputs, key)` over ZeroMQ, the worker calls the real `step` locally, and the outputs come back.
State stays on the device; the local state slot holds only the handle and a step counter, so the
log shape is unchanged. Nothing is fused, reordered, or hidden: `wire` still calls `st.red.step(...)`
and that line is the network round-trip.

**The channel stage under placement.** The channel is the one stage whose meaning changes with where
its neighbors run, so its factory takes the neighbors' placements:

| Red | Blue | Channel stage to use |
|---|---|---|
| sim | sim | `abstract_channel` or `iq_channel` — pure JAX |
| device | device | `rf_channel` — the cable/air. `step` is a timing barrier: it waits for both devices to report step `t` done, returns `truth` from Red's worker and nothing else. The RF path *is* the data flow. |
| sim | device | `sim_to_rf_bridge` — Red's sim emission is synthesized as IQ and transmitted from a Pluto on the sim host; Blue receives on its own Pluto. Lets a simulated Red (with full ground truth and `vmap`-tuned behavior) drive real Blue hardware. |
| device | sim | `rf_to_sim_bridge` — a Pluto on the sim host records Blue's `blue_rx`; Blue's sim action is transmitted back out. Lets a real emitter be attacked by a simulated Blue. |

Each is one stage with the standard `(red_emission, blue_action) -> (red_rx, blue_rx, truth)`
output structure, so sensing/estimation/control stages never know which one they are wired to.

**Mixed Blue.** Blue is several stages (sensor, estimator, controller, actuator), and each has its own
placement. Sensor on the laptop's Pluto, estimator and controller on the Jetson, actuator on the
laptop's Pluto is a valid loop; the wiring is unchanged and each `.step` line in `wire` is either a
local JAX call or a round-trip according to its placement. This is how a part gets moved to hardware
one stage at a time.

**Observers.** A stage with `device="sa"` or a logging-only Pluto is placed like any other and wired
to receive `truth` or `blue_rx` and emit nothing the loop consumes. It appears in the log under its
role tag.

**Worker.** `cewloop/worker.py` is a ~100-line process started on each device:
`python -m cewloop.worker --bind tcp://*:5555`. It imports the same library, receives factory names
and configs, instantiates stages locally, and answers `init`/`step` requests. Pluto-owning stages
(`channelizer`, `jammer_iq`, `pluto_rx`, `pluto_tx`) are the only ones that touch `pyadi-iio`, and
only when their placement names a Pluto.

**Timing.** Under any device placement the runner's step clock is wall time. `run_once` takes
`step_seconds` and sleeps to the boundary; stages that overrun are logged, not blocked. Dials that
model hardware impairments (`with_delay`, `with_quantization`, `with_turnaround_gap`) should not be
applied to stages placed on devices — the device supplies them. `loop.resolve()` warns if it finds one.

### 3.4 Log

The log is whatever the user's `wire` returns. The library provides a `Log` NamedTuple *suggestion*
with the fields the built-in metrics expect (see §6), but nothing requires it. Metrics are functions of
arrays, not of a schema.

---

## 4. Library pieces

Each piece is a `Stage` factory unless noted. Grouped by role in the loop. Every group has a
`README.md` describing the shared input/output conventions for that role.

### 4.1 `emitters/` — Red

The adaptive emitter. State is a Markov chain plus whatever Red needs to estimate its own situation.

| Factory | What it does | State | Inputs | Outputs |
|---|---|---|---|---|
| `fh_emitter` | Frequency hopper with a fixed hop sequence | `(hop_idx, dwell_ctr)` | `red_rx` (Red's own received signal / SINR estimate) | `Emission(channel_mask[C], power[C])` |
| `reactive_fh_emitter` | Hops when its SINR estimate drops below threshold for `k` steps | `+ (sinr_est, below_ctr)` | same | same |
| `regime_switching_emitter` | Switches between emitter modes on a Markov chain (nonstationarity source) | `(mode, inner_state)` | same | same |
| `pulsed_emitter` | PRI/frequency-agile pulse train (radar flavor) | `(pri_idx, freq_idx, phase)` | same | `Emission` |
| `library_emitter` | Picks one of several emitter types at init — makes identification a task | `(type_idx, inner_state)` | same | same + `type_idx` in log |

Red's SINR estimate is computed from `red_rx`, which the channel provides. Red reacts to its estimate,
not to truth. `emitters/sinr_est.py` provides the estimators Red can use (one stage each).

### 4.2 `channel/` — the world

| Factory | Tier | What it does |
|---|---|---|
| `abstract_channel` | A | Path-loss matrix + noise → per-channel SINR for both sides. No samples. |
| `iq_channel` | B | Generates samples: Red waveform + Blue waveform → gain, AWGN → `blue_rx[N]`, `red_rx[N]`. |
| `fading_wrap` | B | Wraps `iq_channel` with a tapped-delay fading process carried in state. |

Inputs: `(red_emission, blue_action)`. Outputs: `(red_rx, blue_rx, truth)` where `truth` is the
per-channel true SINR / occupancy for logging. Tier A and Tier B share the same output structure so
sensing stages can be tested against both.

### 4.3 `sensing/` — Blue's observation

| Factory | What it does |
|---|---|
| `full_view` | Passes the channel's `blue_rx` through unchanged. Baseline. |
| `window` | Observes only `bandwidth` channels centered at `center` — `center` is an **input**, so sensing is an action. |
| `energy_detector` | Per-channel energy → occupancy estimate (Tier A: noisy threshold on SINR; Tier B: `|x|²` sums). |
| `channelizer` | Polyphase channelizer on IQ samples → per-bin power. This is the code that runs on the Pluto. |

Outputs are always an `Obs` pytree. The `Obs` type is user-defined per experiment; library sensors
emit `Obs(occupancy[C], power[C], window_center)` by default.

### 4.4 `dials/` — sensing constraints as wrappers

Each is `(stage, **params) -> Stage`. Each adds exactly one thing to the wrapped stage's state and
does exactly one transformation.

| Wrapper | Dial | Mechanism |
|---|---|---|
| `with_noise(stage, snr_db)` | noise | Adds AWGN to the stage's output. |
| `with_delay(stage, d)` | latency | Ring buffer of length `d` in state; emits output from `d` steps ago. Works on any stage (observation delay on a sensor, action delay on a controller). |
| `with_window(stage, bw)` | sparsity | Restricts output to `bw` channels around a center input. |
| `with_dwell_fraction(stage, f)` | sparsity | Stage runs only on a fraction `f` of steps; holds last output otherwise. |
| `with_quantization(stage, bits)` | hardware | Uniform quantizer on output. |
| `with_turnaround_gap(stage, gap)` | hardware | Blanks output for `gap` steps after a transmit action (the known obscuration). |
| `with_compute_budget(stage, ops)` | compute | Passes `ops` as a traced input the stage uses to limit rollout depth / belief grid / particle count. Stages that honor it declare so in their docstring. |
| `with_drift(stage, rate)` | nonstationarity | Random-walks a named parameter in the wrapped stage's state. For emitters and channels. |

Dials compose: `with_delay(with_quantization(channelizer(...), 12), 2)`. Because each wrapper
appends to the state pytree rather than replacing it, the inner stage's state is still inspectable in
the log.

### 4.5 `estimation/` — optional belief

Consumed by a controller or not, as the user chooses.

| Factory | What it does | State |
|---|---|---|
| `identity` | Passes `Obs` through. Use when the controller consumes raw observations. | none |
| `hmm_filter` | Discrete Bayesian filter over Red's hidden state given transition matrix `P` and observation kernel `B`. | `belief[K]` |
| `kalman_filter` | Linear-Gaussian filter for continuous Red parameters (e.g. center-frequency drift). | `(mean, cov)` |
| `particle_filter` | For nonlinear/non-Gaussian cases; honors `compute_budget` via particle count. | `(particles, weights)` |
| `belief_summary` | Reduces a belief to scalars (MAP, entropy, variance) for controllers that want a point estimate. | none |

`estimation/structure.py` is **not** a stage: it contains `check_mlr(P, B)` and `check_tp2(B)`, which
numerically test the ordering conditions that threshold-policy results depend on. Call them at
experiment-setup time and log the result.

### 4.6 `control/` — Blue's decision

All controllers consume a pytree (raw `Obs` or a belief) and emit an `Action`. The `Action` type is
user-defined; library default is `Action(jam_mask[C], power, sense_center)` so sensing-as-action is
available without a second controller.

| Factory | What it does | Notes |
|---|---|---|
| `ucb_bandit` | UCB over a discrete action set with a user-supplied reward proxy. | Consumes raw `Obs`. |
| `thompson_bandit` | Posterior-sampling counterpart. | |
| `certainty_equivalent` | Takes a belief, computes MAP, applies a user-supplied deterministic rule. | The separation-principle baseline. |
| `threshold_policy` | Threshold in belief space (Krishnamurthy–Djonin form). Threshold vector is a parameter or learned. | Pairs with `hmm_filter`. |
| `rollout` | Monte-Carlo rollout over a horizon using a model stage for Red; honors `compute_budget`. | Charlish–Hoffmann anticipation. |
| `info_gain` | Chooses sensing action maximizing expected entropy reduction of the belief. | Kreucher–Hero. |
| `task_driven` | Chooses action minimizing expected task loss under the belief. | Kreucher–Hero counterpart. |
| `voi_dwell` | Decides "one more dwell or act now" by comparing EVSI to delay cost. | Howard. |
| `fixed_rule` | Non-adaptive baseline for robust-vs-adaptive comparisons. | Thornton–Buehrer. |
| `oracle` | Reads Red's true state from the channel's `truth` output. Only for regret computation. | Never wire this into a real Blue. |

### 4.7 `actuation/` — Blue's transmit path

| Factory | What it does |
|---|---|
| `jammer_abstract` | Maps `Action` → per-channel interference power (Tier A). |
| `jammer_iq` | Synthesizes samples: tone, barrage, swept, follower. The code that runs on the Pluto Tx. |

---

## 5. Data flow conventions

- **Vocabulary.** Three words, used the same way everywhere in code, docs, and plots:
  - **step** — one decision cycle: one pass through `wire`, in which every stage's `.step` is called
    once. The smallest unit of time the loop knows about. `T` steps per run; log axis `[T]`.
  - **run** — `T` consecutive steps under one PRNG key and one starting state. `run_once` produces
    one; `run_batch` produces `S` of them with different keys, log axis `[S, T]`. A run is only a
    unit of independent randomness for averaging; it does not imply a reset means anything to Red
    or Blue (see `init_states`).
  - **sweep** — a grid of runs over scenario parameters, leading axes `[D1, D2, ..., S, T]`.
  Not used: *epoch* (collides with the ML and RF meanings) and *episode* (implies RL semantics the
  library does not have).
- **Step** is the unit of time. Sub-step structure (IQ samples) is internal to a stage's
  inputs/outputs, never to the loop.
- **Shapes are fixed.** `C` channels, `N` samples per step, `K` hidden states, `A` actions — all set
  at construction time. Delays use ring buffers, not growing lists.
- **PRNG.** The runner gives `wire` one key per step; `split_for` fans it out by stage name. Stages
  never store keys.
- **Truth flows from the channel.** Only the channel emits `truth`. Oracle and metrics consume it;
  nothing in Blue does. Enforce this by convention and by the wiring example, not by machinery.
- **Static vs traced.** Anything you might `vmap` over must be an input, not a closure value. The
  factory signature makes this explicit: `fh_emitter(n_channels=8, ...)` is static; the react
  threshold can instead be passed as `step(state, inputs=(red_rx, threshold), key)` when sweeping it.

---

## 6. Metrics and plots

### 6.1 `metrics/`

Pure functions of arrays. Each takes the minimal inputs and returns an array; none take a `Log`.

| Function | Inputs | Returns |
|---|---|---|
| `regret(reward_blue[T], reward_oracle[T])` | | cumulative regret `[T]` |
| `true_effect(truth[T,C], jam_mask[T,C])` | | per-step effect on Red (SINR loss at Red's active channel) `[T]` |
| `proxy_gap(reward_proxy[T], true_effect[T])` | | how far Blue's internal reward is from the referee's `[T]` |
| `time_to_adapt(effect[T], threshold)` | | steps until effect first exceeds threshold (both sides) |
| `tracking_error(estimate[T], truth[T])` | | `[T]` |
| `belief_entropy(belief[T,K])` | | `[T]` |
| `id_accuracy(map_type[T], true_type)` | | `[T]` |
| `rmb_sinr_loss(K, N)` | | Reed–Mallett–Brennan `(K+2−N)/(K+1)` — closed form, for overlaying on measured loss |
| `latency_budget(timestamps)` | | per-stage wall time for `backend="python"` runs |
| `ci(x[S,...], level)` | | mean and bootstrap CI over the seed axis |

### 6.2 `plots/`

Matplotlib; each returns the `Axes` it drew on so figures compose.

| Function | Shows |
|---|---|
| `curve_with_ci(x[S,T], ...)` | mean ± CI over seeds vs step. Regret, effect, tracking error. |
| `occupancy_raster(truth[T,C], jam_mask[T,C], window[T])` | Red's channel use, Blue's jamming, Blue's observation window on one time–frequency raster. The single most useful debugging plot. |
| `belief_raster(belief[T,K], true_state[T])` | belief heatmap with truth overlaid. |
| `dial_sweep(result[D1,D2], d1_vals, d2_vals)` | heatmap of any metric over two dials. |
| `policy_ranking(result[P,D], ...)` | which policy wins as a dial turns — the impairment-budget figure. |
| `threshold_plot(belief[T,K], action[T], threshold)` | belief trajectory on the simplex with the threshold surface. |
| `latency_hist(times)` | per-stage wall-time distribution. |

---

## 7. Sweeps

```python
# cewloop/analysis/sweep.py

def sweep(loop_factory, grid: dict[str, Array], key, n_steps, n_seeds) -> Log:
    """vmap run_batch over a parameter grid.

    loop_factory : (**params) -> Loop, where every param is a traced scalar.
    grid         : {"delay": jnp.arange(0, 5), "snr_db": jnp.linspace(0, 20, 5)}

    Returns a Log with leading axes [len(d1), len(d2), ..., S, T].
    """
```

This requires that swept parameters are traced inputs (§5). If a parameter is structural (e.g.
`n_channels`), the sweep is a Python loop over factories — `sweep_static` does that and stacks results
with `np.stack`.

---

## 8. Moving to hardware

Nothing about the loop changes; only `Loop.placement` does. The progression for a given experiment:

1. **All SIM.** `vmap` over seeds, sweep dials, pick policies. (`backend="scan"`.)
2. **Red on hardware, Blue in sim.** `placement["red"] = Placement("pi-red.local", "pluto:0", "red")`,
   channel becomes `rf_to_sim_bridge`. Validates the emitter model against a real transmitter.
3. **Blue sensor on hardware, rest in sim.** `placement["sense"] = Placement("localhost", "pluto:0", "blue")`.
   Validates the channelizer against the Tier B sim.
4. **Everything on hardware.** Channel becomes `rf_channel`; observer on the SA.

Steps 2–4 run with `backend="python"`, which is selected automatically from placement.

The acceptance test at each stage of the migration: the same `wire` function and the same stage factories, with log
fields identical in name and shape between the SIM run (with impairment dials on) and the placed run
(with them off). `test_placement_parity.py` runs this on a loopback worker on the laptop so the
machinery is tested before any Pluto is plugged in.

Device stages (`channel/pluto_rx.py`, `channel/pluto_tx.py`, the `rf_*` channels, the bridges) are the
only code that imports `pyadi-iio`, and they are stubs with the interface and a `NotImplementedError`
until hardware arrives.

---

## 9. Repository layout

```
cewloop/
├── README.md                     # what this is, the Stage protocol, one 20-line example
├── pyproject.toml                # deps: jax, numpy, matplotlib, pyyaml. Optional: chex (asserts only)
├── cewloop/
│   ├── core/
│   │   ├── stage.py              # Stage NamedTuple + docstring that is the spec
│   │   ├── loop.py               # Loop, split_for, loop.init, loop.resolve
│   │   ├── placement.py          # Placement, SIM, remote_stage
│   │   ├── runner.py             # run_once, run_batch
│   │   └── types.py              # default Obs, Action, Emission, Truth NamedTuples
│   ├── worker.py                 # device-side process: answers init/step over ZeroMQ
│   ├── emitters/
│   │   ├── README.md
│   │   ├── fh.py
│   │   ├── reactive.py
│   │   ├── regime.py
│   │   ├── pulsed.py
│   │   ├── library.py
│   │   └── sinr_est.py
│   ├── channel/
│   │   ├── README.md
│   │   ├── abstract.py
│   │   ├── iq.py
│   │   ├── fading.py
│   │   ├── rf.py                 # rf_channel: timing barrier + truth relay (stub until hardware)
│   │   ├── bridges.py            # sim_to_rf_bridge, rf_to_sim_bridge (stubs)
│   │   ├── pluto_rx.py           # stub
│   │   └── pluto_tx.py           # stub
│   ├── sensing/
│   │   ├── README.md
│   │   ├── window.py
│   │   ├── energy.py
│   │   └── channelizer.py
│   ├── dials/
│   │   ├── README.md
│   │   ├── delay.py
│   │   ├── noise.py
│   │   ├── sparsity.py           # with_window, with_dwell_fraction
│   │   ├── quantize.py
│   │   ├── turnaround.py
│   │   ├── compute.py
│   │   └── drift.py
│   ├── estimation/
│   │   ├── README.md
│   │   ├── identity.py
│   │   ├── hmm.py
│   │   ├── kalman.py
│   │   ├── particle.py
│   │   ├── summary.py
│   │   └── structure.py          # check_mlr, check_tp2 (not stages)
│   ├── control/
│   │   ├── README.md
│   │   ├── bandit.py             # ucb, thompson
│   │   ├── certainty_equiv.py
│   │   ├── threshold.py
│   │   ├── rollout.py
│   │   ├── infogain.py           # info_gain, task_driven
│   │   ├── voi.py
│   │   ├── fixed.py
│   │   └── oracle.py
│   ├── actuation/
│   │   ├── README.md
│   │   ├── jammer_abstract.py
│   │   └── jammer_iq.py
│   ├── metrics/
│   │   ├── regret.py
│   │   ├── effect.py
│   │   ├── tracking.py
│   │   ├── info.py
│   │   ├── rmb.py
│   │   └── ci.py
│   ├── plots/
│   │   ├── curves.py
│   │   ├── rasters.py
│   │   ├── sweeps.py
│   │   └── belief.py
│   ├── analysis/
│   │   └── sweep.py
│   └── wirings/
│       ├── README.md             # "these are examples, copy and edit them"
│       ├── canonical.py          # emitter → channel → sensor → estimator → controller → actuator
│       └── raw_obs.py            # same without estimator
├── examples/
│   ├── 01_minimal_loop.ipynb     # the complete worked example (§10)
│   ├── 02_add_a_dial.ipynb       # take 01, apply with_delay, compare
│   ├── 03_belief_vs_raw.ipynb    # same controller fed by hmm_filter vs identity
│   └── 04_sweep.ipynb            # two-dial heatmap with analysis.sweep
├── tests/
│   ├── test_stage_contract.py    # every factory returns a Stage; init/step shapes are stable under scan+vmap
│   ├── test_dials.py             # each wrapper changes exactly what it claims
│   ├── test_tier_agreement.py    # abstract_channel + energy_detector vs iq_channel + channelizer
│   ├── test_placement_parity.py  # same loop, SIM vs loopback worker on localhost: identical logs
│   └── test_structure.py         # check_mlr/check_tp2 on known-good and known-bad kernels
└── docs/
    └── design.md                 # this file
```

---

## 10. Example notebook: `examples/01_minimal_loop.ipynb`

Each cell does one thing. Prose cells explain the *why*. This is the template every later experiment
copies.

1. **Imports and constants.** `C=8` channels, `K=8` Red states, `T=500` steps, `S=256` seeds.
2. **Build Red.** `red = reactive_fh_emitter(n_channels=C, dwell=4, react_threshold_db=3.0, react_steps=2)`.
3. **Build the channel.** `chan = abstract_channel(n_channels=C, path_loss_db=..., noise_db=...)`.
4. **Build Blue's sensor.** `sense = energy_detector(n_channels=C, snr_db=10.0)`.
5. **Build Blue's estimator.** `est = hmm_filter(P=red_transition_matrix(...), B=detector_kernel(...))`.
   Then `check_mlr(P, B)` — print the result; the notebook explains why it matters.
6. **Build Blue's controller.** `ctrl = certainty_equivalent(rule=jam_map_channel)`.
7. **Build the actuator.** `act = jammer_abstract(n_channels=C, power_db=...)`.
8. **Build the oracle.** `orc = oracle(rule=jam_map_channel)`.
9. **Write the wiring.** Shown in full, ~25 lines, every `.step` call visible:

   ```python
   def wire(st, s, key):
       k = split_for(st, key)
       s.red,  em      = st.red.step(s.red,   s.red_rx_prev, k["red"])
       s.chan, (rrx, brx, truth) = st.chan.step(s.chan, (em, s.act_prev), k["chan"])
       s.sense, obs    = st.sense.step(s.sense, brx, k["sense"])
       s.est,  belief  = st.est.step(s.est, obs, k["est"])
       s.ctrl, action  = st.ctrl.step(s.ctrl, belief, k["ctrl"])
       s.act,  jam     = st.act.step(s.act, action, k["act"])
       s.orc,  jam_orc = st.orc.step(s.orc, truth, k["orc"])
       s.red_rx_prev, s.act_prev = rrx, jam      # one-step feedback, explicit
       log = Log(truth=truth, obs=obs, belief=belief, action=action, jam=jam, jam_orc=jam_orc)
       return s, log
   ```

   (The one-step feedback — Red hears this step's jamming next step — is the only latency built
   into the canonical wiring. Everything else is a dial.)
9b. **Assemble the loop with placement.** All SIM for now, but written out so the hardware version is
   a one-line edit later:

   ```python
   loop = Loop(
       stages={"red": red, "chan": chan, "sense": sense, "est": est, "ctrl": ctrl, "act": act, "orc": orc},
       wire=wire,
       placement={
           "red":   Placement("sim", role="red"),     # later: Placement("pi-red.local", "pluto:0", "red")
           "chan":  Placement("sim"),                 # later: rf_channel / a bridge
           "sense": Placement("sim", role="blue"),    # later: Placement("localhost", "pluto:0", "blue")
           "est":   Placement("sim", role="blue"),    # later: Placement("jetson.local", None, "blue")
           "ctrl":  Placement("sim", role="blue"),
           "act":   Placement("sim", role="blue"),
           "orc":   Placement("sim", role="observer"),
       },
   )
   ```
10. **Run once with `backend="python"`.** Print shapes. Show `occupancy_raster`.
11. **Run the batch.** `log = run_batch(loop, key, T, S)`. Compute `regret`, plot `curve_with_ci`.
12. **Turn one dial.** Rebuild with `sense = with_delay(sense, 2)`, rerun, overlay. The notebook ends
    on this figure and one sentence: if the two curves don't separate, Red isn't reactive enough and
    the loop isn't closed yet.

---

## 11. Commenting standard

Every module has a header docstring with five sections:

```python
"""
ROLE IN LOOP   : where this stage sits and what it feeds.
STATE          : each field, dtype, shape, and what it means.
INPUTS         : pytree structure and shapes.
OUTPUTS        : pytree structure and shapes.
DOES NOT       : what you might expect this to do that it deliberately doesn't,
                 and which other piece does it instead.
"""
```

Inside `step`, comments explain intent, not syntax — why a `lax.select` instead of `where`, why the
ring buffer is rolled this direction, what a shape `[C, K]` means physically. Shape annotations go on
every line that creates or reshapes an array.

---

## 12. JAX rules enforced by convention and tests

- All state is `jnp` arrays in NamedTuples. No Python scalars, lists, or dicts in state.
- No Python `if` on traced values; use `lax.cond`, `lax.select`, `jnp.where`.
- Fixed shapes everywhere; delays and histories are ring buffers.
- PRNG keys are inputs, never state.
- Every factory's `step` must pass `test_stage_contract`: `jit(step)` compiles, `vmap(step)` over a
  batch of states works, and `lax.scan` over `n` steps returns stable shapes.
- `backend="python"` is the escape hatch for debugging and hardware, not a second code path — it calls
  the same `step` functions.

---

## 13. First implementation order

1. `core/` (stage, loop, placement, runner, types) + `test_stage_contract`. `remote_stage` and
   `worker.py` are built now against a loopback worker on localhost (`test_placement_parity`), so
   placement is real from day one even though every stage is SIM.
2. `emitters/reactive.py`, `channel/abstract.py`, `sensing/energy.py`, `estimation/identity.py`,
   `control/bandit.py`, `control/oracle.py`, `actuation/jammer_abstract.py`.
3. `metrics/regret.py`, `metrics/ci.py`, `plots/curves.py`, `plots/rasters.py`.
4. `examples/01_minimal_loop.ipynb` with `identity` as the estimator (bandit on raw obs).
5. `dials/delay.py`; `examples/02_add_a_dial.ipynb`.
6. `estimation/hmm.py`, `estimation/structure.py`, `control/certainty_equiv.py`; swap into example 01.

Everything after that is pulled in by whichever sandbox part is being written.
