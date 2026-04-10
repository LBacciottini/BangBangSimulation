# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simulation of swap-ASAP quantum repeater chains built on QuantumSavory.jl (Julia). Repeaters swap entangled qubits as soon as they hold two qubits facing opposite directions. The simulations evaluate **Bang-Bang scheduling**, which throttles entanglement generation rate on individual links to maintain system stability via a `slack` parameter.

## Build & Run

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run batch simulations (edit parameters inside scratch.jl first)
julia --project=. src/scratch.jl

# Interactive REPL with hot-reloading
julia --project=. -e 'using Revise; using BangBangSimulation'
```

Requires Julia 1.11+. Always activate the project in this folder (`--project=.`) when running Julia code.

## QuantumSavory.jl Reference

A full copy of the QuantumSavory.jl source is vendored at `QuantumSavory.jl/` in this repo. Its own `QuantumSavory.jl/CLAUDE.md` describes the internal structure in detail. Key paths within that tree:

- `src/ProtocolZoo/ProtocolZoo.jl` — protocol implementations (entangler, swapper, cutoff, consumer, tracker, etc.)
- `src/CircuitZoo/` — quantum circuits (e.g., `LocalEntanglementSwap`)
- Core files: `src/tags.jl`, `src/queries.jl`, `src/states_registers.jl`, `src/networks.jl`, `src/messagebuffer.jl`, `src/concurrentsim.jl`

### Key QuantumSavory APIs used by this project

- **`EntanglerProt`** (`ProtocolZoo.jl:181`): generates entangled pairs between two nodes. Important parameters:
  - `chooseslotA` / `chooseslotB` — `Int->Bool` filter functions controlling which register slots the entangler may use at each node (default: `alwaystrue`, i.e., any free slot). Passed to `findfreeslot()`.
  - `attempts` — max attempts per round (`-1` for infinite)
  - `rounds` — how many rounds (`-1` for infinite)
  - `margin` / `hardmargin` — minimum free slots required before attempting
  - `tag` — tag type to apply (we use `PrivateEntanglementCounterpart`)
- **`findfreeslot(reg; chooseslot, randomize, locked, margin)`** — finds a free slot in a register, filtered by `chooseslot(slot_index)::Bool`.
- **`CutoffProt`** — deletes qubits older than `retention_time`, optionally announcing via `EntanglementDelete`.
- **`EntanglementCounterpart`**, **`EntanglementHistory`**, **`EntanglementUpdateX/Z`**, **`EntanglementDelete`** — tag types for tracking entanglement state.

**When implementing new features that may use QuantumSavory API**, always read the relevant source files in `QuantumSavory.jl/src/` first to understand available functions, type signatures, and tag conventions. Do not guess at the API — check the source.

## Files to Ignore

- **`src/enhanced_qtcp.jl`** — Incomplete. Work-in-progress implementation of sequential swaps with congestion control (based on Bacciottini et al., "Leveraging Internet Principles to Build a Quantum Network", IEEE Network, 2025). Ignore this file and all references to it (including `setup_seq` in `setup.jl`) unless explicitly told otherwise.

## Source Architecture (`src/`)

### `BangBangSimulation.jl` — Module entry point
Exports `setup`, `setup_seq`, `dump_log`, `import_data`, `analyze_data`, `analyze_consumer_data`, `Scenario`, `bk_link_capacity`, `mean_ci`, `safe_analyze_consumer_data`, `default_config`, `sliding_window_analysis`, `run_one_sliding`, `run_one`. Includes `setup.jl`, `dataprocess.jl`, and `scenarios.jl`.

### `setup.jl` — Simulation wiring and protocol definitions
Core file. Builds the repeater chain topology and wires up all protocols as ConcurrentSim `@resumable` coroutines on a `RegisterNet`.

**`setup(nrepeaters, nslots, linkcapacity; ...)`** is the main entry point. It:
1. Creates a linear chain of `nrepeaters` + 2 nodes (Alice, repeaters, Bob) via `repeater_chain()`.
2. Starts a `CustomEntanglementTracker` at every node — handles classical swap notifications (`EntanglementUpdateX/Z`, `EntanglementDelete`) and forwards them through swap chains.
3. Starts an `EnhancedSwapperProt` at each repeater (from `enhanced_swapping.jl`).
4. Starts a `SchedulerProt` at each link — the **Bang-Bang controller**. It spawns one-shot `EntanglerProt` instances, promotes freshly generated `PrivateEntanglementCounterpart` tags to public `EntanglementCounterpart` tags, reads queue imbalance via `get_queue_state()`, and toggles between full rate and `capacity * (1 - slack)` rate using `should_slow_down()`.
5. Starts a `CustomEntanglementConsumer` at the two end nodes — finds matching `EntanglementCounterpart` tags across Alice and Bob, measures Z⊗Z and X⊗X observables, logs fidelity and timestamps.

**Tagging system** (tracks entanglement state on register slots):
- `PrivateEntanglementCounterpart` — freshly generated pair, visible only to the scheduler
- `EntanglementCounterpart` — public pair, consumed by swapper and consumer
- `EntanglementHistory` — record of past swaps, used for classical message forwarding

### `enhanced_swapping.jl` — Swap protocol with qubit selection policies
`EnhancedSwapperProt` runs at each repeater. Finds pairs of qubits entangled to nodes on opposite sides and performs Bell-state measurement + classical notification. Configurable `policy` parameter:
- `"OQF"` — oldest qubit first (default in `setup.jl`)
- `"YQF"` — youngest qubit first
- `"RAND"` — random selection (uses `chooseL`/`chooseH`/`chooseslots` functions)

### `dataprocess.jl` — Post-simulation analysis
- `import_data(filename)` — reads CSV logs into DataFrames
- `analyze_consumer_data(df)` — computes fidelity (mean Z⊗Z observable) and throughput from consumer logs (steady state: `time > 100.0`)
- `analyze_data(df, consumer_df)` — per-node throughput, sojourn time (Little's law), and fidelity. Converts raw observable to Bell-state fidelity via `(3F+1)/4`.

### `scratch.jl` — Batch runner script
`run_batch()` sweeps over `slack` values, runs simulations, collects results into a single DataFrame, and writes CSV. Configures link capacity, coherence time, number of repeaters/slots. Output goes to folders like `out_3_rep/`.
