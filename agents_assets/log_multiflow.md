# Multi-Flow BangBang Refactor — Live Log

This file is a **living log** of the multi-flow refactor. Update it after every
non-trivial step so another agent can pick up where the current one left off if
the session is interrupted. Keep it append-only at the bottom (add dated
entries), but freely edit the top summary/status sections.

---

## Context

The current `BangBangSimulation.jl` simulates a single end-to-end flow on a swap-ASAP quantum repeater chain (Alice=1 → Bob=N). Bang-Bang scheduling throttles the per-link entanglement rate based on local queue imbalance (`slack`).

We want to extend the simulator so that **multiple flows can share the chain**, each living on an arbitrary contiguous sub-chain `[src, dst]`, with possibly overlapping spans. The goal is to showcase a **multi-flow BangBang link controller** that, having visibility of every flow using a link, assigns each freshly-generated Bell pair to the flow that needs it the most (the hungriest backlog). Baseline policies (e.g. static slot-pool partitioning per flow) will be added afterwards, so the code must be structured to accept them without churn.

The refactor must:
- Make single-flow a degenerate special case of multi-flow (one `Flow(1, 1, N)`) — no code duplication.
- Preserve the existing public API (`setup`, `run_one`, `run_one_sliding`, tests) via thin backward-compat wrappers.
- Cleanly separate three dimensions so any baseline can plug in later: **link controller** (bang-bang vs. others), **slot allocation** (shared pool vs. partitioned), and **swap policy** (OQF/YQF/RAND/...).

## High-level design

### 1. Flow as a first-class entity

A new `Flow` struct carries: `id::Int`, `src::Int`, `dst::Int` (with `src < dst`), and `name::String`. A helper `uses_link(f, u, v)` returns `true` iff `f.src ≤ min(u,v) && f.dst ≥ max(u,v)`. A `FlowRegistry` (or plain `Vector{Flow}`) indexes flows by id and by link.

### 2. Flow-aware tags (replaces single-flow tags everywhere)

The current tag system carries `(remote_node, remote_slot)` without any flow identity. We introduce a single unified tag type that is used in both single- and multi-flow modes:

```julia
@kwdef struct FlowEntanglementCounterpart
    flow_id::Int
    remote_node::Int
    remote_slot::Int
end
Tag(t::FlowEntanglementCounterpart) = Tag(FlowEntanglementCounterpart, t.flow_id, t.remote_node, t.remote_slot)
```

`PrivateEntanglementCounterpart` is **kept unchanged** (no flow_id). The rationale: `EntanglerProt` from QuantumSavory writes the private tag automatically with only `(remote_node, remote_slot)` — adding a field would require forking `EntanglerProt`. Instead, the flow_id is assigned **at promotion time** by the link controller. This matches the Bang-Bang semantics exactly: the controller sees all flow backlogs when the pair is born and writes the flow_id into the public tag.

A sibling `FlowEntanglementHistory(flow_id, past_remote_node, past_remote_slot, swap_node, swap_slot, local_swap_slot)` replaces `EntanglementHistory` so that swap-history forwarding preserves flow identity.

### 3. Flow-aware protocols (refactored from existing ones)

| Existing (single-flow) | New (flow-aware) |
|---|---|
| `CustomEntanglementTracker` | same name, queries/updates `FlowEntanglementCounterpart` and `FlowEntanglementHistory`, forwards classical messages with the `flow_id` preserved |
| `EnhancedSwapperProt` | `FlowSwapperProt` (or keep the name) — queries `FlowEntanglementCounterpart` filtered by `flow_id`, runs one logical swapper **per flow per repeater** OR one swapper that iterates over flows. See §5 |
| `CustomEntanglementConsumer` | `FlowEntanglementConsumer` — one process per flow, running at `flow.src`, consumes `FlowEntanglementCounterpart(flow.id, flow.dst, ❓)` |
| `SchedulerProt` | replaced by `AbstractLinkController` hierarchy (§4) |

### 4. `AbstractLinkController` hierarchy

```julia
abstract type AbstractLinkController <: QuantumSavory.ProtocolZoo.AbstractProtocol end

# Decides flow assignment dynamically based on backlog at adjacent nodes
@kwdef struct BangBangLinkController <: AbstractLinkController
    sim; net; nodeA::Int; nodeB::Int
    flows::Vector{Flow}                # flows traversing this link
    capacity::Float64
    slack::Float64 = 0.0               # single-flow: may throttle (backward compat); multi-flow: left at 0, no throttling
    chooseslotA::Function = alwaystrue
    chooseslotB::Function = alwaystrue
    out_filename::Union{String,Nothing} = nothing
    _log::Vector{...} = []
end
```

`slack` is preserved only for single-flow backward compatibility: the old `setup()` wrapper passes it through so existing scenarios continue to throttle. In true multi-flow runs, `setup_multiflow()` sets `slack=0.0` and the controller skips throttling entirely — the pure assignment story.

The `@resumable` loop:
1. Spawn a one-shot `EntanglerProt` at full `capacity` (same mechanism as today) using `chooseslotA/B`.
2. On success, compute each candidate flow's signed imbalance `imbalance(F) = qB_F - qA_F` (see §6).
3. Pick the flow with the **largest** imbalance; promote both slots' private tags to `FlowEntanglementCounterpart(flow_id, ...)`.
4. Log per-flow imbalances and the assignment decision.
5. Loop immediately — **no slack throttling**, the link always runs at full rate. Bang-Bang in multi-flow is purely an assignment mechanism.

Future baselines implement the same `AbstractLinkController` interface:
- `PartitionedLinkController` — spawns **one sub-entangler per flow**, each locked to a disjoint slot sub-pool via `chooseslotA/B`. The slot allocator is derived from a `SlotAllocationPolicy` argument.
- `RoundRobinLinkController` — rotates flow assignment without looking at backlogs.
- `StaticRateLinkController` — each flow gets a fixed share of `capacity`.

### 5. Swapper strategy

The swapper needs to pair two qubits on the same flow at a repeater. Two viable approaches:

- **(A) One process per (repeater, flow)**: each runs `FlowSwapperProt` filtered to its own `flow_id`. Simple, scales O(repeaters × flows).
- **(B) One process per repeater, iterating over flows each round**: queries all flows in turn. Fewer processes.

Recommended: **(A)** — cleaner, no artificial coupling between flows, maps trivially to the partitioned baseline, and parallelizes ConcurrentSim waits naturally. `enhancedfindswapablequbits` is extended to accept a `flow_id` and filter queryresults by `tag[2] == flow_id`.

### 6. Imbalance metric for BangBang flow assignment

The metric mirrors the single-flow `get_queue_state` / `should_slow_down` logic, filtered per flow. For a candidate flow `F` on link `(u, v)` (with `F.src ≤ u < v ≤ F.dst`), define a flow-restricted queue state:

```julia
get_flow_queue_state(node, net, F) :: Int
  # at `node`, count F-tagged entangled pairs facing "right within F"
  # minus pairs facing "left within F"
  qright = queryall(reg, FlowEntanglementCounterpart, F.id, r -> r > node && r <= F.dst, ❓)
  qleft  = queryall(reg, FlowEntanglementCounterpart, F.id, r -> r < node && r >= F.src, ❓)
  length(qright) - length(qleft)
```

Then per flow on the link compute:
```
qA_F = get_flow_queue_state(nodeA, net, F)   # signed imbalance at left endpoint, for F
qB_F = get_flow_queue_state(nodeB, net, F)   # signed imbalance at right endpoint, for F
imbalance(F) = qB_F - qA_F
```

The controller assigns the new pair to the flow with the **largest** `imbalance(F) = qB_F - qA_F`. Intuitively, this prioritizes the flow whose adjacent link endpoints are most mismatched in the direction that this link can repair. Ties are broken randomly among flows with equal hunger. Endpoint-flow special cases mirror the single-flow end-node rules (flow at its own src: ignore qA; at its own dst: ignore qB).

With one full-chain flow this reduces exactly to the current single-flow behaviour (no regression).

### 7. Restructured source layout

```
src/
  BangBangSimulation.jl         # module entry, exports, includes
  flows.jl                      # Flow struct, flows_on_link helper
  tags.jl                       # PrivateEntanglementCounterpart, FlowEntanglementCounterpart, FlowEntanglementHistory
  tracker.jl                    # CustomEntanglementTracker (flow-aware)
  swapper.jl                    # FlowSwapperProt (flow-aware enhancedfindswapablequbits) — replaces enhanced_swapping.jl
  consumer.jl                   # FlowEntanglementConsumer
  link_controllers/
    abstract.jl                 # AbstractLinkController + setup_link! dispatch
    bangbang.jl                 # BangBangLinkController (implemented now)
    # round_robin.jl, partitioned.jl, static_rate.jl — added later
  setup.jl                      # repeater_chain + setup_multiflow + backward-compat setup()
  dataprocess.jl                # unchanged
  scenarios.jl                  # Scenario extended to carry Vector{Flow} (default: single span-1-N flow)
  run_scenarios.jl              # unchanged API, consumes multi-flow output
  run_sliding_window.jl         # per-flow sliding window
```

Backward compatibility: `setup(nrepeaters, nslots, linkcapacity; kwargs...)` becomes a thin wrapper around `setup_multiflow(..., [Flow(1, 1, nrepeaters+2)])` returning `(sim, net, consumer, link_controllers)`. Existing tests and run scripts continue to work unchanged. The `Scenario` struct gains an optional `flows::Vector{Flow}` field defaulting to a single full-chain flow.

### 8. New multi-flow entry point

```julia
function setup_multiflow(
    nrepeaters::Int, nslots::Int, linkcapacity::AbstractFloat,
    flows::Vector{Flow};
    link_controller::Symbol = :bangbang,
    slot_allocation::Symbol = :shared,          # or :partitioned (future)
    swap_policy::String = "OQF",
    slack::AbstractFloat = 0.4,
    linklength::AbstractFloat = 0.0,
    coherencetime = nothing, cutofftime = nothing,
    outfolder::String = "./out/", outfile = nothing, usetempfile = false,
)
    # 1. repeater_chain
    # 2. trackers at every node
    # 3. optional CutoffProt at every node
    # 4. swappers: one per (repeater, flow) if :shared; one per flow with slot-pool filter if :partitioned
    # 5. link controllers: dispatch on link_controller symbol
    # 6. consumers: one per flow at flow.src
    return (sim, net, consumers::Vector, link_controllers::Vector)
end
```

Returns a vector of consumers (one per flow) so downstream analysis can compute per-flow fidelity and throughput.

**Output file layout**: one CSV per flow and one CSV per link:
- `consumer_flow{flow.id}_{name}.csv` — columns `time, obs1, obs2` (same schema as today, per flow).
- `scheduler_link{nodeA}_{nodeB}.csv` — columns `time, flow_id, qA, qB, assigned`: the imbalance snapshot and the flow chosen on each round.

Backward compat: single-flow `setup()` defaults to `consumer.csv` and `results.csv` filenames for the sole flow/link to avoid breaking existing scripts.

## Critical files to modify

- `src/BangBangSimulation.jl` — update exports and includes
- `src/setup.jl` — split into `flows.jl`, `tags.jl`, `tracker.jl`, `consumer.jl`, `link_controllers/abstract.jl`, `link_controllers/bangbang.jl`; keep `repeater_chain` + thin `setup()` wrapper here; add `setup_multiflow`
- `src/enhanced_swapping.jl` → `src/swapper.jl`: flow-aware `enhancedfindswapablequbits` (extra `flow_id` arg); `FlowSwapperProt` (or rename of `EnhancedSwapperProt`) queries by flow
- `src/scenarios.jl` — extend `Scenario` with optional `flows` field (default = full-chain)
- `test/runtests.jl` — update the one `EntanglementCounterpart` query in the "EntanglementDelete cleans remote slot" test (lines 77-91) to query `FlowEntanglementCounterpart`; add new multi-flow testset

## Multi-flow BangBang test plan

Tests to add in `test/runtests.jl`, new testset `"Multi-flow BangBang"`:

1. **Two disjoint flows** on a 6-node chain: `F1(1→3)` and `F2(4→6)`. Each flow's consumer should accumulate pairs independently. Assert `length(consumers[1]._log) > 0` and `length(consumers[2]._log) > 0`, and that no pair is logged with mismatched flow_id.
2. **Two overlapping flows** sharing one link: `F1(1→4)` and `F2(2→5)` on a 5-node chain. Run long enough to reach steady state; assert both flows make progress.
3. **Full overlap**: `F1(1→5)` and `F2(1→5)` on a 5-node chain — pure contention. Sanity-check that the assignment is roughly balanced (neither flow gets starved).
4. **Backward compatibility**: the existing single-flow testset must still pass unchanged.
5. **Tag flow-id integrity**: at end of run, every `FlowEntanglementCounterpart` tag on any slot must carry a `flow_id` present in the flow list.

## Execution steps (post-approval)

1. **Commit current repo state** (lots of untracked work) with a sensible message.
2. **Create `./agents_assets/log_multiflow.md`** containing a copy of this plan and a running execution log. Update after every non-trivial step, especially on problems/decisions, so another agent can resume.
3. **Refactor** in the order: `flows.jl` → `tags.jl` → `tracker.jl` → `swapper.jl` → `consumer.jl` → `link_controllers/{abstract,bangbang}.jl` → `setup.jl` (new `setup_multiflow` + wrap `setup`).
4. **Run the existing test suite** to confirm single-flow still passes. Fix fallout (most likely the `EntanglementCounterpart` query in the delete-propagation test, and any places still referencing the old tag).
5. **Implement multi-flow tests** above and make them pass.
6. **Pause and ask the user** about the multi-flow baseline variants (partitioned slot pools, round-robin, static rate, etc.) before touching them.

## Verification

- `julia --project=. -e 'using Pkg; Pkg.test()'` — existing tests must pass, plus the new multi-flow testset.
- Quick smoke run: `julia --project=. -e 'using BangBangSimulation; setup_multiflow(3, 10, 2.0, [Flow(1,1,3), Flow(2,3,5)]; linklength=10.0, coherencetime=10.0) |> (t -> (run(t[1], 30); [length(c._log) for c in t[3]]))'` — should return a vector with positive counts per flow.
- Manual inspection of a consumer log for `F1(1→3)` vs `F2(3→5)` to confirm no cross-flow pairs.

## Key architectural decisions (rationale captured here)

- **Why carry `flow_id` in the public tag, not the private tag?** `EntanglerProt` from QuantumSavory creates the private tag automatically with only `(remote_node, remote_slot)`; adding a field would require forking EntanglerProt. Promotion time is also the right moment for BangBang to decide "which flow is hungriest now".
- **Why one swapper per (repeater, flow)?** Cleaner, no artificial coupling, scales trivially to a partitioned slot-pool baseline.
- **Why drop slack in multi-flow?** User request. Multi-flow Bang-Bang is conceptually about *assignment*, not rate. Slack lives on only as a single-flow backward-compat knob.

---
## Status (as of 2026-04-11)

Multi-flow refactor is in. Both controllers (`:bangbang` shared-pool and `:reserved_pool` partitioned) work for the basic single-flow and small multi-flow cases. The full test suite passes (~524 tests). The current blocker is a multi-flow choke at high physical link rates: even a 2-flow `:bangbang` setup at 5 km wedges in less than 4 seconds of simulated time. Single-flow at the same point sustains ~600 Hz indefinitely. Resolving that wedge is the next milestone — see *Open issue* below.

## Summary of completed work

Condensed from a longer chronology that lived here previously. Each item is a milestone, not a dated entry.

### Phase 1 — Multi-flow refactor (single-flow becomes degenerate case)

- Introduced `Flow(id, src, dst, name)` and split sources into `flows.jl`, `tags.jl`, `tracker.jl`, `swapper.jl`, `consumer.jl`, `link_controllers/{abstract,bangbang}.jl`, `setup.jl`, `slot_allocations.jl`.
- New tags: `PrivateEntanglementCounterpart` (no `flow_id`, written by `EntanglerProt`) and `FlowEntanglementCounterpart(flow_id, …)` written at promotion time by the link controller. `FlowEntanglementHistory` mirrors swap history with `flow_id`.
- `FlowSwapperProt` runs one process per `(repeater, flow)` and only matches same-`flow_id` pairs. `FlowEntanglementConsumer` runs one per flow at `flow.src`.
- `BangBangLinkController` replaces the old `SchedulerProt`. Hunger metric is `q_F(B) - q_F(A)` per flow, where `q_F(node) = right_F(node) - left_F(node)`. Largest hunger wins; ties random.
- `setup_multiflow(nrepeaters, nslots, linkcapacity, flows; …)` is the new entry point. `setup(...)` is a thin wrapper that builds a single full-chain flow.
- Sign error in the original metric (was `min` instead of `max`) was found and fixed early — overlapping-flow runs went from zero throughput to working.
- Cutoff/age-check regression: the original `agelimit` path used `QuantumSavory.isolderthan` which is hard-wired to the old `EntanglementCounterpart` tag. Replaced with a direct `slot.reg.tag_info[qr.id][3]` lookup against `now(sim)` in `findswapablequbits_for_flow`.

### Phase 2 — Reserved-pool baseline (Part 2 in the log)

- Added `slot_allocations.jl` with `LinkSlotReservation`, `default_link_slot_indices` (returns `1:nA-2` and `3:nB` at internal repeaters — the directional reservation), `partitioned_link_slot_indices`, and `build_link_reservations(net, flows; slot_allocation=:shared|:equal_partitioned)`.
- Added `ReservedPoolLinkController`: round-robin cursor over flows on a link, **at most one live `EntanglerProt` per physical link at a time**, slot ranges restricted to the flow's reservation on both endpoints.
- `FlowSwapperProt` accepts `chooseslotsL` / `chooseslotsH` filters so hard reservations also bind the swapper layer.
- `setup_multiflow` now treats `link_controller` and `slot_allocation` as independent knobs; both `:bangbang/:shared` and `:reserved_pool/:equal_partitioned` are first-class.

### Phase 3 — Reserved-pool endpoint bug

- Symptom: `:reserved_pool` was returning all-zero deliveries even for a single full-overlap flow.
- Root cause: `build_link_reservations` was forcing `min(length(slotsA), length(slotsB))` on every link, which collapsed the endpoint side from `5 × nslots` (default `endnode_slot_multiplier = 5`) down to the repeater-side count. Endpoint links became severely undersized.
- Fix: removed the truncation. Each side keeps its full slot region; equal partitioning runs side-independently. Reservation-integrity test rewritten accordingly.
- Post-fix: 7-node, 5-flow reserved-pool delivers `[18, 14, 21, 9, 21]` over 120 s in the relaxed `(coherence=10s, cutoff=1s)` regime — alive again.

### Phase 4 — Bang-Bang multi-flow refinements (correction + ablation)

A first attempt attributed the multi-flow recovery to a single "slack-as-drop" change. An ablation forced a correction of that attribution. The actual situation is that **four small refinements** were added on top of an already-working baseline; none of them is "the fix", but together they buy ~5–7% throughput plus a real variance reduction in the symmetric many-flow regime. They are kept as the current default because none of them hurts.

The four refinements:

| change | contribution to 5-flow mean |
|---|---|
| defer flow selection until after entangler returns | 0% (within seed noise) |
| zero-hunger idle-flow tie-break in `select_flow_for_link` | +2.3% |
| `slack > 0` interpreted as drop probability when every flow has strictly negative hunger | +3.3% |
| 2-slot directional reservation in `default_link_slot_indices` (`1:nA-2`, `3:nB` at internal repeaters) | +2.8% |

`slack` no longer scales the entangler attempt rate. It now means "drop probability for fresh pairs when no flow has positive hunger". Single-flow numbers happen to land within noise of the old behavior in the test regimes, but downstream code that read `slack` as a rate factor needs review.

`per_flow_node_cap` (a hard cap on per-flow public inventory at each repeater) was tried and removed: it was either too tight (`K=20` deadlocked the 5-flow regime to zero deliveries) or beaten by the uncapped controller at every looser tested value.

A note about the historical `[6, 9, 5, 4, 7]` deadlock that this round was supposed to fix: it cannot be reproduced from the current code by any combination of these knobs. The intermediate broken `bangbang.jl` was an untracked file overwritten in this session, so the actual "killer fix" (whatever it was) happened earlier and is gone. All current numbers are stable; the historical dead state is irrecoverable but no longer relevant.

### Phase 5 — Comparison sweep groundwork

- Added a 7-node-chain comparison sweep harness in `agents_assets/sweep_prelim.jl` for the new experiment campaign: 5 identical full-overlap flows over a link length sweep, comparing `BB-nocutoff`, `BB-cutoff(0.2s)`, and `Pool-cutoff(0.2s)` baseline.
- Built physical link rates from `bk_link_capacity(linklength_km, excitation_time_s, static_eff)` so the sweep uses realistic per-length rates (e.g. ~750 Hz at 5 km, ~29 Hz at 40 km).
- Calibration probes turned up the issue described in *Open issue* below.

### Tests

- Full suite passes: `Test Summary: BangBangSimulation.jl | 524 passed`. Totals fluctuate by a few because several testsets iterate over `consumer._log` and assert per-entry, so timing differences shift the count slightly.
- Coverage includes: legacy single-flow, reservation-integrity, identical-flow fairness on both controllers, cutoff cleanup, basic delete-propagation regression.

---

## Open issue — multi-flow choke at high link rates

### Symptom

Calibration for the new comparison sweep at 5 km exposed a hard wedge in the multi-flow `:bangbang` shared-pool path. Single-flow runs are healthy, multi-flow runs collapse within ~3.6 seconds of simulated time and never recover. Same physical setup, same controller, same RNG seed.

Direct test (`agents_assets/scaling_choke.jl`, 5 km, 5 repeaters, `nslots=500`, `coherence=2s`, `cutoff=nothing`, `slack=0.3`, `swap_policy="YQF"`, seed 1, T=15 s):

| nflows | total deliveries | tmax (sim s) | per-flow counts | sustained? |
|---|---|---|---|---|
| 1 | 8640 | 15.0 (full) | `[8640]` | **yes — 576 Hz steady state** |
| 2 | 96 | 3.7 then dead | `[52, 44]` | no |
| 3 | 118 | 3.6 then dead | `[42, 40, 36]` | no |
| 5 | 150 | 3.6 then dead | `[32, 27, 26, 31, 34]` | no |

The pattern is too clean to be statistical:

- 1-flow runs the full simulation at full throughput.
- Any nflows ≥ 2 reaches a wedge at sim t ≈ 3.6 s. The wedge time is essentially flow-count-independent.
- Per-flow counts before the wedge are balanced — the selector is not starving any one flow.
- Total deliveries grow only slowly with nflows (96 → 118 → 150) because a few more pre-wedge events squeeze in before the chain dies.
- Even *before* the wedge, 2-flow throughput is ~26 Hz aggregate against 1-flow's 576 Hz steady state. So multi-flow is broken-slow, then dies.
- 500 slots per repeater is massively oversized for the rate, so the wedge is not memory pressure.

### Why this matters

This wedge was hidden by all earlier multi-flow numbers in this log because they used `nslots=50` and `linkcapacity=2.0` (artificial low rate). At realistic physical link rates (`bk_link_capacity` produces ~750 Hz at 5 km) the chain hits whatever this failure mode is in the first few seconds. None of the "Phase 4 refinements" prevent it: `slack=0.3`, idle tie-break, deferred selection, and directional reservation are all active during this run.

Until this is understood and fixed, the comparison sweep cannot be trusted, and the existing Bang-Bang fairness numbers in earlier sections may also be misleading at their tested rates.

---

## Investigation plan (next step)

The goal is to identify the exact failure mode at sim t ≈ 3.6 s with 2 flows at
5 km, then land the smallest code change that restores sustained throughput
without regressing the low-rate behavior already covered by tests.

### Phase 6 plan

1. Reproduce the choke with the existing `agents_assets/scaling_choke.jl`
   workflow and confirm the current code still shows the `nflows=1` healthy /
   `nflows>=2` wedged split.
2. Add focused diagnostics, not broad tracing:
   - per-link free-slot counts on the slot regions the entangler is allowed to
     use,
   - per-node public inventory split by flow and by left/right orientation,
   - swappable-pair counts per `(repeater, flow)`,
   - counts of assigned-vs-dropped pairs in `BangBangLinkController`.
3. Test the three main hypotheses against that instrumentation:
   - **H1: memory saturation by unswappable public pairs**. Expect free slots to
     collapse while swappable-pair counts go to zero and inventories remain
     nonzero.
   - **H2: bookkeeping leak / stuck locks / stale tags**. Expect free slots to
     remain available but some local state to become internally inconsistent
     (tagged-but-unassigned, assigned-but-untagged, or permanently locked).
   - **H3: controller-selection pathology**. Expect inventories to stay
     directional but the chosen flows or hungers to stop making physical sense.
4. If H1 is confirmed, change the controller/allocation logic so the
   multi-flow path stops manufacturing long-lived unswappable inventory at high
   rate. Keep the fix local to the current Bang-Bang path unless the data shows
   a deeper invariant violation.
5. Re-run:
   - the choke reproducer,
   - a small-rate multiflow smoke run,
   - the full `Pkg.test()` suite.

### First implementation target

Start by building an inspection script under `agents_assets/` that can stop a
run near the wedge and print, for every repeater and flow:

- left/right public inventory,
- number of swappable pairs currently visible,
- free-slot counts in the controller-allowed slot regions,
- any obvious tag/state inconsistencies.

That should make it clear whether we are looking at a queueing deadlock or a
state-corruption bug before touching the controller.

### 2026-04-11 — Phase 6 kickoff

- Updated `AGENTS.md` to match the current multi-flow codebase and reduced the
  root `CLAUDE.md` to a pointer to `AGENTS.md`.
- Investigation will begin from the existing 5 km choke reproducer, with the
  first milestone being a structural snapshot of the chain immediately before
  and after the wedge.

### 2026-04-11 — Reciprocity finding

- Added `agents_assets/check_reciprocity.jl` to scan live
  `FlowEntanglementCounterpart` tags for reciprocal consistency.
- First concrete corruption appears very early, at `t = 0.15 s`, well before
  the visible throughput wedge:
  - local slot `1.1` carried `FlowEntanglementCounterpart(flow=2, remote=5.42)`
  - remote slot `5.42` was already reused for a different pair
- This is not a consumer artifact. It is a known QuantumSavory slot-reuse issue:
  a slot can be recycled quickly enough that its old history/in-flight classical
  update path is effectively overwritten before the old remote endpoint has been
  retired cleanly.
- Conclusion for this repo: do **not** try to patch around this locally in the
  controller logic. Operate experiments in a regime where rapid slot reuse is
  unlikely, and use the reciprocity checker as a diagnostic sanity check when
  calibrating new sweeps.

### 2026-04-11 — Sweep retune

- Retuned `src/run_multiflow_5flow_comparison.jl` and
  `agents_assets/sweep_prelim.jl` to support a conservative
  `RATE_SCALE` (default `0.05`) so comparison jobs can be run in a lower-reuse
  regime by default.
- Added an optional reciprocity guard to those sweep paths:
  - `RECIPROCITY_GUARD=true|false`
  - `RECIPROCITY_CHECK_STEP_S=<seconds>`
- When the guard is enabled, runs stop incrementally and are marked failed if a
  first non-reciprocal public tag is observed instead of silently emitting
  misleading throughput/fidelity numbers.

### 2026-04-13 — Asset cleanup and tag-inconsistency root cause

- Cleaned `agents_assets/` down to the active diagnostics relevant to the
  current investigation. Removed stale one-off probes:
  `ablate.jl`, `ablate_slots.jl`, `compare_slack.jl`,
  `confirm_safe_regime_40km.jl`, `diagnose.jl`,
  `find_safe_regime_40km.jl`, `find_safe_regime_40km_small.jl`,
  `pyqf_probe.jl`, and `smoke_bb_vs_pool.jl`.
- Verified the multi-flow inconsistency is **not** rooted in the link
  controller. The first concrete multi-flow-only bug is in the classical
  swap-update path:
  - `FlowSwapperProt` used the stock QuantumSavory `EntanglementUpdateX/Z`
    message shape, which carries no `flow_id`.
  - `CustomEntanglementTracker` therefore matched incoming updates with
    `querydelete!(..., FlowEntanglementCounterpart, ❓, pastremotenode,
    pastremoteslotid)` and the analogous wildcard `FlowEntanglementHistory`
    query.
  - Under slot reuse, a stale update from flow A can legally match a newly
    reused slot for flow B if the physical `(remote_node, remote_slot)` pair is
    the same. That retags the flow-B slot with flow-A lineage. This failure mode
    is impossible in the degenerate single-flow case because every live slot has
    the same `flow_id`.
- Added a focused reproducer to `test/runtests.jl` that creates exactly that
  stale-message scenario and checks that the slot is left unchanged.
- Fix landed:
  - added flow-aware swap-update tags `FlowEntanglementUpdateX/Z`,
  - updated `FlowSwapperProt` to send those messages,
  - updated `CustomEntanglementTracker` to require exact `flow_id` matches for
    update/history handling.
- Verification:
  - direct stale-message reproducer now leaves the wrong-flow slot unchanged and
    emits only the expected "unhandleable message" warning,
  - full suite passes again: `Test Summary: BangBangSimulation.jl | 530 passed`.
- Important correction to the earlier "known QuantumSavory bug" attribution:
  the upstream rapid-slot-reuse hazard is still real, but the
  **multi-flow-only** inconsistency investigated here was a local design bug in
  our tracker/message path, not a scheduler bug.

### 2026-04-13 — Remaining reciprocity failure is not multi-flow-specific

- Rechecked the high-rate 5 km regime after the tracker `flow_id` fix with the
  reciprocity guard:
  - `nflows=1` still fails at `t = 0.15 s`
  - `nflows=2` still fails at `t = 0.15 s`
  - `nflows=5` still fails at `t = 0.15 s`
- So the remaining inconsistency is **not** multi-flow-specific.

First failing single-flow state (`nflows=1`):

- local slot `1.67`: `FlowEntanglementCounterpart(flow=1, remote=5.131)`
- remote slot `5.131`: `assigned = false`, no counterpart, but carries
  `FlowEntanglementHistory(flow=1, past_remote=2.133, swap_node=7, swap_slot=69, local_swap_slot=136)`

First failing two-flow state (`nflows=2`):

- local slot `1.3`: `FlowEntanglementCounterpart(flow=2, remote=5.9)`
- remote slot `5.9`: still assigned, but now carries
  `FlowEntanglementCounterpart(flow=2, remote=6.9)` **plus**
  `FlowEntanglementHistory(flow=2, past_remote=2.1, swap_node=7, swap_slot=7, local_swap_slot=8)`

Interpretation:

- the old remote endpoint has already been swapped/reused locally at node 5,
  but node 1 still holds a live public counterpart pointing at that old slot;
- in the 2-flow case the remote slot has already been recycled into a new pair
  before the old endpoint at node 1 was retired;
- this is exactly the rapid slot-reuse / stale classical-update hazard suspected
  earlier, and it reproduces even in single-flow, so it is not caused by
  multi-flow tag handling.

### 2026-04-13 — Root cause of the deep-chain collapse

- The remaining single-flow and multi-flow deep-chain failure was traced to the
  custom tracker's "mirror history" rewrite in `src/tracker.jl`.
- When a stale update hit a slot that had already been swapped away, the
  tracker correctly refreshed that slot's `FlowEntanglementHistory` and
  forwarded the update to the swap target.
- But our implementation also tried to rewrite the second measured swap slot's
  history to keep the pair of history tags symmetric.
- That extra rewrite is wrong in deep swap cascades: it mutates the partner
  branch even though the incoming update only traversed one forwarding branch.
  The result is that valid history chains get overwritten with stale remote
  endpoints.
- Evidence:
  - before disabling the mirror rewrite, a 7-node single-flow chain at `20 km`,
    `slack=0`, `t=10 s` had `1408` bad public endpoints and only about
    `53 Hz` end-to-end rate;
  - after disabling it, the same regime had `0` public reciprocity mismatches
    and roughly full physical throughput.
- Post-fix single-flow `20 km`, `slack=0`, `15 s` rates with `500` slots:
  - `0` repeaters: `133.47 Hz`
  - `1` repeater: `136.27 Hz`
  - `2` repeaters: `134.60 Hz`
  - `3` repeaters: `131.93 Hz`
  - `4` repeaters: `132.33 Hz`
  - `5` repeaters: `132.27 Hz`
- Added regression tests to catch this class of failure earlier:
  - a stale cross-flow tracker update test,
  - a `slack=0` single-link physical-rate sanity check,
  - and a high-rate deep single-flow reciprocity/throughput regression.
