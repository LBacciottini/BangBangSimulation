using Distributed
using Logging

# Optional parallelism via JULIA_NUM_WORKERS env var (default: 0 = sequential)
const _num_workers = parse(Int, get(ENV, "JULIA_NUM_WORKERS", "0"))
const _added_procs = if _num_workers > 0
    addprocs(_num_workers; exeflags="--project=.")
else
    Int[]
end

@everywhere using BangBangSimulation
@everywhere using Random
@everywhere using Statistics
@everywhere using DataFrames
@everywhere using CSV

# ── Link-length sweep (new default mode) ────────────────────────────────

"""
    adaptive_sim_times(linkcapacity; target_wall=20.0) -> (sim_time, steady_state_start)

Choose simulation time to yield roughly constant wall-clock time per job across
all link lengths.  Empirically, wall_time ≈ sim_time × 0.007 × rate, so we set
sim_time = target_wall / (0.007 × rate), clamped to [3, 120] seconds.
Warmup is half the sim time.
"""
function adaptive_sim_times(linkcapacity::Float64; target_wall::Float64=20.0)
    k = 0.007  # empirical: wall_seconds_per_sim_second ≈ k × rate
    sim_time = clamp(target_wall / (k * linkcapacity), 3.0, 120.0)
    steady_state_start = sim_time / 2
    return (sim_time, steady_state_start)
end

function run_sweep()
    cfg = default_config()

    # Configurable parameters from environment
    link_lengths_str = get(ENV, "LINK_LENGTHS", "5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0")
    link_lengths = parse.(Float64, split(link_lengths_str, ","))
    seed_count = parse(Int, get(ENV, "SEED_COUNT", "32"))
    target_wall = parse(Float64, get(ENV, "TARGET_WALL_S", "20.0"))
    out_root = get(ENV, "OUT_ROOT", "./out_scenarios")

    seeds = collect(1:seed_count)
    mkpath(out_root)

    run_fn = nworkers() > 1 ? pmap : map
    total_jobs = length(link_lengths) * length(cfg.scenarios) * seed_count
    @info "Sweep configuration" link_lengths seed_count target_wall nworkers=nworkers() total_jobs

    # Summary accumulator
    summary_rows = NamedTuple[]

    for (ll_idx, linklength_km) in enumerate(link_lengths)
        bk = bk_link_capacity(
            linklength_km=linklength_km,
            excitation_time_s=cfg.excitation_time_s,
            static_eff=cfg.static_eff,
        )
        linkcapacity = bk.rate

        # Determine sim_time and warmup for this link length
        sim_time, steady_state_start = adaptive_sim_times(linkcapacity; target_wall=target_wall)

        @info "Starting link length $(ll_idx)/$(length(link_lengths))" linklength_km linkcapacity sim_time steady_state_start

        # Override slack for scenarios with NaN slack (optimal per link length)
        scenarios_ll = map(cfg.scenarios) do s
            isnan(s.slack) ? Scenario(s.name, s.nslots, optimal_slack(linklength_km), s.policy, s.cutofftime) : s
        end

        # Build flat job list: (scenario, seed) pairs for this link length
        jobs = [(scenario, seed) for scenario in scenarios_ll for seed in seeds]

        all_results = run_fn(jobs) do (scenario, seed)
            result = run_one_steady(
                seed, scenario;
                nrepeaters=cfg.nrepeaters,
                linkcapacity=linkcapacity,
                linklength_km=linklength_km,
                coherencetime=cfg.coherencetime,
                sim_time=sim_time,
                steady_state_start=steady_state_start,
            )
            (scenario=scenario, result=result)
        end

        # Aggregate per scenario
        for scenario in scenarios_ll
            scenario_results = [r.result for r in all_results if r.scenario.name == scenario.name]
            fidelities = collect(Float64, (r.fidelity for r in scenario_results))
            throughputs = collect(Float64, (r.throughput for r in scenario_results))
            skrs = collect(Float64, (bb84_skr(r.fidelity, r.throughput) for r in scenario_results))

            mf, lf, hf = mean_ci(fidelities)
            mt, lt, ht = mean_ci(throughputs)
            ms, ls, hs = mean_ci(skrs)

            cutoff_val = isnothing(scenario.cutofftime) ? NaN : scenario.cutofftime
            row = (
                scenario=scenario.name,
                linklength_km=linklength_km,
                linkcapacity=linkcapacity,
                n=length(scenario_results),
                mean_fidelity=mf,
                ci_fidelity_low=lf,
                ci_fidelity_high=hf,
                mean_throughput=mt,
                ci_throughput_low=lt,
                ci_throughput_high=ht,
                mean_skr=ms,
                ci_skr_low=ls,
                ci_skr_high=hs,
                coherencetime=cfg.coherencetime,
                nslots=scenario.nslots,
                slack=scenario.slack,
                policy=scenario.policy,
                cutofftime=cutoff_val,
                sim_time=sim_time,
                steady_state_start=steady_state_start,
            )
            push!(summary_rows, row)

            @info "  $(scenario.name) @ $(linklength_km) km: fidelity=$(round(mf, digits=4)) [$(round(lf, digits=4)), $(round(hf, digits=4))], throughput=$(round(mt, digits=4)) [$(round(lt, digits=4)), $(round(ht, digits=4))]"
        end

        # Write incrementally after each link length so partial results are saved
        outpath = joinpath(out_root, "scenario_sweep.csv")
        summary_df = DataFrame(summary_rows)
        CSV.write(outpath, summary_df)

        @info "Completed link length $(ll_idx)/$(length(link_lengths)): $(linklength_km) km — results saved to $(outpath)"
    end

    outpath = joinpath(out_root, "scenario_sweep.csv")
    @info "Done. Summary written to $(outpath)" total_rows=length(summary_rows)

    # Write parameters CSV
    params = DataFrame(
        key=["nrepeaters", "coherencetime", "excitation_time_s", "static_eff",
             "target_wall_s", "seed_count", "link_lengths"],
        value=[string(cfg.nrepeaters), string(cfg.coherencetime), string(cfg.excitation_time_s),
               string(cfg.static_eff), string(target_wall),
               string(seed_count), link_lengths_str],
    )
    CSV.write(joinpath(out_root, "sweep_params.csv"), params)

    # Cleanup workers
    if !isempty(_added_procs)
        rmprocs(_added_procs)
    end
end

# ── Single link-length mode (old behavior) ──────────────────────────────

function run_all()
    cfg = default_config()
    seeds = collect(1:cfg.seed_count)

    mkpath(cfg.out_root)

    # Build flat job list: (scenario, seed) pairs
    jobs = [(scenario, seed) for scenario in cfg.scenarios for seed in seeds]

    run_fn = nworkers() > 1 ? pmap : map
    @info "Running $(length(jobs)) jobs with $(nworkers()) worker(s)..."

    all_results = run_fn(jobs) do (scenario, seed)
        result = run_one(
            seed,
            scenario;
            nrepeaters=cfg.nrepeaters,
            linkcapacity=cfg.linkcapacity,
            linklength_km=cfg.linklength_km,
            coherencetime=cfg.coherencetime,
            sim_time=cfg.sim_time,
            out_root=cfg.out_root,
        )
        (scenario=scenario, result=result)
    end

    # Group results by scenario and compute summaries
    summary_all = DataFrame(
        scenario=String[],
        n=Int[],
        mean_fidelity=Float64[],
        ci_fidelity_low=Float64[],
        ci_fidelity_high=Float64[],
        mean_throughput=Float64[],
        ci_throughput_low=Float64[],
        ci_throughput_high=Float64[],
        linklength_km=Float64[],
        linkcapacity=Float64[],
        coherencetime=Float64[],
        nslots=Int[],
        slack=Float64[],
        policy=String[],
        cutofftime=Float64[],
    )

    for scenario in cfg.scenarios
        scenario_results = [r.result for r in all_results if r.scenario.name == scenario.name]
        runs = DataFrame(scenario_results)

        fidelities = collect(runs.fidelity)
        throughputs = collect(runs.throughput)

        mf, lf, hf = mean_ci(fidelities)
        mt, lt, ht = mean_ci(throughputs)

        scenario_dir = joinpath(cfg.out_root, scenario.name)
        CSV.write(joinpath(scenario_dir, "runs.csv"), runs)

        cutoff_val = isnothing(scenario.cutofftime) ? NaN : scenario.cutofftime
        push!(
            summary_all,
            (
                scenario=scenario.name,
                n=length(scenario_results),
                mean_fidelity=mf,
                ci_fidelity_low=lf,
                ci_fidelity_high=hf,
                mean_throughput=mt,
                ci_throughput_low=lt,
                ci_throughput_high=ht,
                linklength_km=cfg.linklength_km,
                linkcapacity=cfg.linkcapacity,
                coherencetime=cfg.coherencetime,
                nslots=scenario.nslots,
                slack=scenario.slack,
                policy=scenario.policy,
                cutofftime=cutoff_val,
            ),
        )

        @info "$(scenario.name): fidelity=$(round(mf, digits=4)) [$(round(lf, digits=4)), $(round(hf, digits=4))], throughput=$(round(mt, digits=4)) [$(round(lt, digits=4)), $(round(ht, digits=4))]"

        CSV.write(joinpath(scenario_dir, "summary.csv"), summary_all[end:end, :])
    end

    CSV.write(joinpath(cfg.out_root, "summary_all.csv"), summary_all)
    CSV.write(joinpath(cfg.out_root, "bk_params.csv"), DataFrame(cfg.bk))

    @info "Done. Results in $(cfg.out_root)"

    # Clean up workers
    if !isempty(_added_procs)
        rmprocs(_added_procs)
    end
end

# ── Summarize-only mode ─────────────────────────────────────────────────

function summarize_only()
    cfg = default_config()

    summary_all = DataFrame(
        scenario=String[],
        n=Int[],
        mean_fidelity=Float64[],
        ci_fidelity_low=Float64[],
        ci_fidelity_high=Float64[],
        mean_throughput=Float64[],
        ci_throughput_low=Float64[],
        ci_throughput_high=Float64[],
        linklength_km=Float64[],
        linkcapacity=Float64[],
        coherencetime=Float64[],
        nslots=Int[],
        slack=Float64[],
        policy=String[],
        cutofftime=Float64[],
    )

    for scenario in cfg.scenarios
        scenario_dir = joinpath(cfg.out_root, scenario.name)
        if !isdir(scenario_dir)
            @warn "Scenario directory $(scenario_dir) does not exist. Skipping."
            continue
        end
        run_files = filter(f -> startswith(f, "run_seed") && endswith(f, ".csv"), readdir(scenario_dir; join=false))
        if isempty(run_files)
            @warn "No run files found for $(scenario.name) in $(scenario_dir). Skipping."
            continue
        end

        runs = reduce(vcat, (CSV.read(joinpath(scenario_dir, f), DataFrame) for f in run_files))
        CSV.write(joinpath(scenario_dir, "runs.csv"), runs)

        fidelities = collect(runs.fidelity)
        throughputs = collect(runs.throughput)

        mf, lf, hf = mean_ci(fidelities)
        mt, lt, ht = mean_ci(throughputs)

        cutoff_val = isnothing(scenario.cutofftime) ? NaN : scenario.cutofftime
        push!(
            summary_all,
            (
                scenario=scenario.name,
                n=nrow(runs),
                mean_fidelity=mf,
                ci_fidelity_low=lf,
                ci_fidelity_high=hf,
                mean_throughput=mt,
                ci_throughput_low=lt,
                ci_throughput_high=ht,
                linklength_km=cfg.linklength_km,
                linkcapacity=cfg.linkcapacity,
                coherencetime=cfg.coherencetime,
                nslots=scenario.nslots,
                slack=scenario.slack,
                policy=scenario.policy,
                cutofftime=cutoff_val,
            ),
        )

        CSV.write(joinpath(scenario_dir, "summary.csv"), summary_all[end:end, :])
    end

    CSV.write(joinpath(cfg.out_root, "summary_all.csv"), summary_all)
    CSV.write(joinpath(cfg.out_root, "bk_params.csv"), DataFrame(cfg.bk))

    @info "Done. Summaries in $(cfg.out_root)"
end

# ── CLI dispatch ────────────────────────────────────────────────────────

if length(ARGS) >= 1 && ARGS[1] == "--run-one"
    if length(ARGS) < 3
        error("Usage: julia --project=. src/run_scenarios.jl --run-one <scenario_name> <seed>")
    end
    scenario_name = ARGS[2]
    seed = parse(Int, ARGS[3])

    cfg = default_config()

    scenario = findfirst(s -> s.name == scenario_name, cfg.scenarios)
    scenario === nothing && error("Unknown scenario name $(scenario_name)")
    scenario = cfg.scenarios[scenario]

    run_one(
        seed,
        scenario;
        nrepeaters=cfg.nrepeaters,
        linkcapacity=cfg.linkcapacity,
        linklength_km=cfg.linklength_km,
        coherencetime=cfg.coherencetime,
        sim_time=cfg.sim_time,
        out_root=cfg.out_root,
    )

    @info "Done. Scenario $(scenario.name), seed $(seed)"
elseif length(ARGS) >= 1 && ARGS[1] == "--summarize"
    summarize_only()
elseif length(ARGS) >= 1 && ARGS[1] == "--single"
    run_all()
else
    run_sweep()
end

# ==========================
# How to Run (copy/paste)
# ==========================
#
# Run link-length sweep (default mode, 10 link lengths × 7 scenarios × 32 seeds):
#   julia --project=. src/run_scenarios.jl
#
# Run sweep with custom link lengths and parallelism:
#   LINK_LENGTHS="10.0,20.0,30.0" SEED_COUNT=16 JULIA_NUM_WORKERS=4 julia --project=. src/run_scenarios.jl
#
# Run single link-length batch (old behavior):
#   julia --project=. src/run_scenarios.jl --single
#
# Run single link-length batch in parallel:
#   JULIA_NUM_WORKERS=4 julia --project=. src/run_scenarios.jl --single
#
# Run a single scenario+seed:
#   julia --project=. src/run_scenarios.jl --run-one S1_infinite_no_bangbang 1
#
# Summarize after parallel runs:
#   julia --project=. src/run_scenarios.jl --summarize
#
# Environment variables:
#   LINK_LENGTHS         Comma-separated link lengths in km (default: 5,10,...,50)
#   SEED_COUNT           Number of seeds per scenario (default: 32)
#   SIM_TIME_S           Simulation time in seconds (default: 100.0)
#   STEADY_STATE_START   Discard events before this time (default: 50.0)
#   OUT_ROOT             Output directory (default: ./out_scenarios)
#   JULIA_NUM_WORKERS    Number of parallel workers (default: 0 = sequential)
#
# Outputs:
#   out_scenarios/scenario_sweep.csv    — aggregated results (sweep mode)
#   out_scenarios/sweep_params.csv      — parameters used
#   out_scenarios/summary_all.csv       — aggregated results (single mode)
