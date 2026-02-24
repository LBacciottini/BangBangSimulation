using Distributed

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

"""
Run the 5 scenario batch (or per-seed jobs) for the 3-repeater chain with BK-derived
link capacity. See "How to Run" at the end of this file for copy/paste commands.
"""

function run_all()
    cfg = default_config()
    seeds = collect(1:cfg.seed_count)

    mkpath(cfg.out_root)

    # Build flat job list: (scenario, seed) pairs
    jobs = [(scenario, seed) for scenario in cfg.scenarios for seed in seeds]

    run_fn = nworkers() > 1 ? pmap : map
    println("Running $(length(jobs)) jobs with $(nworkers()) worker(s)...")

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

        CSV.write(joinpath(scenario_dir, "summary.csv"), summary_all[end:end, :])
    end

    CSV.write(joinpath(cfg.out_root, "summary_all.csv"), summary_all)
    CSV.write(joinpath(cfg.out_root, "bk_params.csv"), DataFrame(cfg.bk))

    println("Done. Results in $(cfg.out_root)")

    # Clean up workers
    if !isempty(_added_procs)
        rmprocs(_added_procs)
    end
end

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

    println("Done. Summaries in $(cfg.out_root)")
end

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

    println("Done. Scenario $(scenario.name), seed $(seed)")
elseif length(ARGS) >= 1 && ARGS[1] == "--summarize"
    summarize_only()
else
    run_all()
end

# ==========================
# How to Run (copy/paste)
# ==========================
#
# Run all scenarios sequentially (32 seeds each):
#   julia --project=. src/run_scenarios.jl
#
# Run all scenarios in parallel with 4 Julia workers:
#   JULIA_NUM_WORKERS=4 julia --project=. src/run_scenarios.jl
#
# Run per-seed jobs in parallel via shell (example with 4 processes):
#   for s in S1_infinite_no_bangbang S2_finite10_no_bangbang S3_infinite_yqf_no_bangbang S4_infinite_cutoff_no_bangbang S5_infinite_bangbang_slack30; do
#     for seed in $(seq 1 32); do
#       echo "$s $seed"
#     done
#   done | xargs -n2 -P4 bash -lc 'julia --project=. src/run_scenarios.jl --run-one "$0" "$1"'
#
# Summarize after parallel runs:
#   julia --project=. src/run_scenarios.jl --summarize
#
# Outputs:
#   out_scenarios/<scenario>/run_seed*.csv
#   out_scenarios/<scenario>/runs.csv
#   out_scenarios/<scenario>/summary.csv
#   out_scenarios/summary_all.csv
