using Distributed

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

function adaptive_sim_times(linkcapacity::Float64; target_wall::Float64=20.0)
    k = 0.007
    sim_time = clamp(target_wall / (k * linkcapacity), 3.0, 120.0)
    steady_state_start = sim_time / 2
    return (sim_time, steady_state_start)
end

function run_s9_sweep()
    cfg = default_config()

    link_lengths_str = get(ENV, "LINK_LENGTHS", "5.0,9.0,13.0,17.0,21.0,25.0,29.0,33.0,37.0,40.0")
    link_lengths = parse.(Float64, split(link_lengths_str, ","))
    seed_count = parse(Int, get(ENV, "SEED_COUNT", "32"))
    target_wall = parse(Float64, get(ENV, "TARGET_WALL_S", "20.0"))
    out_root = get(ENV, "OUT_ROOT", "./out_scenarios")

    seeds = collect(1:seed_count)
    run_fn = nworkers() > 1 ? pmap : map

    s9_template = findfirst(s -> startswith(s.name, "S9"), cfg.scenarios)
    s9_template === nothing && error("No S9 scenario found")
    s9_base = cfg.scenarios[s9_template]

    total_jobs = length(link_lengths) * seed_count
    @info "S9-only sweep" link_lengths seed_count target_wall nworkers=nworkers() total_jobs

    new_rows = NamedTuple[]

    for (ll_idx, linklength_km) in enumerate(link_lengths)
        bk = bk_link_capacity(
            linklength_km=linklength_km,
            excitation_time_s=cfg.excitation_time_s,
            static_eff=cfg.static_eff,
        )
        linkcapacity = bk.rate
        sim_time, steady_state_start = adaptive_sim_times(linkcapacity; target_wall=target_wall)

        scenario = s9_base

        @info "Starting $(ll_idx)/$(length(link_lengths))" linklength_km linkcapacity sim_time

        results = run_fn(seeds) do seed
            run_one_steady(
                seed, scenario;
                nrepeaters=cfg.nrepeaters,
                linkcapacity=linkcapacity,
                linklength_km=linklength_km,
                coherencetime=cfg.coherencetime,
                sim_time=sim_time,
                steady_state_start=steady_state_start,
            )
        end

        fidelities = collect(Float64, (r.fidelity for r in results))
        throughputs = collect(Float64, (r.throughput for r in results))
        skrs = collect(Float64, (bb84_skr(r.fidelity, r.throughput) for r in results))

        mf, lf, hf = mean_ci(fidelities)
        mt, lt, ht = mean_ci(throughputs)
        ms, ls, hs = mean_ci(skrs)

        cutoff_val = isnothing(scenario.cutofftime) ? NaN : scenario.cutofftime
        row = (
            scenario=scenario.name,
            linklength_km=linklength_km,
            linkcapacity=linkcapacity,
            n=length(results),
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
        push!(new_rows, row)

        @info "  $(scenario.name) @ $(linklength_km) km: fidelity=$(round(mf, digits=4)), throughput=$(round(mt, digits=4)), skr=$(round(ms, digits=4))"
    end

    outpath = joinpath(out_root, "scenario_sweep.csv")
    existing = CSV.read(outpath, DataFrame)

    filter!(r -> !startswith(r.scenario, "S9"), existing)

    if !hasproperty(existing, :mean_skr)
        existing.mean_skr = bb84_skr.(existing.mean_fidelity, existing.mean_throughput)
        existing.ci_skr_low .= NaN
        existing.ci_skr_high .= NaN
    end

    new_df = DataFrame(new_rows)
    combined = vcat(existing, new_df; cols=:union)
    sort!(combined, [:linklength_km, :scenario])
    CSV.write(outpath, combined)

    @info "Done. Updated $(outpath) — added $(length(new_rows)) S9 rows"

    if !isempty(_added_procs)
        rmprocs(_added_procs)
    end
end

run_s9_sweep()
