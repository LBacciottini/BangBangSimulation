using Distributed

# Optional parallelism via JULIA_NUM_WORKERS env var (default: 4 to preserve existing behavior)
const _num_workers = parse(Int, get(ENV, "JULIA_NUM_WORKERS", "4"))
const _added_procs = if _num_workers > 0 && nworkers() < _num_workers
    addprocs(_num_workers - nworkers() + 1; exeflags="--project=.")
else
    Int[]
end

@everywhere using BangBangSimulation
@everywhere using DataFrames
@everywhere using CSV

function main()
    cfg = default_config()
    sim_time = parse(Float64, get(ENV, "SIM_TIME_S", "200.0"))
    window_size = parse(Float64, get(ENV, "WINDOW_SIZE_S", "10.0"))
    window_step = parse(Float64, get(ENV, "WINDOW_STEP_S", "1.0"))
    seed = parse(Int, get(ENV, "SEED", "42"))
    out_root = get(ENV, "OUT_ROOT", "./out_sliding")

    mkpath(out_root)

    # Save parameters
    params = DataFrame(
        key=["nrepeaters", "linklength_km", "coherencetime", "linkcapacity",
             "sim_time", "window_size", "window_step", "seed"],
        value=[string(cfg.nrepeaters), string(cfg.linklength_km), string(cfg.coherencetime),
               string(cfg.linkcapacity), string(sim_time), string(window_size),
               string(window_step), string(seed)],
    )
    CSV.write(joinpath(out_root, "bk_params.csv"), params)

    run_fn = nworkers() > 1 ? pmap : map
    println("Running $(length(cfg.scenarios)) scenarios with $(nworkers()) worker(s)...")
    println("  sim_time=$(sim_time)s, window_size=$(window_size)s, window_step=$(window_step)s, seed=$(seed)")

    results = run_fn(cfg.scenarios) do scenario
        println("  Starting scenario: $(scenario.name)")
        df = run_one_sliding(
            scenario;
            nrepeaters=cfg.nrepeaters,
            linkcapacity=cfg.linkcapacity,
            linklength_km=cfg.linklength_km,
            coherencetime=cfg.coherencetime,
            sim_time=sim_time,
            window_size=window_size,
            window_step=window_step,
            seed=seed,
        )
        # Write per-scenario CSV
        scenario_dir = joinpath(out_root, scenario.name)
        mkpath(scenario_dir)
        CSV.write(joinpath(scenario_dir, "sliding_window.csv"), df)
        println("  Finished scenario: $(scenario.name) ($(nrow(df)) windows)")
        return df
    end

    # Combine all results
    all_df = vcat(results...)
    CSV.write(joinpath(out_root, "sliding_window_all.csv"), all_df)
    println("Done. Combined results written to $(joinpath(out_root, "sliding_window_all.csv"))")
    println("Total rows: $(nrow(all_df))")

    # Clean up workers
    if !isempty(_added_procs)
        rmprocs(_added_procs)
    end
end

main()
