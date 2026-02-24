using Statistics
using Random

struct Scenario
    name::String
    nslots::Int
    slack::Float64
    policy::String
    cutofftime::Union{Float64,Nothing}
end

function bk_link_capacity(; linklength_km::Float64, excitation_time_s::Float64, static_eff::Float64)
    speed_km_s = 204_000.0
    loss_db_per_km = 0.2
    arm_km = linklength_km / 2
    attenuation_db = loss_db_per_km * arm_km
    p_det = static_eff * 10.0^(-attenuation_db / 10.0)
    p_success = (p_det^2) / 2
    attempt_time = (linklength_km / speed_km_s) + excitation_time_s
    rate = p_success / attempt_time
    return (; p_det, p_success, attempt_time, rate)
end

function mean_ci(x::Vector{Float64})
    valid = filter(!isnan, x)
    n_dropped = length(x) - length(valid)
    n_dropped > 0 && @warn "mean_ci: dropped $n_dropped NaN values out of $(length(x))"
    n = length(valid)
    n == 0 && return (NaN, NaN, NaN)
    m = mean(valid)
    n <= 1 && return (m, m, m)
    s = std(valid)
    s == 0.0 && return (m, m, m)
    z = 1.96
    se = s / sqrt(n)
    return (m, m - z * se, m + z * se)
end

function safe_analyze_consumer_data(df::DataFrame)
    steady_state = filter(row -> row.time > 100.0, df)
    if nrow(steady_state) < 2
        return (NaN, NaN)
    end
    fidelity_raw, throughput = analyze_consumer_data(df)
    fidelity = (3 * fidelity_raw + 1) / 4
    return (fidelity, throughput)
end

function default_config()
    nrepeaters = 3
    sim_time = parse(Float64, get(ENV, "SIM_TIME_S", "200.0"))
    nslots_infinite = parse(Int, get(ENV, "NSLOTS_INF", "5000"))
    nslots_finite = 10
    linklength_km = parse(Float64, get(ENV, "LINKLENGTH_KM", "20.0"))
    coherencetime = parse(Float64, get(ENV, "COHERENCE_TIME_S", "2.0"))
    excitation_time_s = parse(Float64, get(ENV, "EXCITATION_TIME_S", "17e-6"))
    static_eff = parse(Float64, get(ENV, "STATIC_EFF", "0.28"))
    out_root = get(ENV, "OUT_ROOT", "./out_scenarios")
    seed_count = parse(Int, get(ENV, "SEED_COUNT", "32"))

    bk = bk_link_capacity(linklength_km=linklength_km, excitation_time_s=excitation_time_s, static_eff=static_eff)
    linkcapacity = bk.rate

    scenarios = [
        Scenario("S1_infinite_no_bangbang", nslots_infinite, 0.0, "OQF", nothing),
        Scenario("S2_finite10_no_bangbang", nslots_finite, 0.0, "OQF", nothing),
        Scenario("S3_infinite_yqf_no_bangbang", nslots_infinite, 0.0, "YQF", nothing),
        Scenario("S4_infinite_cutoff_no_bangbang", nslots_infinite, 0.0, "OQF", coherencetime / 50.0),
        Scenario("S5_infinite_bangbang_slack30", nslots_infinite, 0.30, "OQF", nothing),
    ]

    return (;
        nrepeaters,
        sim_time,
        nslots_infinite,
        nslots_finite,
        linklength_km,
        coherencetime,
        excitation_time_s,
        static_eff,
        out_root,
        seed_count,
        bk,
        linkcapacity,
        scenarios,
    )
end

function sliding_window_analysis(
    log::Vector{@NamedTuple{t::Float64, obs1::Float64, obs2::Float64}};
    window_size::Float64=10.0,
    step::Float64=1.0,
    start_time::Float64=window_size,
)::DataFrame
    isempty(log) && return DataFrame(time=Float64[], fidelity=Float64[], throughput=Float64[])

    times_vec = [entry.t for entry in log]
    obs1_vec = [entry.obs1 for entry in log]

    max_time = maximum(times_vec)
    window_centers = start_time:step:max_time

    result_time = Float64[]
    result_fidelity = Float64[]
    result_throughput = Float64[]

    for t in window_centers
        lo = t - window_size
        # Half-open interval (lo, t]: exclude exact lower bound
        i_first = searchsortedfirst(times_vec, lo, lt=(<=))
        i_last = searchsortedlast(times_vec, t)

        if i_first > i_last
            push!(result_time, t)
            push!(result_fidelity, NaN)
            push!(result_throughput, 0.0)
        else
            window_obs1 = @view obs1_vec[i_first:i_last]
            fidelity_raw = mean(window_obs1)
            fidelity = (3 * fidelity_raw + 1) / 4
            count = i_last - i_first + 1
            throughput = count / window_size

            push!(result_time, t)
            push!(result_fidelity, fidelity)
            push!(result_throughput, throughput)
        end
    end

    return DataFrame(time=result_time, fidelity=result_fidelity, throughput=result_throughput)
end

function run_one(
    seed::Int, scenario::Scenario;
    nrepeaters::Int, linkcapacity::Float64, linklength_km::Float64,
    coherencetime::Float64, sim_time::Float64, out_root::String,
)
    Random.seed!(seed)

    scenario_dir = joinpath(out_root, scenario.name)
    mkpath(scenario_dir)

    sched_file = "scheduler_seed$(seed).csv"
    consumer_file = "consumer_seed$(seed).csv"

    sim, _, consumer, _ = setup(
        nrepeaters,
        scenario.nslots,
        linkcapacity;
        linklength=linklength_km,
        slack=scenario.slack,
        coherencetime=coherencetime,
        cutofftime=scenario.cutofftime,
        policy=scenario.policy,
        outfolder=scenario_dir,
        outfile=sched_file,
        usetempfile=false,
    )

    ConcurrentSim.run(sim, sim_time)
    dump_log(consumer, scenario_dir, consumer_file)

    df = import_data(joinpath(scenario_dir, consumer_file))
    fidelity, throughput = safe_analyze_consumer_data(df)

    result = (seed=seed, fidelity=fidelity, throughput=throughput)
    CSV.write(joinpath(scenario_dir, "run_seed$(seed).csv"), DataFrame([result]))
    return result
end

function run_one_sliding(
    scenario::Scenario;
    nrepeaters::Int, linkcapacity::Float64, linklength_km::Float64,
    coherencetime::Float64, sim_time::Float64,
    window_size::Float64=10.0, window_step::Float64=1.0, seed::Int=42,
)::DataFrame
    Random.seed!(seed)
    tmpdir = mktempdir()
    sim, net, consumer, schedulers = setup(
        nrepeaters, scenario.nslots, linkcapacity;
        linklength=linklength_km, slack=scenario.slack,
        coherencetime=coherencetime, cutofftime=scenario.cutofftime,
        policy=scenario.policy, outfolder=tmpdir, usetempfile=true,
    )
    ConcurrentSim.run(sim, sim_time)
    df = sliding_window_analysis(consumer._log; window_size=window_size, step=window_step)
    df[!, :scenario] .= scenario.name
    return df
end
