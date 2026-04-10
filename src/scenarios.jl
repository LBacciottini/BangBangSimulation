using Statistics
using Random

function binary_entropy(x::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    return -x * log2(x) - (1.0 - x) * log2(1.0 - x)
end

function bb84_skr(fidelity::Float64, throughput::Float64)
    qber = 2.0 * (1.0 - fidelity) / 3.0
    r = 1.0 - 2.0 * binary_entropy(qber)
    return throughput * max(0.0, r)
end

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

function safe_analyze_consumer_data(df::DataFrame; steady_state_start::Float64=100.0)
    steady_state = filter(row -> row.time > steady_state_start, df)
    if nrow(steady_state) < 2
        return (NaN, NaN)
    end
    fidelity_raw, throughput = analyze_consumer_data(df; steady_state_start)
    fidelity = (3 * fidelity_raw + 1) / 4
    return (fidelity, throughput)
end

"""
    optimal_slack(linklength_km) -> Float64

Interpolate the optimal Bang-Bang slack for a given link length (km).
Based on numerical optimization results.
"""
function optimal_slack(linklength_km::Float64)
    # (length_km, optimal_slack) from optimization
    table = [
        ( 5.0, 0.0030),
        (10.0, 0.0030),
        (15.0, 0.2438),
        (20.0, 0.3745),
        (25.0, 0.4781),
        (30.0, 0.5797),
        (35.0, 0.6989),
        (40.0, 0.8531),
    ]
    # Clamp to table range
    linklength_km <= table[1][1] && return table[1][2]
    linklength_km >= table[end][1] && return table[end][2]
    # Linear interpolation
    for i in 1:length(table)-1
        l1, s1 = table[i]
        l2, s2 = table[i+1]
        if l1 <= linklength_km <= l2
            t = (linklength_km - l1) / (l2 - l1)
            return s1 + t * (s2 - s1)
        end
    end
    return table[end][2]
end

function default_config()
    nrepeaters = 3
    sim_time = parse(Float64, get(ENV, "SIM_TIME_S", "100.0"))
    nslots_infinite = parse(Int, get(ENV, "NSLOTS_INF", "500"))
    nslots_finite = 6
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
        Scenario("S2_finite6_no_bangbang", nslots_finite, 0.0, "OQF", nothing),
        Scenario("S3_finite50_no_bangbang", 50, 0.0, "OQF", nothing),
        Scenario("S4_infinite_yqf_no_bangbang", nslots_infinite, 0.0, "YQF", nothing),
        Scenario("S7_infinite_bangbang_optslack", nslots_infinite, NaN, "YQF", nothing),
        Scenario("S8_infinite_pyqf_no_bangbang", nslots_infinite, 0.0, "PYQF", nothing),
        Scenario("S9_infinite_yqf_cutoff02", nslots_infinite, 0.0, "YQF", 0.2),
        Scenario("S10_infinite_bangbang_cutoff02", nslots_infinite, NaN, "YQF", 0.2),
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

function run_one_steady(
    seed::Int, scenario::Scenario;
    nrepeaters::Int, linkcapacity::Float64, linklength_km::Float64,
    coherencetime::Float64, sim_time::Float64,
    steady_state_start::Float64=50.0,
)
    Random.seed!(seed)
    tmpdir = mktempdir()
    sim, _, consumer, _ = setup(
        nrepeaters, scenario.nslots, linkcapacity;
        linklength=linklength_km, slack=scenario.slack,
        coherencetime=coherencetime, cutofftime=scenario.cutofftime,
        policy=scenario.policy, outfolder=tmpdir, usetempfile=true,
    )
    ConcurrentSim.run(sim, sim_time)
    log = consumer._log
    steady = filter(e -> e.t > steady_state_start, log)
    if length(steady) < 2
        return (seed=seed, fidelity=NaN, throughput=NaN)
    end
    fidelity_raw = mean(e.obs1 for e in steady)
    fidelity = (3 * fidelity_raw + 1) / 4
    throughput = length(steady) / (maximum(e.t for e in steady) - minimum(e.t for e in steady))
    return (seed=seed, fidelity=fidelity, throughput=throughput)
end
