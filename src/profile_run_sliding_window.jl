using Distributed
using Profile
using ConcurrentSim

# Optional parallelism via JULIA_NUM_WORKERS env var (default: 1 for cleaner profiling)
const _num_workers = parse(Int, get(ENV, "JULIA_NUM_WORKERS", "1"))
const _added_procs = if _num_workers > 0 && nworkers() < _num_workers
    addprocs(_num_workers - nworkers() + 1; exeflags="--project=.")
else
    Int[]
end

@everywhere using BangBangSimulation
@everywhere using DataFrames
@everywhere using CSV
@everywhere using Random

const EVENT_PROCESSED = Dict{DataType, Int}()
const EVENT_SEEN = Set{UInt}()
const EVENT_BY_PROTOCOL = Dict{Tuple{DataType, String}, Int}()
const EVENT_BY_RESOURCE = Dict{Tuple{DataType, String}, Int}()

function reset_event_counts!()
    empty!(EVENT_PROCESSED)
    empty!(EVENT_SEEN)
    empty!(EVENT_BY_PROTOCOL)
    empty!(EVENT_BY_RESOURCE)
end

function _bbs_count_processed(ev)
    id = ev.bev.id
    if !(id in EVENT_SEEN)
        push!(EVENT_SEEN, id)
        t = typeof(ev)
        EVENT_PROCESSED[t] = get(EVENT_PROCESSED, t, 0) + 1
    end
    return nothing
end

function _bbs_event_type_from_callbacks(bev::ConcurrentSim.BaseEvent)
    for cb in bev.callbacks
        nfields = fieldcount(typeof(cb))
        for i in 1:nfields
            v = getfield(cb, i)
            if v isa ConcurrentSim.AbstractEvent
                return typeof(v)
            end
        end
    end
    return nothing
end

function _bbs_find_nested(val, ::Type{T}, depth::Int=2) where {T}
    val isa T && return val
    depth <= 0 && return nothing
    if val isa Tuple
        for v in val
            found = _bbs_find_nested(v, T, depth - 1)
            found !== nothing && return found
        end
    elseif val isa NamedTuple
        for v in values(val)
            found = _bbs_find_nested(v, T, depth - 1)
            found !== nothing && return found
        end
    else
        nfields = fieldcount(typeof(val))
        for i in 1:nfields
            v = nothing
            try
                v = getfield(val, i)
            catch err
                if err isa UndefRefError
                    continue
                else
                    rethrow(err)
                end
            end
            found = _bbs_find_nested(v, T, depth - 1)
            found !== nothing && return found
        end
    end
    return nothing
end

function _bbs_find_in_callback(cb, ::Type{T}) where {T}
    nfields = fieldcount(typeof(cb))
    for i in 1:nfields
        v = getfield(cb, i)
        found = _bbs_find_nested(v, T, 2)
        found !== nothing && return found
    end
    return nothing
end

function _bbs_protocol_label(proc::ConcurrentSim.Process)
    return string(typeof(proc.fsmi))
end

function _bbs_resource_label(res::ConcurrentSim.AbstractResource)
    return string(typeof(res)) * "#" * string(objectid(res))
end

function _bbs_step_count(sim::ConcurrentSim.Simulation)
    isempty(sim.heap) && throw(ConcurrentSim.EmptySchedule())
    (bev, key) = ConcurrentSim.DataStructures.peek(sim.heap)
    ConcurrentSim.DataStructures.dequeue!(sim.heap)
    sim.time = key.time
    bev.state = ConcurrentSim.processed

    ev_type = _bbs_event_type_from_callbacks(bev)
    if ev_type === nothing
        ev_type = ConcurrentSim.BaseEvent
    end
    EVENT_PROCESSED[ev_type] = get(EVENT_PROCESSED, ev_type, 0) + 1

    prot_label = "(none)"
    res_label = "(none)"
    for cb in bev.callbacks
        if prot_label == "(none)"
            proc = _bbs_find_in_callback(cb, ConcurrentSim.Process)
            if proc !== nothing
                prot_label = _bbs_protocol_label(proc)
            end
        end
        if res_label == "(none)"
            res = _bbs_find_in_callback(cb, ConcurrentSim.AbstractResource)
            if res !== nothing
                res_label = _bbs_resource_label(res)
            end
        end
        if prot_label != "(none)" && res_label != "(none)"
            break
        end
    end
    EVENT_BY_PROTOCOL[(ev_type, prot_label)] = get(EVENT_BY_PROTOCOL, (ev_type, prot_label), 0) + 1
    EVENT_BY_RESOURCE[(ev_type, res_label)] = get(EVENT_BY_RESOURCE, (ev_type, res_label), 0) + 1

    for callback in bev.callbacks
        callback()
    end
end

function _bbs_run(env::ConcurrentSim.Environment, until::ConcurrentSim.AbstractEvent)
    ConcurrentSim.append_callback(ConcurrentSim.stop_simulation, until)
    try
        while true
            _bbs_step_count(env)
        end
    catch exc
        if isa(exc, ConcurrentSim.StopSimulation)
            return exc.value
        else
            rethrow(exc)
        end
    end
end

function _bbs_run(env::ConcurrentSim.Environment, until::Number=Inf)
    return _bbs_run(env, ConcurrentSim.timeout(env, until - ConcurrentSim.now(env)))
end


function print_event_counts()
    function _print_table(title, dict)
        println(title)
        entries = sort(collect(dict), by = x -> -x[2])
        if isempty(entries)
            println("  (none)")
            return
        end
        for (t, cnt) in entries
            println("  $(lpad(cnt, 8))  $(t)")
        end
    end
    _print_table("Events processed (by type):", EVENT_PROCESSED)

    println("Events by protocol (top 15):")
    proto_entries = sort(collect(EVENT_BY_PROTOCOL), by = x -> -x[2])
    for (k, cnt) in proto_entries[1:min(end, 15)]
        (ev_type, prot_label) = k
        println("  $(lpad(cnt, 8))  $(ev_type)  $(prot_label)")
    end

    println("Events by resource (top 15):")
    res_entries = sort(collect(EVENT_BY_RESOURCE), by = x -> -x[2])
    for (k, cnt) in res_entries[1:min(end, 15)]
        (ev_type, res_label) = k
        println("  $(lpad(cnt, 8))  $(ev_type)  $(res_label)")
    end
end

function parse_bool(s::AbstractString)
    v = lowercase(strip(s))
    return v in ("1", "true", "t", "yes", "y", "on")
end

function run_one_sliding_profile(
    scenario;
    nrepeaters::Int, linkcapacity::Float64, linklength_km::Float64,
    coherencetime::Float64, sim_time::Float64,
    window_size::Float64, window_step::Float64, seed::Int,
)
    Random.seed!(seed)
    tmpdir = mktempdir()
    sim, net, consumer, schedulers = setup(
        nrepeaters, scenario.nslots, linkcapacity;
        linklength=linklength_km, slack=scenario.slack,
        coherencetime=coherencetime, cutofftime=scenario.cutofftime,
        policy=scenario.policy, outfolder=tmpdir, usetempfile=true,
    )
    _bbs_run(sim, sim_time)
    df = sliding_window_analysis(consumer._log; window_size=window_size, step=window_step)
    df[!, :scenario] .= scenario.name
    return df
end

function run_one_scenario(scenario, cfg; sim_time, window_size, window_step, seed)
    return run_one_sliding_profile(
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
end

function parse_count_before_at(line::AbstractString)
    if !occursin("@", line)
        return 1
    end
    prefix = split(line, "@"; limit=2)[1]
    nums = collect(eachmatch(r"\\d+", prefix))
    return isempty(nums) ? 1 : parse(Int, nums[end].match)
end

function query_caller_summary()
    io = IOBuffer()
    Profile.print(io; format=:tree, sortedby=:count, maxdepth=30, mincount=1, C=false)
    lines = split(String(take!(io)), '\n')

    callers = Dict{String, Int}()
    stack = Vector{Tuple{Int, String}}()

    for line in lines
        isempty(strip(line)) && continue
        occursin("@", line) || continue

        first_nonspace = findfirst(c -> c != ' ', line)
        first_nonspace === nothing && continue
        indent = first_nonspace - 1

        while !isempty(stack) && stack[end][1] >= indent
            pop!(stack)
        end

        if occursin("queries.jl", line)
            cnt = parse_count_before_at(line)
            parent = "(root)"
            # Walk up to find a non-query, non-dict/hash parent for attribution
            for i in length(stack):-1:1
                candidate = stack[i][2]
                if !occursin("queries.jl", candidate) &&
                   !occursin("Base/dict.jl", candidate) &&
                   !occursin("Base/hashing.jl", candidate)
                    parent = candidate
                    break
                end
            end
            callers[parent] = get(callers, parent, 0) + cnt
        end

        push!(stack, (indent, line))
    end

    entries = sort(collect(callers), by = x -> -x[2])
    println("Query caller summary (aggregated by immediate caller):")
    if isempty(entries)
        println("  (no query frames found)")
        return
    end
    for (caller, cnt) in entries[1:min(end, 10)]
        println("  $(lpad(cnt, 6))  $(caller)")
    end
end

function main()
    cfg = default_config()
    sim_time = parse(Float64, get(ENV, "SIM_TIME_S", "100.0"))
    window_size = parse(Float64, get(ENV, "WINDOW_SIZE_S", "10.0"))
    window_step = parse(Float64, get(ENV, "WINDOW_STEP_S", "1.0"))
    seed = parse(Int, get(ENV, "SEED", "42"))
    out_root = get(ENV, "OUT_ROOT", "./out_sliding")
    scenario_index = parse(Int, get(ENV, "SCENARIO_INDEX", "1"))
    do_warmup = parse_bool(get(ENV, "PROFILE_WARMUP", "true"))
    do_write = parse_bool(get(ENV, "PROFILE_WRITE", "false"))

    scenario_index = clamp(scenario_index, 1, length(cfg.scenarios))
    scenario = cfg.scenarios[scenario_index]

    println("Profiling scenario $(scenario_index): $(scenario.name)")
    println("  sim_time=$(sim_time)s, window_size=$(window_size)s, window_step=$(window_step)s, seed=$(seed)")
    println("  workers=$(nworkers()), warmup=$(do_warmup), write=$(do_write)")

    if do_warmup
        println("Warming up...")
        _ = run_one_scenario(scenario, cfg; sim_time, window_size, window_step, seed)
    end

    reset_event_counts!()
    Profile.clear()
    println("Running @profile ...")
    @profile begin
        _ = run_one_scenario(scenario, cfg; sim_time, window_size, window_step, seed)
    end

    println("Profile summary:")
    Profile.print(format=:tree, sortedby=:count, maxdepth=15)
    query_caller_summary()
    print_event_counts()

    if do_write
        mkpath(out_root)
        scenario_dir = joinpath(out_root, "profile_" * scenario.name)
        mkpath(scenario_dir)
        df = run_one_scenario(scenario, cfg; sim_time, window_size, window_step, seed)
        CSV.write(joinpath(scenario_dir, "sliding_window.csv"), df)
        println("Wrote results to $(scenario_dir)")
    end

    if !isempty(_added_procs)
        rmprocs(_added_procs)
    end
end

main()
