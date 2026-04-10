#!/usr/bin/env julia
#
# Diagnostic comparison of OQF vs YQF swapping policies.
# Phase A: no decoherence (expect identical results)
# Phase B: with decoherence (expect same throughput, different fidelity)

using Pkg; Pkg.activate(".")
using BangBangSimulation
using Random
using Statistics
using DataFrames
using Printf
import ConcurrentSim
using ConcurrentSim: Simulation, now
using QuantumSavory
using QuantumSavory.ProtocolZoo: EntanglementCounterpart, EntanglementHistory

# ── Snapshot: sample register state at a point in time ───────────────────

struct NodeSnapshot
    time::Float64
    node::Int
    occupied::Int
    total_slots::Int
    entangled_left::Int
    entangled_right::Int
    mean_age::Float64
    max_age::Float64
end

function snapshot_node(net, node::Int, t::Float64)
    reg = net[node]
    n = nsubsystems(reg)

    occupied = sum(isassigned(reg, i) for i in 1:n)

    left  = length(queryall(reg, EntanglementCounterpart, <(node), ❓; assigned=true))
    right = length(queryall(reg, EntanglementCounterpart, >(node), ❓; assigned=true))

    ages = Float64[]
    for i in 1:n
        if isassigned(reg, i)
            push!(ages, t - reg.accesstimes[i])
        end
    end
    mean_a = isempty(ages) ? 0.0 : mean(ages)
    max_a  = isempty(ages) ? 0.0 : maximum(ages)

    return NodeSnapshot(t, node, occupied, n, left, right, mean_a, max_a)
end

function snapshot_all(net, total_nodes::Int, t::Float64)
    return [snapshot_node(net, node, t) for node in 1:total_nodes]
end

# ── Run one simulation, sampling state periodically ──────────────────────

function run_diagnostic(;
    policy::String,
    nrepeaters::Int=3,
    nslots::Int=100,
    linkcapacity::Float64,
    linklength_km::Float64=20.0,
    coherencetime::Union{Float64,Nothing}=nothing,
    sim_time::Float64=50.0,
    seed::Int=42,
    sample_interval::Float64=1.0,
)
    Random.seed!(seed)
    tmpdir = mktempdir()

    sim, net, consumer, schedulers = setup(
        nrepeaters, nslots, linkcapacity;
        linklength=linklength_km,
        slack=0.0,
        coherencetime=coherencetime,
        cutofftime=nothing,
        policy=policy,
        outfolder=tmpdir,
        usetempfile=true,
    )

    total_nodes = nrepeaters + 2
    snapshots = Vector{NodeSnapshot}()

    # Run simulation in chunks, sampling between each
    t = sample_interval
    while t <= sim_time
        ConcurrentSim.run(sim, t)
        append!(snapshots, snapshot_all(net, total_nodes, now(sim)))
        t += sample_interval
    end
    # Run to exact end if needed
    if now(sim) < sim_time
        ConcurrentSim.run(sim, sim_time)
    end

    # Flush scheduler logs
    for s in schedulers
        dump_log(s)
    end

    return (; consumer, snapshots, schedulers, net, nrepeaters)
end

# ── Analysis helpers ─────────────────────────────────────────────────────

function compute_metrics(result; steady_state_start::Float64=10.0)
    clog = result.consumer._log
    steady = filter(e -> e.t > steady_state_start, clog)

    if length(steady) < 2
        fidelity = NaN
        throughput = NaN
    else
        obs1_vals = [e.obs1 for e in steady]
        fidelity_raw = mean(obs1_vals)
        fidelity = (3 * fidelity_raw + 1) / 4
        throughput = length(steady) / (maximum(e.t for e in steady) - minimum(e.t for e in steady))
    end

    return (; fidelity, throughput, total_pairs=length(clog))
end

function occupancy_stats(snapshots::Vector{NodeSnapshot}, node::Int; after::Float64=10.0)
    log = filter(r -> r.node == node && r.time > after, snapshots)
    isempty(log) && return (mean_occ=NaN, peak_occ=NaN, mean_age=NaN, peak_age=NaN,
                            mean_left=NaN, mean_right=NaN)
    return (
        mean_occ  = mean(r.occupied for r in log),
        peak_occ  = maximum(r.occupied for r in log),
        mean_age  = mean(r.mean_age for r in log),
        peak_age  = maximum(r.max_age for r in log),
        mean_left = mean(r.entangled_left for r in log),
        mean_right= mean(r.entangled_right for r in log),
    )
end

# ── Pretty printing ──────────────────────────────────────────────────────

function print_comparison(label::String, res_oqf, res_yqf, nrepeaters::Int)
    m_oqf = compute_metrics(res_oqf)
    m_yqf = compute_metrics(res_yqf)

    println("\n", "="^70)
    println("  $label")
    println("="^70)

    println("\n── End-to-end metrics ──")
    println("                    OQF           YQF           Ratio(YQF/OQF)")
    thr_ratio = isnan(m_oqf.throughput) || m_oqf.throughput == 0 ? NaN : m_yqf.throughput / m_oqf.throughput
    fid_ratio = isnan(m_oqf.fidelity) || m_oqf.fidelity == 0 ? NaN : m_yqf.fidelity / m_oqf.fidelity
    @printf("  Throughput:   %12.4f  %12.4f  %12.4f\n", m_oqf.throughput, m_yqf.throughput, thr_ratio)
    @printf("  Fidelity:    %12.4f  %12.4f  %12.4f\n", m_oqf.fidelity, m_yqf.fidelity, fid_ratio)
    @printf("  Total pairs: %12d  %12d\n", m_oqf.total_pairs, m_yqf.total_pairs)

    println("\n── Per-node occupancy (steady state, t > 10) ──")
    println("  Node  | OQF mean_occ  peak_occ  mean_age  peak_age | YQF mean_occ  peak_occ  mean_age  peak_age")
    println("  ", "-"^100)
    total_nodes = nrepeaters + 2
    for node in 1:total_nodes
        so = occupancy_stats(res_oqf.snapshots, node)
        sy = occupancy_stats(res_yqf.snapshots, node)
        @printf("  %4d  | %8.1f  %8d  %8.3f  %8.3f | %8.1f  %8d  %8.3f  %8.3f\n",
                node, so.mean_occ, so.peak_occ, so.mean_age, so.peak_age,
                      sy.mean_occ, sy.peak_occ, sy.mean_age, sy.peak_age)
    end

    println("\n── Per-node entanglement breakdown (steady state means) ──")
    println("  Node  | OQF  left  right | YQF  left  right")
    println("  ", "-"^50)
    for node in 1:total_nodes
        so = occupancy_stats(res_oqf.snapshots, node)
        sy = occupancy_stats(res_yqf.snapshots, node)
        @printf("  %4d  | %7.1f  %7.1f | %7.1f  %7.1f\n",
                node, so.mean_left, so.mean_right,
                      sy.mean_left, sy.mean_right)
    end
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    bk = bk_link_capacity(linklength_km=20.0, excitation_time_s=17e-6, static_eff=0.28)
    linkcapacity = bk.rate
    nrepeaters = 3
    nslots = 100
    sim_time = 50.0
    seed = 42

    common = (nrepeaters=nrepeaters, nslots=nslots, linkcapacity=linkcapacity,
              linklength_km=20.0, sim_time=sim_time, seed=seed)

    # ── Phase A: No decoherence ──
    println("Running Phase A: No decoherence...")
    println("  OQF...")
    res_a_oqf = run_diagnostic(; policy="OQF", coherencetime=nothing, common...)
    println("  YQF...")
    res_a_yqf = run_diagnostic(; policy="YQF", coherencetime=nothing, common...)
    print_comparison("PHASE A: No decoherence (should be IDENTICAL)", res_a_oqf, res_a_yqf, nrepeaters)

    # ── Phase B: With decoherence ──
    println("\nRunning Phase B: With decoherence (coherencetime=2.0)...")
    println("  OQF...")
    res_b_oqf = run_diagnostic(; policy="OQF", coherencetime=2.0, common...)
    println("  YQF...")
    res_b_yqf = run_diagnostic(; policy="YQF", coherencetime=2.0, common...)
    print_comparison("PHASE B: With decoherence (throughput should be similar)", res_b_oqf, res_b_yqf, nrepeaters)

    # ── Phase A validation ──
    m_a_oqf = compute_metrics(res_a_oqf)
    m_a_yqf = compute_metrics(res_a_yqf)
    println("\n", "="^70)
    println("  VERDICT")
    println("="^70)
    if !isnan(m_a_oqf.throughput) && !isnan(m_a_yqf.throughput)
        thr_ratio = m_a_yqf.throughput / m_a_oqf.throughput
        fid_ratio = m_a_yqf.fidelity / m_a_oqf.fidelity
        if abs(thr_ratio - 1.0) > 0.15
            println("  WARNING: Phase A throughput differs by $(round((thr_ratio-1)*100, digits=1))% -- possible bug!")
        else
            println("  Phase A throughput: OK (ratio = $(round(thr_ratio, digits=3)))")
        end
        if abs(fid_ratio - 1.0) > 0.05
            println("  WARNING: Phase A fidelity differs by $(round((fid_ratio-1)*100, digits=1))% -- possible bug!")
        else
            println("  Phase A fidelity: OK (ratio = $(round(fid_ratio, digits=3)))")
        end
    else
        println("  WARNING: Not enough data in Phase A to compare.")
    end
end

main()
