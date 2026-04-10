"""
Verify that stale EntanglementHistory cleanup works.
Run with 3 repeaters, high capacity, and check that stale_EH stays at 0.
"""

using BangBangSimulation
using QuantumSavory
using QuantumSavory.ProtocolZoo: EntanglementCounterpart, EntanglementHistory
using Random
import ConcurrentSim
using ConcurrentSim: now

function count_stale_eh(net, node)
    reg = net[node]
    n = nsubsystems(reg)
    count = 0
    for i in 1:n
        slot = reg[i]
        ec = query(slot, EntanglementCounterpart, ❓, ❓)
        if !isnothing(ec)
            hists = queryall(slot, EntanglementHistory, ❓, ❓, ❓, ❓, ❓)
            count += length(hists)
        end
    end
    return count
end

function run_diagnostic()
    Random.seed!(42)
    tmpdir = mktempdir()

    sim, net, consumer, schedulers = setup(
        3, 10, 135.7;
        slack=0.0, coherencetime=nothing, cutofftime=nothing,
        policy="OQF", outfolder=tmpdir, usetempfile=true,
    )

    dt = 0.001
    t = 0.0
    max_t = 0.5
    max_stale = 0

    while t < max_t
        ConcurrentSim.run(sim, t + dt)
        t_now = now(sim)
        for node in 2:4
            s = count_stale_eh(net, node)
            if s > max_stale
                max_stale = s
                println("t=$(round(t_now, digits=4)) node $node: stale_EH=$s")
            end
        end
        t = t_now
    end

    println("\n=== Result ===")
    println("Max stale EH observed: $max_stale")
    println("Consumer entries: $(length(consumer._log))")
end

run_diagnostic()
