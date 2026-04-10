using BangBangSimulation
using DataFrames
using Random
import ConcurrentSim
using ConcurrentSim: now
using QuantumSavory
using QuantumSavory.ProtocolZoo: EntanglementCounterpart, EntanglementHistory
import QuantumSavory: isolderthan

cfg = default_config()
s2 = cfg.scenarios[2]

println("=== S2 Deadlock Diagnostic ===")
println("nrepeaters=$(cfg.nrepeaters), nslots=$(s2.nslots), slack=$(s2.slack), policy=$(s2.policy), cutofftime=$(s2.cutofftime)")
println()

Random.seed!(42)
tmpdir = mktempdir()
sim, net, consumer, schedulers = setup(
    cfg.nrepeaters, s2.nslots, cfg.linkcapacity;
    linklength=cfg.linklength_km, slack=s2.slack,
    coherencetime=cfg.coherencetime, cutofftime=s2.cutofftime,
    policy=s2.policy, outfolder=tmpdir, usetempfile=true,
)

total_nodes = cfg.nrepeaters + 2

function dump_node_state(net, node, total_nodes)
    reg = net[node]
    n = nsubsystems(reg)
    assigned_count = sum(isassigned(reg[i]) for i in 1:n)
    locked_count = sum(islocked(reg[i]) for i in 1:n)

    ec_left = queryall(reg, EntanglementCounterpart, <(node), ❓)
    ec_right = queryall(reg, EntanglementCounterpart, >(node), ❓)

    # Count EntanglementHistory tags (slots used by past swaps)
    eh_count = 0
    for i in 1:n
        slot = reg[i]
        # Query for any EntanglementHistory tag
        h = query(slot, EntanglementHistory, ❓, ❓, ❓, ❓, ❓)
        if !isnothing(h)
            eh_count += 1
        end
    end

    # Count PrivateEntanglementCounterpart tags
    pec_count = 0
    for i in 1:n
        slot = reg[i]
        p = query(slot, BangBangSimulation.PrivateEntanglementCounterpart, ❓, ❓)
        if !isnothing(p)
            pec_count += 1
        end
    end

    # Count assigned but with no known tag (orphaned)
    tagged_slots = Set{Int}()
    for r in ec_left
        push!(tagged_slots, r.slot.idx)
    end
    for r in ec_right
        push!(tagged_slots, r.slot.idx)
    end
    # Also count EH-tagged slots
    for i in 1:n
        h = query(reg[i], EntanglementHistory, ❓, ❓, ❓, ❓, ❓)
        if !isnothing(h)
            push!(tagged_slots, i)
        end
        p = query(reg[i], BangBangSimulation.PrivateEntanglementCounterpart, ❓, ❓)
        if !isnothing(p)
            push!(tagged_slots, i)
        end
    end
    orphaned = sum(isassigned(reg[i]) && !(i in tagged_slots) for i in 1:n)

    return (
        node=node, n=n,
        assigned=assigned_count, free=n - assigned_count, locked=locked_count,
        ec_left=length(ec_left), ec_right=length(ec_right),
        eh=eh_count, pec=pec_count, orphaned=orphaned
    )
end

function print_state_table(net, total_nodes, t)
    println("--- t=$(round(t, digits=2)) ---")
    println("Node | Slots | Assigned | Free | Locked | EC_Left | EC_Right | EH | PEC | Orphaned")
    println("-----|-------|----------|------|--------|---------|----------|----|-----|--------")
    for node in 1:total_nodes
        s = dump_node_state(net, node, total_nodes)
        println("  $(s.node)  |  $(s.n)  |    $(lpad(s.assigned,2))    |  $(lpad(s.free,2))  |   $(lpad(s.locked,2))   |   $(lpad(s.ec_left,2))    |    $(lpad(s.ec_right,2))    | $(lpad(s.eh,2)) |  $(lpad(s.pec,2)) |    $(lpad(s.orphaned,2))")
    end
    println()
end

# Run simulation in incremental steps, monitoring for stall
let step_size = 0.5,
    max_time = 50.0,
    last_consumer_count = 0,
    stall_start = nothing,
    stall_threshold = 5.0,
    deadlocked = false,
    t = 0.0

    while t < max_time
        t += step_size
        ConcurrentSim.run(sim, t)
        current_t = now(sim)

        consumer_count = length(consumer._log)

        if consumer_count > last_consumer_count
            stall_start = nothing
            last_consumer_count = consumer_count
        else
            if stall_start === nothing
                stall_start = current_t
            elseif current_t - stall_start >= stall_threshold
                # Check if sim is truly stuck by trying to advance 0.001 more
            prev_t = now(sim)
            ConcurrentSim.run(sim, current_t + 0.001)
            post_t = now(sim)
            println("DEADLOCK DETECTED at t=$(round(current_t, digits=2)) (no consumer progress since t=$(round(stall_start, digits=2)))")
            println("Sim time before micro-advance: $prev_t, after: $post_t (same=$(prev_t==post_t))")
            println("Consumer log entries: $consumer_count")
            println()
            print_state_table(net, total_nodes, current_t)

                # Detailed per-slot dump for repeaters
                for node in 2:(cfg.nrepeaters + 1)
                    reg = net[node]
                    n = nsubsystems(reg)
                    println("=== Detailed Node $node ($n slots) ===")
                    for i in 1:n
                        slot = reg[i]
                        a = isassigned(slot)
                        l = islocked(slot)
                        ec = query(slot, EntanglementCounterpart, ❓, ❓)
                        eh = query(slot, EntanglementHistory, ❓, ❓, ❓, ❓, ❓)
                        pec = query(slot, BangBangSimulation.PrivateEntanglementCounterpart, ❓, ❓)

                        tags_str = ""
                        if !isnothing(ec)
                            dir = ec.tag[2] < node ? "LEFT" : "RIGHT"
                            tags_str *= "EC($(ec.tag[2]),$(ec.tag[3]))=$dir "
                        end
                        if !isnothing(eh)
                            tags_str *= "EH($(eh.tag[2]),$(eh.tag[3])→$(eh.tag[4]),$(eh.tag[5])) "
                        end
                        if !isnothing(pec)
                            tags_str *= "PEC($(pec.tag[2]),$(pec.tag[3])) "
                        end
                        if tags_str == "" && a
                            tags_str = "ASSIGNED_NO_TAG"
                        elseif !a
                            tags_str = "free"
                        end
                        println("  slot $i: assigned=$a locked=$l | $tags_str")
                    end
                    println()
                end

                # Check what the swapper at each repeater sees
                println("=== Swapper Diagnostics ===")
                for node in 2:(cfg.nrepeaters + 1)
                    reg = net[node]
                    low_qr = queryall(reg, EntanglementCounterpart, <(node), ❓; locked=false, assigned=true)
                    high_qr = queryall(reg, EntanglementCounterpart, >(node), ❓; locked=false, assigned=true)
                    println("Swapper @node $node: LEFT=$(length(low_qr)) RIGHT=$(length(high_qr)) → $(isempty(low_qr) || isempty(high_qr) ? "CANNOT swap" : "CAN swap")")
                end
                println()

                # Check entangler status: which links have free slots?
                println("=== Link Free Slot Check ===")
                for link_nodeA in 1:(cfg.nrepeaters + 1)
                    nodeA = link_nodeA
                    nodeB = link_nodeA + 1
                    nA = nsubsystems(net[nodeA])
                    nB = nsubsystems(net[nodeB])
                    chooseslotA = (nodeA == 1 || nodeA == total_nodes) ? QuantumSavory.alwaystrue : (i -> i <= nA - 1)
                    chooseslotB = (nodeB == 1 || nodeB == total_nodes) ? QuantumSavory.alwaystrue : (i -> i >= 2)
                    freeA = [i for i in 1:nA if !isassigned(net[nodeA][i]) && !islocked(net[nodeA][i]) && chooseslotA(i)]
                    freeB = [i for i in 1:nB if !isassigned(net[nodeB][i]) && !islocked(net[nodeB][i]) && chooseslotB(i)]
                    println("Link($nodeA,$nodeB): freeA=$freeA freeB=$freeB → $(isempty(freeA) || isempty(freeB) ? "BLOCKED" : "can generate")")
                end
                println()

                deadlocked = true
                break
            end
        end

        # Print periodic summary every 5s
        if mod(t, 5.0) < step_size
            print_state_table(net, total_nodes, current_t)
            println("Consumer entries: $consumer_count (last_count=$last_consumer_count)")
            println()
        end
    end

    if !deadlocked
        println("Simulation completed without deadlock (t=$max_time)")
        println("Consumer log entries: $(length(consumer._log))")
        print_state_table(net, total_nodes, now(sim))
    end
end
