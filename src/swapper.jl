"""
Flow-aware swap protocol. One logical swap process runs per
`(repeater, flow)` pair and only considers qubits tagged for its own
flow. This keeps flows strictly isolated at the swapping layer, which
also lets a partitioned-slot-pool baseline reuse the exact same code
path.
"""

random_index(arr) = rand(keys(arr))

function slot_within_agelimit(qr, sim, agelimit)
    isnothing(agelimit) && return true
    isassigned(qr.slot) || return false
    tag_info = get(qr.slot.reg.tag_info, qr.id, nothing)
    tag_info === nothing && return false
    tag_time = tag_info[3]
    return (now(sim) - tag_time) <= agelimit
end

"""
    findswapablequbits_for_flow(net, node, flow_id, pred_low, pred_high,
                                choose_low, choose_high, chooseslots;
                                agelimit=nothing, policy="OQF")

Find two qubits at `node` that are entangled to remote nodes satisfying
`pred_low` and `pred_high`, and that belong to flow `flow_id`. Returns
`nothing` if no compatible pair exists.

Policies: `"OQF"` (oldest qubit first), `"YQF"` (youngest first),
`"PYQF"` (priority YQF — furthest node first, break ties by youngest),
`"RAND"` (random among the slots passing `chooseslots`).
"""
function findswapablequbits_for_flow(net, node, flow_id,
                                     pred_low, pred_high,
                                     choose_low, choose_high, chooseslots_low, chooseslots_high;
                                     agelimit=nothing, policy="OQF", sim=get_time_tracker(net))
    reg = net[node]
    low_queryresults  = [
        n for n in queryall(reg, FlowEntanglementCounterpart, flow_id, pred_low,  ❓; locked=false, assigned=true)
        if slot_within_agelimit(n, sim, agelimit)
    ]
    high_queryresults = [
        n for n in queryall(reg, FlowEntanglementCounterpart, flow_id, pred_high, ❓; locked=false, assigned=true)
        if slot_within_agelimit(n, sim, agelimit)
    ]
    (isempty(low_queryresults) || isempty(high_queryresults)) && return nothing

    choosefunc_low = chooseslots_low isa Vector{Int} ? in(chooseslots_low) : chooseslots_low
    choosefunc_high = chooseslots_high isa Vector{Int} ? in(chooseslots_high) : chooseslots_high
    low_queryresults  = [qr for qr in low_queryresults if choosefunc_low(qr.slot.idx)]
    high_queryresults = [qr for qr in high_queryresults if choosefunc_high(qr.slot.idx)]
    (isempty(low_queryresults) || isempty(high_queryresults)) && return nothing

    # For OQF / YQF we use reg.accesstimes[slot.idx] as a proxy for the
    # local initialization time. This is only valid as long as Pauli
    # corrections are applied without an explicit time argument; see the
    # historical note in enhanced_swapping.jl (now removed).
    created_at(slot::RegRef) = slot.reg.accesstimes[slot.idx]

    # Tag layout for FlowEntanglementCounterpart: (type, flow_id, remote_node, remote_slot)
    # so tag[3] is remote_node.
    if policy == "RAND"
        il = choose_low((qr.tag[3] for qr in low_queryresults))
        ih = choose_high((qr.tag[3] for qr in high_queryresults))
        return (low_queryresults[il], high_queryresults[ih])
    elseif policy == "OQF"
        il = argmin([created_at(qr.slot) for qr in low_queryresults],  dims=1)[1]
        ih = argmin([created_at(qr.slot) for qr in high_queryresults], dims=1)[1]
        return (low_queryresults[il], high_queryresults[ih])
    elseif policy == "YQF"
        il = argmax([created_at(qr.slot) for qr in low_queryresults],  dims=1)[1]
        ih = argmax([created_at(qr.slot) for qr in high_queryresults], dims=1)[1]
        return (low_queryresults[il], high_queryresults[ih])
    elseif policy == "PYQF"
        il = argmin([(qr.tag[3], -created_at(qr.slot)) for qr in low_queryresults])
        ih = argmax([(qr.tag[3],  created_at(qr.slot)) for qr in high_queryresults])
        return (low_queryresults[il], high_queryresults[ih])
    end

    error("findswapablequbits_for_flow: unknown policy '$policy'")
end


"""
    FlowSwapperProt(sim, net, node, flow; kwargs...)

Entanglement swapping protocol for a single `flow` at a single repeater
`node`. Queries `FlowEntanglementCounterpart` tags belonging to
`flow.id` only. A `node` that is not strictly internal to the flow
(i.e. `node == flow.src` or `node == flow.dst`) does nothing — this is
so `setup_multiflow` can blindly spawn a swapper per `(node, flow)` and
let the no-op cases return instantly.
"""
@kwdef struct FlowSwapperProt <: QuantumSavory.ProtocolZoo.AbstractProtocol
    sim::Simulation
    net::RegisterNet
    "the repeater node where this swapper runs"
    node::Int
    "the flow this swapper operates on"
    flow::Flow
    "filter restricting which local slots this swapper may use for left-facing entanglement"
    chooseslotsL::Union{Vector{Int},Function} = idx->true
    "filter restricting which local slots this swapper may use for right-facing entanglement"
    chooseslotsH::Union{Vector{Int},Function} = idx->true
    "predicate selecting the 'low' (left) remote nodes — defaults to any node strictly to the left but still inside `flow`'s span"
    nodeL::Any = ❓
    "predicate for the 'high' (right) remote nodes"
    nodeH::Any = ❓
    "tie-breaker among 'low' candidates (only used by RAND policy)"
    chooseL::Function = random_index
    "tie-breaker among 'high' candidates (only used by RAND policy)"
    chooseH::Function = random_index
    "fixed busy time after a swap"
    local_busy_time::Float64 = 0.0
    "how long to wait before retrying when no swappable pair exists (`nothing` to wait on tag changes)"
    retry_lock_time::Union{Float64,Nothing} = 0.1
    "number of rounds (`-1` for infinite)"
    rounds::Int = -1
    "oldest a qubit may be to still be swappable (`nothing` for no limit)"
    agelimit::Union{Float64,Nothing} = nothing
    "swap policy: 'OQF', 'YQF', 'PYQF', 'RAND'"
    policy::String = "OQF"
end

FlowSwapperProt(sim::Simulation, net::RegisterNet, node::Int, flow::Flow; kwargs...) =
    FlowSwapperProt(; sim, net, node, flow, kwargs...)

FlowSwapperProt(net::RegisterNet, node::Int, flow::Flow; kwargs...) =
    FlowSwapperProt(get_time_tracker(net), net, node, flow; kwargs...)

@resumable function (prot::FlowSwapperProt)()
    # Nothing to do at the flow's endpoints.
    is_internal_node(prot.flow, prot.node) || return

    # Default low/high predicates restrict to the flow's span so swappers
    # for different flows never collide.
    flow, node = prot.flow, prot.node
    predL = prot.nodeL === ❓ ? (r -> flow.src <= r < node) : prot.nodeL
    predH = prot.nodeH === ❓ ? (r -> node < r <= flow.dst) : prot.nodeH

    rounds = prot.rounds
    round = 1
    while rounds != 0
        qubit_pair_ = findswapablequbits_for_flow(
            prot.net, prot.node, prot.flow.id,
            predL, predH,
            prot.chooseL, prot.chooseH, prot.chooseslotsL, prot.chooseslotsH;
            agelimit=prot.agelimit, policy=prot.policy, sim=prot.sim,
        )
        if isnothing(qubit_pair_)
            if isnothing(prot.retry_lock_time)
                @yield onchange(prot.net[prot.node], Tag)
            else
                @yield timeout(prot.sim, prot.retry_lock_time::Float64)
            end
            continue
        end
        qubit_pair = qubit_pair_::NTuple{2, Base.NamedTuple{(:slot, :id, :tag), Base.Tuple{RegRef, Int128, Tag}}}

        (q1, id1, tag1) = qubit_pair[1].slot, qubit_pair[1].id, qubit_pair[1].tag
        (q2, id2, tag2) = qubit_pair[2].slot, qubit_pair[2].id, qubit_pair[2].tag

        @yield lock(q1) & lock(q2)

        # tag1/tag2 have the layout (FlowEntanglementCounterpart, flow_id, remote_node, remote_slot)
        flow_id = tag1[2]
        remote1_node, remote1_slot = tag1[3], tag1[4]
        remote2_node, remote2_slot = tag2[3], tag2[4]

        untag!(q1, id1)
        tag!(q1, FlowEntanglementHistory,
             flow_id, remote1_node, remote1_slot, remote2_node, remote2_slot, q2.idx)

        untag!(q2, id2)
        tag!(q2, FlowEntanglementHistory,
             flow_id, remote2_node, remote2_slot, remote1_node, remote1_slot, q1.idx)

        uptotime!((q1, q2), now(prot.sim))
        swapcircuit = QuantumSavory.CircuitZoo.LocalEntanglementSwap()
        xmeas, zmeas = swapcircuit(q1, q2)

        # Carry flow_id in the update message so a stale message cannot be
        # applied to a newly reused slot belonging to a different flow.
        msg1 = Tag(FlowEntanglementUpdateX(flow_id, prot.node, q1.idx, remote1_slot,
                   remote2_node, remote2_slot, Int(xmeas)))
        put!(channel(prot.net, prot.node => remote1_node; permit_forward=true), msg1)

        msg2 = Tag(FlowEntanglementUpdateZ(flow_id, prot.node, q2.idx, remote2_slot,
                   remote1_node, remote1_slot, Int(zmeas)))
        put!(channel(prot.net, prot.node => remote2_node; permit_forward=true), msg2)

        @yield timeout(prot.sim, prot.local_busy_time)
        unlock(q1)
        unlock(q2)
        rounds == -1 || (rounds -= 1)
        round += 1
    end
end
