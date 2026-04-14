"""
    CustomEntanglementTracker(sim, net, node)

Per-node process that listens for `EntanglementUpdateX`,
`EntanglementUpdateZ`, and `EntanglementDelete` messages on the node's
message buffer and maintains the flow-aware tag state accordingly:

  * If the affected local slot still carries a
    `FlowEntanglementCounterpart`, apply the Pauli correction and
    update the tag to point at the new remote (or delete the qubit).
  * Otherwise, if the slot carries a `FlowEntanglementHistory` (the
    qubit was already swapped away), forward the message to the
    remote node recorded in the history.
  * Otherwise, if the slot already carries an `EntanglementDelete`
    tag (a cutoff/swap race), update or drop the message as needed.

All of the above preserve the `flow_id` of the affected pair. Swap
updates carry `flow_id` explicitly so the tracker can reject stale
messages after cross-flow slot reuse; delete breadcrumbs still use the
plain QuantumSavory delete tag.
"""
@kwdef struct CustomEntanglementTracker <: QuantumSavory.ProtocolZoo.AbstractProtocol
    "time-and-schedule-tracking instance from ConcurrentSim"
    sim::Simulation
    "register network"
    net::RegisterNet
    "the vertex of the node where this tracker runs"
    node::Int
end

CustomEntanglementTracker(net::RegisterNet, node::Int) =
    CustomEntanglementTracker(get_time_tracker(net), net, node)

# Updating the mirror history on the second measured swap slot looks
# symmetric, but it corrupts the forwarding chain in deep repeater
# paths. The original slot's history is the only breadcrumb needed for
# future updates arriving along that branch. Keep the env switch only
# as a debugging escape hatch.
const BB_UPDATE_MIRROR_HISTORY = get(ENV, "BB_UPDATE_MIRROR_HISTORY", "0") != "0"

@resumable function (prot::CustomEntanglementTracker)()
    nodereg = prot.net[prot.node]
    mb = messagebuffer(prot.net, prot.node)
    while true
        workwasdone = true
        while workwasdone
            workwasdone = false
            for (updatetagsymbol, updategate) in ((FlowEntanglementUpdateX, Z),
                                                  (FlowEntanglementUpdateZ, X),
                                                  (QuantumSavory.ProtocolZoo.EntanglementDelete, nothing))
                if !isnothing(updategate) # EntanglementUpdate{X,Z}
                    msg = querydelete!(mb, updatetagsymbol, ❓, ❓, ❓, ❓, ❓, ❓)
                    isnothing(msg) && continue
                    (src, (_, flow_id, pastremotenode, pastremoteslotid, localslotid, newremotenode, packed_tail)) = msg
                    newremoteslotid, correction = unpack_flow_update_tail(packed_tail)
                else # EntanglementDelete
                    msg = querydelete!(mb, updatetagsymbol, ❓, ❓, ❓, ❓)
                    isnothing(msg) && continue
                    (src, (_, pastremotenode, pastremoteslotid, _, localslotid)) = msg
                end

                @debug "CustomEntanglementTracker @$(prot.node): got $(msg.tag) from $(msg.src) at $(now(prot.sim))"
                workwasdone = true
                localslot = nodereg[localslotid]
                match_flow_id = isnothing(updategate) ? ❓ : flow_id

                # Case 1: the slot still holds the pair we're being notified about.
                @yield lock(localslot)
                counterpart = querydelete!(localslot, FlowEntanglementCounterpart,
                                           match_flow_id, pastremotenode, pastremoteslotid)
                unlock(localslot)
                if !isnothing(counterpart)
                    @yield lock(localslot)
                    if !isassigned(localslot)
                        unlock(localslot)
                        error("CustomEntanglementTracker @$(prot.node): the local slot lost its " *
                              "quantum state although it still carried a FlowEntanglementCounterpart tag.")
                    end
                    if !isnothing(updategate) # Pauli frame correction
                        correction == 2 && apply!(localslot, updategate)
                        if newremotenode != -1
                            tag!(localslot, FlowEntanglementCounterpart,
                                 flow_id, newremotenode, newremoteslotid)
                        end
                    else # EntanglementDelete — throw out the qubit
                        traceout!(localslot)
                    end
                    unlock(localslot)
                    continue
                end

                # Case 2: the slot was already swapped away. Forward to the swap target.
                history = querydelete!(localslot, FlowEntanglementHistory,
                                       match_flow_id,
                                       pastremotenode, pastremoteslotid,  # who we were entangled to
                                       ❓, ❓,                       # who we swapped with
                                       ❓)                           # local slot used in the swap
                if !isnothing(history)
                    _, _, _, _, whoweswappedwith_node, whoweswappedwith_slotidx, swappedlocal_slotidx = history.tag
                    if !isnothing(updategate) # EntanglementUpdate{X,Z}
                        # refresh the history tag with the new remote (after the update propagates)
                        tag!(localslot, FlowEntanglementHistory,
                             flow_id, newremotenode, newremoteslotid,
                             whoweswappedwith_node, whoweswappedwith_slotidx, swappedlocal_slotidx)
                        msghist = Tag(updatetagsymbol(flow_id, pastremotenode, pastremoteslotid,
                                      whoweswappedwith_slotidx, newremotenode, newremoteslotid, correction))
                        put!(channel(prot.net, prot.node => whoweswappedwith_node; permit_forward=true), msghist)

                        if BB_UPDATE_MIRROR_HISTORY
                            # Also refresh the mirror history on the second slot that participated in the swap.
                            second_localslot = nodereg[swappedlocal_slotidx]
                            history2 = querydelete!(second_localslot, FlowEntanglementHistory,
                                                    flow_id,
                                                    whoweswappedwith_node, whoweswappedwith_slotidx,
                                                    pastremotenode, pastremoteslotid,
                                                    localslot.idx)
                            @assert !isnothing(history2)
                            tag!(second_localslot, FlowEntanglementHistory,
                                 flow_id, whoweswappedwith_node, whoweswappedwith_slotidx,
                                 newremotenode, newremoteslotid, localslot.idx)
                        end
                    else # EntanglementDelete — forward and leave a breadcrumb
                        msghist = Tag(updatetagsymbol, pastremotenode, pastremoteslotid,
                                      whoweswappedwith_node, whoweswappedwith_slotidx)
                        tag!(localslot, updatetagsymbol, prot.node, localslot.idx,
                             whoweswappedwith_node, whoweswappedwith_slotidx)
                        put!(channel(prot.net, prot.node => whoweswappedwith_node; permit_forward=true), msghist)
                    end
                    continue
                end

                # Case 3: the slot already carries an EntanglementDelete breadcrumb
                # from a delete/update race. Resolve and keep going.
                if !isnothing(querydelete!(localslot, QuantumSavory.ProtocolZoo.EntanglementDelete,
                                           prot.node, localslot.idx, pastremotenode, pastremoteslotid))
                    if !isnothing(updategate)
                        tag!(localslot, QuantumSavory.ProtocolZoo.EntanglementDelete,
                             prot.node, localslot.idx, newremotenode, newremoteslotid)
                    end
                    continue
                end

                @warn "CustomEntanglementTracker @$(prot.node) received an unhandleable message $(msg) " *
                      "(no matching FlowEntanglementCounterpart / FlowEntanglementHistory / EntanglementDelete tag). " *
                      "This may happen transiently if CutoffProt fires during a swap."
            end
        end
        @yield onchange(mb)
    end
end

get_tracker(sim, net, node) = CustomEntanglementTracker(sim, net, node)
