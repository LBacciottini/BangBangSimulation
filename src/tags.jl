"""
Tag types used across the BangBang multi-flow simulator. All public
entanglement tags carry a `flow_id`, so single-flow runs are just a
degenerate multi-flow case with `flow_id == 1`.

Swap-generated update messages are flow-aware so the tracker can
reject stale messages that belong to a different flow after a slot is
reused. Delete messages remain separate because they are emitted by
cutoff logic.
"""

"""
    PrivateEntanglementCounterpart(remote_node, remote_slot)

Short-lived tag created by `EntanglerProt` the instant a Bell pair is
born. It carries **no** `flow_id`: assignment to a flow is the link
controller's job and happens at promotion time (single atomic step,
see `BangBangLinkController`). Promotion removes this tag and replaces
it with a `FlowEntanglementCounterpart`.
"""
@kwdef struct PrivateEntanglementCounterpart
    "remote node of the freshly generated pair"
    remote_node::Int
    "remote slot index of the freshly generated pair"
    remote_slot::Int
end
Base.show(io::IO, tag::PrivateEntanglementCounterpart) =
    print(io, "(Private) Entangled to $(tag.remote_node).$(tag.remote_slot)")
Tag(tag::PrivateEntanglementCounterpart) =
    Tag(PrivateEntanglementCounterpart, tag.remote_node, tag.remote_slot)


"""
    FlowEntanglementCounterpart(flow_id, remote_node, remote_slot)

Public counterpart of `EntanglementCounterpart` augmented with a flow
identifier. This is the tag that swappers, consumers, and the link
controller query on. A slot that holds this tag is entangled with
`remote_node.remote_slot` and the pair belongs to flow `flow_id`.
"""
@kwdef struct FlowEntanglementCounterpart
    flow_id::Int
    remote_node::Int
    remote_slot::Int
end
Base.show(io::IO, t::FlowEntanglementCounterpart) =
    print(io, "Flow $(t.flow_id): entangled to $(t.remote_node).$(t.remote_slot)")
Tag(t::FlowEntanglementCounterpart) =
    Tag(FlowEntanglementCounterpart, t.flow_id, t.remote_node, t.remote_slot)


"""
    FlowEntanglementHistory(flow_id, past_remote_node, past_remote_slot,
                            swap_node, swap_slot, local_swap_slot)

Swap-history record replacing `EntanglementHistory` with a leading
flow identifier. Used by the tracker to forward `EntanglementUpdate*`
and `EntanglementDelete` messages along the swap chain while
preserving flow identity.
"""
@kwdef struct FlowEntanglementHistory
    flow_id::Int
    past_remote_node::Int
    past_remote_slot::Int
    swap_node::Int
    swap_slot::Int
    local_swap_slot::Int
end
Base.show(io::IO, t::FlowEntanglementHistory) = print(io,
    "Flow $(t.flow_id): was entangled to $(t.past_remote_node).$(t.past_remote_slot), " *
    "swapped with $(t.swap_node).$(t.swap_slot) at local slot .$(t.local_swap_slot)")
Tag(t::FlowEntanglementHistory) = Tag(
    FlowEntanglementHistory,
    t.flow_id, t.past_remote_node, t.past_remote_slot,
    t.swap_node, t.swap_slot, t.local_swap_slot,
)


"""
    FlowEntanglementUpdateX(flow_id, past_local_node, past_local_slot,
                            past_remote_slot, new_remote_node,
                            new_remote_slot, correction)

Flow-aware version of QuantumSavory's `EntanglementUpdateX`. The extra
`flow_id` lets the tracker reject stale updates that would otherwise
match a newly reused slot belonging to a different flow.
"""
@kwdef struct FlowEntanglementUpdateX
    flow_id::Int
    past_local_node::Int
    past_local_slot::Int
    past_remote_slot::Int
    new_remote_node::Int
    new_remote_slot::Int
    correction::Int

    function FlowEntanglementUpdateX(
        flow_id::Int,
        past_local_node::Int,
        past_local_slot::Int,
        past_remote_slot::Int,
        new_remote_node::Int,
        new_remote_slot::Int,
        correction::Int,
    )
        new(flow_id, past_local_node, past_local_slot, past_remote_slot, new_remote_node, new_remote_slot, correction)
    end

    function FlowEntanglementUpdateX(
        flow_id::Int,
        past_local_node::Int,
        past_local_slot::Int,
        past_remote_slot::Int,
        new_remote_node::Int,
        packed_tail::Int,
    )
        new_remote_slot, correction = unpack_flow_update_tail(packed_tail)
        new(flow_id, past_local_node, past_local_slot, past_remote_slot, new_remote_node, new_remote_slot, correction)
    end
end
Base.show(io::IO, tag::FlowEntanglementUpdateX) =
    print(io, "Flow $(tag.flow_id): update slot .$(tag.past_remote_slot) which used to be entangled to " *
              "$(tag.past_local_node).$(tag.past_local_slot) to be entangled to " *
              "$(tag.new_remote_node).$(tag.new_remote_slot) and apply correction Z$(tag.correction)")
Tag(tag::FlowEntanglementUpdateX) = Tag(
    FlowEntanglementUpdateX,
    tag.flow_id,
    tag.past_local_node,
    tag.past_local_slot,
    tag.past_remote_slot,
    tag.new_remote_node,
    pack_flow_update_tail(tag.new_remote_slot, tag.correction),
)

"""
    FlowEntanglementUpdateZ(flow_id, past_local_node, past_local_slot,
                            past_remote_slot, new_remote_node,
                            new_remote_slot, correction)

Flow-aware version of QuantumSavory's `EntanglementUpdateZ`.
"""
@kwdef struct FlowEntanglementUpdateZ
    flow_id::Int
    past_local_node::Int
    past_local_slot::Int
    past_remote_slot::Int
    new_remote_node::Int
    new_remote_slot::Int
    correction::Int

    function FlowEntanglementUpdateZ(
        flow_id::Int,
        past_local_node::Int,
        past_local_slot::Int,
        past_remote_slot::Int,
        new_remote_node::Int,
        new_remote_slot::Int,
        correction::Int,
    )
        new(flow_id, past_local_node, past_local_slot, past_remote_slot, new_remote_node, new_remote_slot, correction)
    end

    function FlowEntanglementUpdateZ(
        flow_id::Int,
        past_local_node::Int,
        past_local_slot::Int,
        past_remote_slot::Int,
        new_remote_node::Int,
        packed_tail::Int,
    )
        new_remote_slot, correction = unpack_flow_update_tail(packed_tail)
        new(flow_id, past_local_node, past_local_slot, past_remote_slot, new_remote_node, new_remote_slot, correction)
    end
end
Base.show(io::IO, tag::FlowEntanglementUpdateZ) =
    print(io, "Flow $(tag.flow_id): update slot .$(tag.past_remote_slot) which used to be entangled to " *
              "$(tag.past_local_node).$(tag.past_local_slot) to be entangled to " *
              "$(tag.new_remote_node).$(tag.new_remote_slot) and apply correction X$(tag.correction)")
Tag(tag::FlowEntanglementUpdateZ) = Tag(
    FlowEntanglementUpdateZ,
    tag.flow_id,
    tag.past_local_node,
    tag.past_local_slot,
    tag.past_remote_slot,
    tag.new_remote_node,
    pack_flow_update_tail(tag.new_remote_slot, tag.correction),
)

pack_flow_update_tail(new_remote_slot::Int, correction::Int) = (new_remote_slot << 2) | (correction & 0x03)
unpack_flow_update_tail(packed::Int) = (packed >> 2, packed & 0x03)
