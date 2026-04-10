"""
This file should be ignored for now.
When complete, it will enable evaluation of quantum TCP against swap-asap with bang bang.
"""


@kwdef struct EndNodeRateController <: QuantumSavory.ProtocolZoo.AbstractProtocol
    "the simulation object"
    sim::Simulation
    "the register network"
    net::RegisterNet
    "the end node on which the rate controller is applied"
    node::Int
    "the rate at which the flows controlled should generate QuantumSavory.ProtocolZoo.QDatagrams"
    sending_rate::Float64
end
EndNodeRateController(net::RegisterNet, node::Int, sending_rate::Float64) = EndNodeRateController(get_time_tracker(net), net, node, sending_rate)

###
# Protocol Implementation
###

@resumable function (prot::EndNodeRateController)()
    (;sim, net, node, sending_rate) = prot
    mb = messagebuffer(net, node)

    current_flows = Set{Int}() # the uuids of flows currently being processed
    # these are keyed by uuid:
    qdatagrams_in_flight   = Dict{Int,Int}() # number of datagrams currently believed to be in flight # TODO turn value to ordered list
    qdatagrams_sent        = Dict{Int,Int}() # total number of datagrams already sent
    pairs_left_to_fulfill = Dict{Int,Int}() # total number of pairs still to be established
    destination            = Dict{Int,Int}() # the destination

    next_generation_time = now(sim)

    while true
        # check if there is an approved flow and start it
        flow = querydelete!(mb, QuantumSavory.ProtocolZoo.QTCP.Flow, node, ❓, ❓, ❓)
        if !isnothing(flow)
            _, _, dst, npairs, uuid = flow.tag
            push!(current_flows, uuid)
            qdatagrams_in_flight[uuid]   = 0
            qdatagrams_sent[uuid]        = 0
            pairs_left_to_fulfill[uuid] = npairs
            destination[uuid]            = dst
            @debug "[$(now(sim))]: flow $(uuid) started"
        end

        # check if there are datagram acknowledgements
        success = querydelete!(mb, QuantumSavory.ProtocolZoo.QTCP.QDatagramSuccess, ❓, ❓, ❓)
        if !isnothing(success)
            # TODO implement drop detection and window modification
            _, flow_uuid, seq_num, start_time = success.tag
            qdatagrams_in_flight[flow_uuid]   -= 1
            pairs_left_to_fulfill[flow_uuid] -= 1

            # Check if there are any LinkLevelReplyAtSource messages and turn them into QTCPPairBegin messages
            link_reply = querydelete!(mb, LinkLevelReplyAtSource, ❓, ❓, ❓)
            @assert !isnothing(link_reply) "No LinkLevelReplyAtSource message found after the success of flow $(flow_uuid), sequence $(seq_num) at node $(node)"
            _, flow_uuid, seq_num, memory_slot = link_reply.tag
            pair_begin = QTCPPairBegin(;
                flow_uuid,
                flow_src=node,
                flow_dst=success.src,
                seq_num,
                memory_slot,
                start_time
            )
            put!(net[node], pair_begin)
            @debug "[$(now(sim))]: datagram success notification from flow $(flow_uuid) pair $(seq_num) returned to start node"

            # if we have fulfilled all pairs, remove the flow in every data structure
            if pairs_left_to_fulfill[flow_uuid] == 0
                delete!(current_flows, flow_uuid)
                delete!(qdatagrams_in_flight, flow_uuid)
                delete!(qdatagrams_sent, flow_uuid)
                delete!(pairs_left_to_fulfill, flow_uuid)
                delete!(destination, flow_uuid)
                current_time = now(sim)
                @debug "[$(current_time)]: flow $(flow_uuid) completed and deallocated"
            end
        end

        # check if we just received a qdatagram for which we are the flow destination
        qdatagram = querydelete!(mb, QuantumSavory.ProtocolZoo.QDatagram, ❓, ❓, node, ❓, ❓, ❓)
        if !isnothing(qdatagram)
            # We need to generate a QuantumSavory.ProtocolZoo.QTCP.QDatagramSuccess message and send it back to the flow source
            _, flow_uuid, flow_src, flow_dst, corrections, seq_num, start_time = qdatagram.tag
            qdatagram_success = QuantumSavory.ProtocolZoo.QTCP.QDatagramSuccess(flow_uuid, seq_num, start_time)
            put!(channel(net, node=>flow_src; permit_forward=true), qdatagram_success)
            # TODO implement Pauli corrections

            # Check if there are any LinkLevelReplyAtHop messages and turn them into QTCPPairEnd messages
            link_reply = querydelete!(mb, LinkLevelReplyAtHop, ❓, ❓, ❓)
            @assert !isnothing(link_reply) "No LinkLevelReplyAtHop message found after the success of flow $(flow_uuid), sequence $(seq_num) at node $(node)"
            _, flow_uuid, seq_num, memory_slot = link_reply.tag
            pair_end = QTCPPairEnd(;
                flow_uuid,
                flow_src=0, # TODO we do not actually know the source node, it is not something that was kept track of
                flow_dst=node,
                seq_num,
                memory_slot,
                start_time
            )
            put!(net[node], pair_end)
            @debug "[$(now(sim))]: datagram from flow $(flow_uuid) pair $(seq_num) reached final destination"
        end

        # generate qdatagrams
        if now(sim) >= next_generation_time
            next_generation_time = next_generation_time + 1 / sending_rate
            for uuid in current_flows
                @debug "[$(now(sim))]: flow $(uuid) generating datagram"
                qdatagrams_in_flight[uuid] += 1
                dst        = destination[uuid]
                seq_num    = qdatagrams_sent[uuid] += 1
                start_time = now(sim)
                corrections = 0 # TODO implement Pauli corrections
                qdatagram = QuantumSavory.ProtocolZoo.QDatagram(uuid, node, dst, corrections, seq_num, start_time)
                put!(net[node], qdatagram)
            end
        end

        # wait until we have received a message or the timer has expired
        @yield (onchange(mb) | timeout(sim, next_generation_time - now(sim)))
    end
end


@kwdef struct EnhancedLinkController <: QuantumSavory.ProtocolZoo.AbstractProtocol
    """time-and-schedule-tracking instance from `ConcurrentSim`"""
    sim::Simulation
    """a network graph of registers"""
    net::RegisterNet
    """the vertex index of one of the nodes in the link (Alice)"""
    nodeA::Int
    """the vertex index of one of the nodes in the link (Bob)"""
    nodeB::Int
    """the probability with which the link can generate Bell pairs"""
    attempt_probability::Float64
    """the time it takes for an attempt to generate Bell pairs"""
    attempt_time::Float64
end

EnhancedLinkController(net::RegisterNet, nodeA::Int, nodeB::Int, attempt_probability::Float64, attempt_time::Float64) = EnhancedLinkController(get_time_tracker(net), net, nodeA, nodeB, attempt_probability, attempt_time)

EnhancedLinkController(net::RegisterNet, nodeA::Int, nodeB::Int; rate::Float64=1.0) = EnhancedLinkController(get_time_tracker(net), net, nodeA, nodeB, 0.001*rate, 0.001)

@resumable function (prot::EnhancedLinkController)()
    (;sim, net, nodeA, nodeB, attempt_probability, attempt_time) = prot
    mbA = messagebuffer(net, nodeA)
    mbB = messagebuffer(net, nodeB)

    request_queue = []

    while true

        # poll the message buffers for new requests
        while (llrequestA = querydelete!(mbA, LinkLevelRequest, ❓, ❓, nodeB)) != nothing
            @debug "[$(now(sim))]: LinkController $(nodeA) $(nodeB) received a request from node $(nodeA) to node $(nodeB)"
            push!(request_queue, (llrequestA, nodeA, nodeB))
        end
        while (llrequestB = querydelete!(mbB, LinkLevelRequest, ❓, ❓, nodeA)) != nothing
            @debug "[$(now(sim))]: LinkController $(nodeA) $(nodeB) received a request from node $(nodeB) to node $(nodeA)"
            push!(request_queue, (llrequestB, nodeB, nodeA))
        end

        # schedule and serve a request
        llrequest, originator_node, destination_node = isempty(request_queue) ? (nothing, nothing, nothing) : popfirst!(request_queue)

        if !isnothing(llrequest)

            @assert originator_node != destination_node "LinkController $(nodeA) $(nodeB) has a link request with originator node $(originator_node) equal to the destination node $(destination_node)"
            _, flow_uuid, seq_num, remote_node = llrequest.tag
            server = LinkRequestServer(;
                sim, net,
                nodeA, nodeB,
                attempt_probability, attempt_time,
                flow_uuid, seq_num
            )
            proc = @process server()

        end
        # wait until we have served a request
        @yield proc
    end
end

@kwdef struct LinkRequestServer <: QuantumSavory.ProtocolZoo.AbstractProtocol
    """time-and-schedule-tracking instance from `ConcurrentSim`"""
    sim::Simulation
    """a network graph of registers"""
    net::RegisterNet
    """the vertex index of one of the nodes in the link (Alice)"""
    nodeA::Int
    """the vertex index of one of the nodes in the link (Bob)"""
    nodeB::Int
    """the probability with which the link can generate Bell pairs"""
    attempt_probability::Float64
    """the time it takes for an attempt to generate Bell pairs"""
    attempt_time::Float64
    """uuid of the request to serve"""
    flow_uuid::Int
    """the sequence number of the request to serve"""
    seq_num::Int
end

@resumable function (prot::LinkRequestServer)()
    (;sim, net, nodeA, nodeB, attempt_probability, attempt_time, flow_uuid, seq_num) = prot

    entangler = EntanglerProt(;
                sim, net,
                nodeA, nodeB,
                tag=nothing,
                rounds=1, attempts=-1, success_prob=attempt_probability, attempt_time=attempt_time
            )
    # TODO have a timeout on how long to wait for an entangler to complete
    @debug "[$(now(sim))]: LinkController $(nodeA) $(nodeB) received a request for flow $(flow_uuid) pair $(seq_num) from node $(originator_node) to node $(destination_node)" 
    proc = @process entangler()
    @debug "[$(now(sim))]: LinkController $(nodeA) $(nodeB) completed entanglement generation for flow $(flow_uuid) pair $(seq_num)"
    _, slotA, _, slotB = @yield proc
    # Create the reply with the flow information and memory slot
    reply = LinkLevelReply(
        flow_uuid=flow_uuid,
        seq_num=seq_num,
        memory_slot=slotA
    )
    reply_at_destination = LinkLevelReplyAtHop(
        flow_uuid=flow_uuid,
        seq_num=seq_num,
        memory_slot=slotB
    )

    # Put the reply in the appropriate node's message buffer
    put!(net[originator_node], reply)
    put!(net[destination_node], reply_at_destination)

end

@kwdef struct E2EEntanglementTrackerProt <: QuantumSavory.ProtocolZoo.AbstractProtocol
    """time-and-schedule-tracking instance from `ConcurrentSim`"""
    sim::Simulation
    """a network graph of registers"""
    net::RegisterNet
    """the vertex index of the node that will be tracked (Alice)"""
    nodeA::Int
    """the vertex index of the node that will be tracked (Bob)"""
    nodeB::Int
end

E2EEntanglementTrackerProt(net::RegisterNet, nodeA::Int, nodeB::Int) = E2EEntanglementTrackerProt(get_time_tracker(net), net, nodeA, nodeB)

@resumable function (prot::E2EEntanglementTrackerProt)()
    (;sim, net, nodeA, nodeB) = prot
    mbA = messagebuffer(net, nodeA)
    mbB = messagebuffer(net, nodeB)
    deleted_pair_ends = []

    while true
        # check if there is a QTCPPairEnd message
        while (pair_end = querydelete!(mbB, QTCPPairEnd, ❓, ❓, ❓, ❓, ❓, ❓)) != nothing
            @debug "[$(now(sim))]: E2ETrackerProt $(nodeA) $(nodeB) received a QTCPPairEnd message for flow $(pair_end.tag[1]) pair $(pair_end.tag[5]) at node $(nodeB)"
            push!(deleted_pair_ends, pair_end)
            _, flow_uuid, flow_src, flow_dst, seq_num, memory_slot, start_time = pair_end.tag
            @assert flow_src == 0 "E2ETrackerProt $(nodeA) $(nodeB) received a QTCPPairEnd message with flow source $(flow_src) instead of 0"
            @assert flow_dst == nodeB "E2ETrackerProt $(nodeA) $(nodeB) received a QTCPPairEnd message with flow destination $(flow_dst) instead of $(nodeB)"
            @debug "[$(now(sim))]: QTCPPairEnd message for flow $(flow_uuid) pair $(seq_num) received at node $(nodeB)"
        end

        for pair_end in deleted_pair_ends
            _, flow_uuid, flow_src, flow_dst, seq_num, memory_slot, start_time = pair_end.tag
            # check if there is a corresponding QTCPPairBegin message
            pair_begin = querydelete!(mbA, QTCPPairBegin, flow_uuid, ❓, flow_dst, seq_num, ❓, ❓)
            if !isnothing(pair_begin) "E2ETrackerProt $(nodeA) $(nodeB) received a QTCPPairEnd message for flow $(flow_uuid) pair $(seq_num) but no corresponding QTCPPairBegin message was found"
                _, _, _, _, _, src_memory_slot, start_time = pair_begin.tag

                @debug "[$(now(sim))]: E2ETrackerProt $(nodeA) $(nodeB) received a QTCPPairEnd message for flow $(flow_uuid) pair $(seq_num) with corresponding QTCPPairBegin message from node $(flow_src)"

                # create an EntanglementCounterpart tag and use it to tag the memory slot, for both end nodes
                counterpart_alice = QuantumSavory.ProtocolZoo.EntanglementCounterpart(flow_dst, memory_slot)
                counterpart_bob  = QuantumSavory.ProtocolZoo.EntanglementCounterpart(flow_src, src_memory_slot)
                tag!(net[nodeA][src_memory_slot], counterpart_alice)
                tag!(net[nodeB][memory_slot], counterpart_bob)
                # the entanglement consumer will then catch these tags

                # finally remove the QTCPPairEnd message from the array
                delete!(deleted_pair_ends, pair_end)
            end
        end

        # wait until we have received a message
        @yield (onchange(mbB)|onchange(mbA))
    end
end


@kwdef struct EnhancedNetworkNodeController <: QuantumSavory.ProtocolZoo.AbstractProtocol
    """time-and-schedule-tracking instance from `ConcurrentSim`"""
    sim::Simulation
    """a network graph of registers"""
    net::RegisterNet
    """the vertex index of where the protocol is located"""
    node::Int
end


@resumable function (prot::EnhancedNetworkNodeController)()
    (;sim, net, node) = prot
    mb = messagebuffer(net, node)
    datagrams_in_waiting = Dict{Tuple{Int,Int},Tuple{Tag,Int}}() # keyed by flow_uuid, seq_num; storing datagram and next hop
    @debug "[$(now(sim))]: EnhancedNetworkNodeController at node $(node) started"
    while true
        qdatagram = querydelete!(mb, QuantumSavory.ProtocolZoo.QDatagram, ❓, ❓, !=(node), ❓, ❓, ❓)
        if !isnothing(qdatagram)
            @debug "[$(now(sim))]: QDatagram received at node $(node) for flow $(qdatagram.tag[2]) pair $(qdatagram.tag[6])"
            _, flow_uuid, flow_src, flow_dst, corrections, seq_num, start_time = qdatagram.tag
            nexthop = first(Graphs.a_star(net.graph, node, flow_dst::Int)).dst
            request = LinkLevelRequest(flow_uuid, seq_num, nexthop)
            datagrams_in_waiting[(flow_uuid, seq_num)] = (qdatagram.tag, nexthop)
            put!(mb, request)
        end

        # Check for LinkLevelReply messages
        # TODO have timeouts on how long to wait for a reply
        llreply = querydelete!(mb, LinkLevelReply, ❓, ❓, ❓)
        if !isnothing(llreply)
            _, flow_uuid, seq_num, memory_slot = llreply.tag
            # Find the corresponding QDatagram that matches this reply
            qdatagram, nexthop = pop!(datagrams_in_waiting, (flow_uuid, seq_num))
            # Process the entanglement and forward the datagram
            _, flow_uuid, flow_src, flow_dst, corrections, seq_num, start_time = qdatagram

            # Perform entanglement swapping
            if node == flow_src
                put!(net[node], LinkLevelReplyAtSource(flow_uuid, seq_num, memory_slot))
            else
                # Find the corresponding LinkLevelReplyAtHop message from the previous hop (when the current node was the destination node)
                llreply_at_destination = querydelete!(mb, LinkLevelReplyAtHop, ❓, ❓, ❓)
                @assert !isnothing(llreply_at_destination) "No LinkLevelReplyAtHop message found for flow $(flow_uuid), sequence $(seq_num) at node $(node)"
                _, _, _, memory_slot_at_destination = llreply_at_destination.tag
                swapcircuit = QuantumSavory.CircuitZoo.LocalEntanglementSwap()
                xmeas, zmeas = swapcircuit(net[node,memory_slot], net[node,memory_slot_at_destination])
                # TODO: use xmeas and zmeas to add a correction to the datagram
            end

            # Forward the datagram to the next node in the path
            new_qdatagram = QuantumSavory.ProtocolZoo.QDatagram(flow_uuid, flow_src, flow_dst, corrections, seq_num, start_time)
            put!(channel(net, node=>nexthop; permit_forward=false), new_qdatagram)
        end

        # Wait until we have received a message
        @yield onchange(mb)
    end
end
