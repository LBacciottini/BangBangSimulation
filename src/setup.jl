

using QuantumSavory.ProtocolZoo
using QuantumSavory
using Graphs
using CSV
import ConcurrentSim
using ConcurrentSim: Simulation, @yield, timeout, @process, now, Process
import ResumableFunctions
using ResumableFunctions: @resumable
import QuantumSavory: Tag

include("enhanced_swapping.jl")

function repeater_chain(nrepeaters::Int, nslots::Int; linklength::AbstractFloat=0.0, coherencetime::Union{AbstractFloat, Nothing}=nothing)

    speedOfLightFiber = 2e8 # km/s
    delay = linklength / speedOfLightFiber
    delay_us = delay * 1e6 # in microseconds

    graph = grid([nrepeaters + 2])
    if coherencetime === nothing
        noisemodel = nothing
        registers = vcat([Register(nslots*5)], [Register(nslots) for _ in 1:nrepeaters], [Register(nslots*5)])
    else
        noisemodel = Depolarization(coherencetime)
        registers = vcat([Register(nslots*5)], [Register(nslots, noisemodel) for _ in 1:nrepeaters], [Register(nslots*5)])
    end
    return RegisterNet(graph, registers; classical_delay=delay_us)
end

@kwdef struct PrivateEntanglementCounterpart
    "the id of the remote node to which we are entangled"
    remote_node::Int
    "the slot in the remote node containing the qubit we are entangled to"
    remote_slot::Int
end
Base.show(io::IO, tag::PrivateEntanglementCounterpart) = print(io, "(Private) Entangled to $(tag.remote_node).$(tag.remote_slot)")
Tag(tag::PrivateEntanglementCounterpart) = Tag(PrivateEntanglementCounterpart, tag.remote_node, tag.remote_slot)


function get_entangler(sim, net, nodeA, nodeB, rate , capacity, slack)

    # compute the max number of attempts to wait, to avoid waiting too much
    attempt_time = 0.001
    attempt_probability = attempt_time * rate
    slack_window = (1 - slack) / capacity
    max_waiting_time = slack_window
    max_attempts = max(1, Int(ceil(max_waiting_time / attempt_time)))

    return EntanglerProt(sim, net, nodeA, nodeB; attempt_time=attempt_time, success_prob=attempt_probability, rounds=1, tag=PrivateEntanglementCounterpart, margin=1, retry_lock_time=nothing, attempts=max_attempts)
end

function get_swapper(sim, net, node)
    return EnhancedSwapperProt(sim, net, node; nodeL= <(node), nodeH= >(node), retry_lock_time=nothing, policy="OQF")
end


@kwdef struct CustomEntanglementTracker <: QuantumSavory.ProtocolZoo.AbstractProtocol
    """time-and-schedule-tracking instance from `ConcurrentSim`"""
    sim::Simulation
    """a network graph of registers"""
    net::RegisterNet
    """the vertex of the node where the tracker is working"""
    node::Int
end

CustomEntanglementTracker(net::RegisterNet, node::Int) = CustomEntanglementTracker(get_time_tracker(net), net, node)

@resumable function (prot::CustomEntanglementTracker)()
    nodereg = prot.net[prot.node]
    mb = messagebuffer(prot.net, prot.node)
    while true
        workwasdone = true # waiting is not enough because we might have multiple rounds of work to do
        while workwasdone
            workwasdone = false
            for (updatetagsymbol, updategate) in ((EntanglementUpdateX, Z), (EntanglementUpdateZ, X), (QuantumSavory.ProtocolZoo.EntanglementDelete, nothing)) # TODO this is getting ugly. Refactor EntanglementUpdateX and EntanglementUpdateZ to be the same parameterized tag
                # look for EntanglementUpdate? or EntanglementDelete message sent to us
                if !isnothing(updategate) # EntanglementUpdate
                    msg = querydelete!(mb, updatetagsymbol, ❓, ❓, ❓, ❓, ❓, ❓)
                    isnothing(msg) && continue
                    (src, (_, pastremotenode, pastremoteslotid, localslotid, newremotenode, newremoteslotid, correction)) = msg
                else # EntanglementDelete
                    msg = querydelete!(mb, updatetagsymbol, ❓, ❓, ❓, ❓)
                    isnothing(msg) && continue
                    (src, (_, pastremotenode, pastremoteslotid, _, localslotid)) = msg
                end

                @info "CustomEntanglementTracker (entry found) @$(prot.node): Received from $(msg.src).$(msg.tag[3]) | message=`$(msg.tag)` | time=$(now(prot.sim))"
                workwasdone = true
                localslot = nodereg[localslotid]

                # Check if the local slot is still present and believed to be entangled.
                # We will need to perform a correction operation due to the swap or a deletion due to the qubit being thrown out,
                # but there will be no message forwarding necessary.
                @debug "CustomEntanglementTracker @$(prot.node): EntanglementCounterpart requesting lock at $(now(prot.sim))"
                @yield lock(localslot)
                @debug "CustomEntanglementTracker @$(prot.node): EntanglementCounterpart getting lock at $(now(prot.sim))"
                counterpart = querydelete!(localslot, EntanglementCounterpart, pastremotenode, pastremoteslotid)
                unlock(localslot)
                if !isnothing(counterpart)
                    # time_before_lock = now(prot.sim)
                    @yield lock(localslot)
                    # time_after_lock = now(prot.sim)
                    # time_before_lock != time_after_lock && @debug "CustomEntanglementTracker @$(prot.node): Needed Δt=$(time_after_lock-time_before_lock) to get a lock"
                    if !isassigned(localslot)
                        unlock(localslot)
                        error("There was an error in the entanglement tracking protocol `CustomEntanglementTracker`. We were attempting to forward a classical message from a node that performed a swap to the remote entangled node. However, on reception of that message it was found that the remote node has lost track of its part of the entangled state although it still keeps a `Tag` as a record of it being present.") # TODO make it configurable whether an error is thrown and plug it into the logging module
                    end
                    if !isnothing(updategate) # EntanglementUpdate
                        # Pauli frame correction gate
                        if correction==2
                            apply!(localslot, updategate)
                        end
                        if newremotenode != -1 #TODO: this is a bit hacky
                            # tag local with updated EntanglementCounterpart new_remote_node new_remote_slot_idx
                            tag!(localslot, EntanglementCounterpart, newremotenode, newremoteslotid)
                        end
                    else # EntanglementDelete
                        traceout!(localslot)
                    end
                    unlock(localslot)
                    continue
                end

                # If there is nothing still stored locally, check if we have a record of the entanglement being swapped to a different remote node,
                # and forward the message to that node.
                history = querydelete!(localslot, EntanglementHistory,
                                    pastremotenode, pastremoteslotid, # who we were entangled to (node, slot)
                                    ❓, ❓,                             # who we swapped with (node, slot)
                                    ❓)                                # which local slot used to be entangled with whom we swapped with
                if !isnothing(history)
                    _, _, _, whoweswappedwith_node, whoweswappedwith_slotidx, swappedlocal_slotidx = history.tag
                    if !isnothing(updategate) # EntanglementUpdate
                        # Forward the update tag to the swapped node and store a new history tag so that we can forward the next update tag to the new node
                        tag!(localslot, EntanglementHistory, newremotenode, newremoteslotid, whoweswappedwith_node, whoweswappedwith_slotidx, swappedlocal_slotidx)
                        @info "CustomEntanglementTracker (entry not found) @$(prot.node): history=`$(history)` | message=`$msg` | Sending to $(whoweswappedwith_node).$(whoweswappedwith_slotidx)"
                        msghist = Tag(updatetagsymbol, pastremotenode, pastremoteslotid, whoweswappedwith_slotidx, newremotenode, newremoteslotid, correction)
                        put!(channel(prot.net, prot.node=>whoweswappedwith_node; permit_forward=true), msghist)

                        # In this case, we should have another EntanglementHistory entry relative to the other slot that was swapped. We also need to update that history tag and notify the new remote node
                        second_localslot = nodereg[swappedlocal_slotidx]
                        history2 = querydelete!(second_localslot, EntanglementHistory, #  Query the second slot involved in the swap described by the history at this node. There should be an entry for it too
                                                whoweswappedwith_node, whoweswappedwith_slotidx,  # Who the second slot was entangled to
                                                pastremotenode, pastremoteslotid,                   # Who the first slot was entangled to
                                                localslot.idx)                                         # The first slot index
                        @assert !isnothing(history2)
                        tag!(second_localslot, EntanglementHistory, whoweswappedwith_node, whoweswappedwith_slotidx, newremotenode, newremoteslotid, localslot.idx)
                        # @info "CustomEntanglementTracker @$(prot.node): history2=`$(history2)` | message=`$msg` | Sending to $(newremotenode).$(newremoteslotid)"
                        # msghist2 = Tag(updatetagsymbol, prot.node, localslot.idx, whoweswappedwith_node, whoweswappedwith_slotidx)
                        # put!(channel(prot.net, prot.node=>newremotenode; permit_forward=true), msghist2)

                    else # EntanglementDelete
                        # We have a delete message but the qubit was swapped so add a tag and forward to swapped node
                        @debug "CustomEntanglementTracker @$(prot.node): history=`$(history)` | message=`$msg` | Sending to $(whoweswappedwith_node).$(whoweswappedwith_slotidx)"
                        msghist = Tag(updatetagsymbol, pastremotenode, pastremoteslotid, whoweswappedwith_node, whoweswappedwith_slotidx)
                        tag!(localslot, updatetagsymbol, prot.node, localslot.idx, whoweswappedwith_node, whoweswappedwith_slotidx)
                        put!(channel(prot.net, prot.node=>whoweswappedwith_node; permit_forward=true), msghist)
                    end
                    continue
                

                
                end

                # Finally, if there the history of a swap is not present in the log anymore,
                # it must be because a delete message was received, and forwarded,
                # and the entanglement history was deleted, and replaced with an entanglement delete tag.
                if !isnothing(querydelete!(localslot, QuantumSavory.ProtocolZoo.EntanglementDelete, prot.node, localslot.idx, pastremotenode, pastremoteslotid)) #deletion from both sides of the swap, deletion msg when both qubits of a pair are deleted, or when EU arrives after ED at swap node with two simultaneous swaps and deletion on one side
                    if !(isnothing(updategate)) # EntanglementUpdate
                        # to handle a possible delete-swap-swap case, we need to update the EntanglementDelete tag
                        tag!(localslot, QuantumSavory.ProtocolZoo.EntanglementDelete, prot.node, localslot.idx, newremotenode, newremoteslotid)
                        @debug "CustomEntanglementTracker @$(prot.node): message=`$msg` for deleted qubit handled and EntanglementDelete tag updated"
                    else # EntanglementDelete
                        # when the message is EntanglementDelete and the slot history also has an EntanglementDelete tag (both qubits were deleted), do nothing
                        @debug "CustomEntanglementTracker @$(prot.node): message=`$msg` is for a deleted qubit and is thus dropped"
                    end
                    continue
                end

                error("`CustomEntanglementTracker` on node $(prot.node) received a message $(msg) that it does not know how to handle (due to the absence of corresponding `EntanglementCounterpart` or `EntanglementHistory` or `EntanglementDelete` tags). This might have happened due to `CutoffProt` deleting qubits while swaps are happening. Make sure that the retention times in `CutoffProt` are sufficiently larger than the `agelimit` in `SwapperProt`. Otherwise, this is a bug in the protocol and should not happen -- please report an issue at QuantumSavory's repository.")
            end
        end
        # @debug "CustomEntanglementTracker @$(prot.node): Starting message wait at $(now(prot.sim)) with MessageBuffer containing: $(mb.buffer)"
        @yield onchange(mb)
        @debug "CustomEntanglementTracker @$(prot.node): Message wait ends at $(now(prot.sim))"
    end
end


function get_tracker(sim, net, node)
    return CustomEntanglementTracker(sim, net, node)
end

@kwdef struct CustomConsumerProt <: QuantumSavory.ProtocolZoo.AbstractProtocol
    sim::Simulation
    net::RegisterNet
    nodeA::Int
    nodeB::Int
end




@kwdef struct CustomEntanglementConsumer <: QuantumSavory.ProtocolZoo.AbstractProtocol
    """time-and-schedule-tracking instance from `ConcurrentSim`"""
    sim::Simulation
    """a network graph of registers"""
    net::RegisterNet
    """the vertex index of node A"""
    nodeA::Int
    """the vertex index of node B"""
    nodeB::Int
    """time period between successive queries on the nodes (`nothing` for queuing up and waiting for available pairs)"""
    period::Union{Float64,Nothing} = 0.1
    """tag type which the consumer is looking for -- the consumer query will be `query(node, EntanglementConsumer.tag, remote_node)` and it will be expected that `remote_node` possesses the symmetric reciprocal tag; defaults to `EntanglementCounterpart`"""
    tag::Any = EntanglementCounterpart
    """stores the time and resulting observable from querying nodeA and nodeB for `EntanglementCounterpart`"""
    _log::Vector{@NamedTuple{t::Float64, obs1::Float64, obs2::Float64}} = @NamedTuple{t::Float64, obs1::Float64, obs2::Float64}[]
end

function CustomEntanglementConsumer(sim::Simulation, net::RegisterNet, nodeA::Int, nodeB::Int; kwargs...)
    return CustomEntanglementConsumer(;sim, net, nodeA, nodeB, kwargs...)
end
function CustomEntanglementConsumer(net::RegisterNet, nodeA::Int, nodeB::Int; kwargs...)
    return CustomEntanglementConsumer(get_time_tracker(net), net, nodeA, nodeB; kwargs...)
end

permits_virtual_edge(::CustomEntanglementConsumer) = true

@resumable function (prot::CustomEntanglementConsumer)()
    regA = prot.net[prot.nodeA]
    regB = prot.net[prot.nodeB]
    while true
        query1 = query(regA, prot.tag, prot.nodeB, ❓; locked=false, assigned=true) # TODO Need a `querydelete!` dispatch on `Register` rather than using `query` here followed by `untag!` below
        if isnothing(query1)
            if isnothing(prot.period)
                @info "$(QuantumSavory.timestr(prot.sim)) CustomEntanglementConsumer($(QuantumSavory.compactstr(regA)), $(QuantumSavory.compactstr(regB))): query on first node found no entanglement. Waiting on tag updates in $(QuantumSavory.compactstr(regA))."
                @yield onchange(regA, Tag)
            else
                @info "$(QuantumSavory.timestr(prot.sim)) CustomEntanglementConsumer($(QuantumSavory.compactstr(regA)), $(QuantumSavory.compactstr(regB))): query on first node found no entanglement. Waiting a fixed amount of time."
                @yield timeout(prot.sim, prot.period::Float64)
            end
            continue
        else
            query2 = query(regB, prot.tag, prot.nodeA, query1.slot.idx; locked=false, assigned=true)
            if isnothing(query2) # in case EntanglementUpdate hasn't reached the second node yet, but the first node has the EntanglementCounterpart
                if isnothing(prot.period)
                    @info "$(QuantumSavory.timestr(prot.sim)) CustomEntanglementConsumer($(QuantumSavory.compactstr(regA)), $(QuantumSavory.compactstr(regB))): query on second node found no entanglement (yet...). Waiting on tag updates in $(QuantumSavory.compactstr(regB))."
                    @yield onchange(regB, Tag)
                else
                    @info "$(QuantumSavory.timestr(prot.sim)) CustomEntanglementConsumer($(QuantumSavory.compactstr(regA)), $(QuantumSavory.compactstr(regB))): query on second node found no entanglement (yet...). Waiting a fixed amount of time."
                    @yield timeout(prot.sim, prot.period::Float64)
                end
                continue
            end
        end

        q1 = query1.slot
        q2 = query2.slot
        @yield lock(q1) & lock(q2)

        @info "$(QuantumSavory.timestr(prot.sim)) CustomEntanglementConsumer($(QuantumSavory.compactstr(regA)), $(QuantumSavory.compactstr(regB))): queries successful, consuming entanglement between .$(q1.idx) and .$(q2.idx)"
        untag!(q1, query1.id)
        untag!(q2, query2.id)
        # TODO do we need to add EntanglementHistory or EntanglementDelete and should that be a different EntanglementHistory since the current one is specifically for Swapper
        # TODO currently when calculating the observable we assume that EntanglerProt.pairstate is always (|00⟩ + |11⟩)/√2, make it more general for other states

        if observable((q1, q2), Z⊗Z) === nothing || observable((q1, q2), X⊗X) === nothing
            @error "CustomEntanglementConsumer: One of the observables is not defined for the qubit pair. This should not happen. Investigating."
            @error "RegRef pair A: $(QuantumSavory.compactstr(q1)), $(query1.tag[2]).$(query1.tag[3]), RegRef B: $(QuantumSavory.compactstr(q2)), $(query2.tag[2]).$(query2.tag[3])"
            @error "Node $(prot.nodeA), slot .$(q1.idx), other node $(prot.nodeB), slot .$(q2.idx). Quantum states (should match): q1=$(regA.staterefs[q1.idx]), q2=$(regB.staterefs[q2.idx])"
        end

        ob1 = real(observable((q1, q2), Z⊗Z))
        ob2 = real(observable((q1, q2), X⊗X))

        traceout!(regA[q1.idx], regB[q2.idx])
        push!(prot._log, (now(prot.sim), ob1, ob2))
        unlock(q1)
        unlock(q2)
        if !isnothing(prot.period)
            @yield timeout(prot.sim, prot.period)
        end
    end
end







function get_consumer(sim, net, nodeA, nodeB)
    return CustomEntanglementConsumer(sim, net, nodeA, nodeB, period=nothing)
end

@kwdef struct SchedulerProt <: QuantumSavory.ProtocolZoo.AbstractProtocol
    sim::Simulation
    net::RegisterNet
    nodeA::Int
    nodeB::Int
    capacity::AbstractFloat = 10
    slack::AbstractFloat = 0.4
    out_filename::Union{String, Nothing} = nothing
    _log::Vector{@NamedTuple{node::Int, queueA::Int, queueB::Int, high_rate::Bool, time::Float64}} = @NamedTuple{node::Int, queueA::Int, queueB::Int, high_rate::Bool, time::Float64}[]
end
SchedulerProt(sim, net, nodeA, nodeB; capacity=10, slack=0.4, out_filename=nothing) = SchedulerProt(sim, net, nodeA, nodeB, capacity, slack, out_filename, @NamedTuple{node::Int, queueA::Int, queueB::Int, high_rate::Bool, time::Float64}[])

function log(prot::SchedulerProt; kwargs...)

    # if kwargs are exactly the fields of the log, we store them in the internal log
    if keys(kwargs) == (:node, :queueA, :queueB, :high_rate, :time)
        nt = (; kwargs...)
        row = NamedTuple{(:node, :queueA, :queueB, :high_rate, :time), Tuple{Int, Int, Int, Bool, Float64}}(nt)
        push!(prot._log, row)
    else
        @warn "Logging called with unexpected fields: $(keys(kwargs)). Expected (:node, :queueA, :queueB, :high_rate, :time). Ignoring log."
    end

    # check that the log has at most 10000 entries to avoid memory issues
    length(prot._log) > 100 && dump_log(prot)
end

function dump_log(prot::SchedulerProt)
    fn = prot.out_filename
    fn === nothing && return
    first_write = !isfile(fn)
    if first_write
        header = NamedTuple{Tuple(collect(keys(prot._log[1])))}(Tuple(Vector{Any}() for _ in keys(prot._log[1])))  # 0-row, named columns
        CSV.write(fn, header; append=false)
    end
    CSV.write(fn, prot._log; append=true, header=false)
    empty!(prot._log)
end

function dump_log(prot::CustomEntanglementConsumer, outfolder::String, consumer_out_file::String)
    log = prot._log
    # log is a vector of (time, obs1, obs2) tuples. Store it in a CSV file
    df = DataFrame(time=Float64[], obs1=Float64[], obs2=Float64[])
    for (t, o1, o2) in log
        push!(df, (t, o1, o2))
    end
    CSV.write(joinpath(outfolder, consumer_out_file), df)
end

function get_queue_state(node::Int, net::RegisterNet)
    reg = net[node]
    entangled_left = queryall(reg, EntanglementCounterpart, <(node), ❓)
    entangled_right = queryall(reg, EntanglementCounterpart, >(node), ❓)
    # assert that one side is empty (up to the newly generated entanglement)
    @assert (length(entangled_left) <= 1) || (length(entangled_right) <= 1)
    return length(entangled_right) - length(entangled_left)
end

function should_slow_down(qlenA, qlenB, node, net)
    
    # TODO: ugly hack to determine if we are at the first or last link
    total_nodes = size(net.graph)[1]

    is_first_link = (node == 1)
    is_last_link = (node == total_nodes - 1)
    if is_first_link
        return qlenB < 0
    elseif is_last_link
        return qlenA > 0
    else
        return (qlenB - qlenA <= 0) && !(qlenB == 0 && qlenA == 0)
    end
end


@resumable function (prot::SchedulerProt)()
    sim, net, nodeA, nodeB = prot.sim, prot.net, prot.nodeA, prot.nodeB
    new_rate = prot.capacity
    last_time_logged = now(sim)
    while true
        entangler = get_entangler(sim, net, nodeA, nodeB, new_rate, prot.capacity, prot.slack)
        proc = @process entangler()

        @debug "Scheduler at link $nodeA, $nodeB waiting for entanglement to be generated at rate $new_rate."
        _, slotA, _, slotB = @yield proc

        should_log = false

        if last_time_logged <= now(sim) - 1.0 # log at most once per second
            last_time_logged = now(sim)
            should_log = true
        end

        if slotA !== nothing && slotB !== nothing

            # now we double check that the entanglement is on this link
            @debug "Scheduler at link $nodeA, $nodeB woke up to new entanglement at time $(now(sim))."
            private_tagged = query(net[nodeA], PrivateEntanglementCounterpart, nodeB, slotB)
            @assert !isnothing(private_tagged)
            slot, id, tag = private_tagged.slot, private_tagged.id, private_tagged.tag
            tag = PrivateEntanglementCounterpart(tag[2], tag[3])
            @assert tag.remote_node == nodeB

            # Our Entangler Protocol has generated a new Bell pair.

            # then, we replace the private tag with a public one to be used by the swapper
            untag!(slot, id)
            new_tag = EntanglementCounterpart(tag.remote_node, tag.remote_slot)
            tag!(slot, new_tag)

            # need to do the same thing on nodeB
            private_tagged_B = query(net[nodeB], PrivateEntanglementCounterpart, nodeA, slotA)
            @assert !isnothing(private_tagged_B)
            slotB, idB, tagB = private_tagged_B.slot, private_tagged_B.id, private_tagged_B.tag
            tagB = PrivateEntanglementCounterpart(tagB[2], tagB[3])
            untag!(slotB, idB)
            new_tagB = EntanglementCounterpart(tagB.remote_node, tagB.remote_slot)
            tag!(slotB, new_tagB)

            @debug "Link $nodeA, $nodeB generated new entanglement. Replaced tag $tag with $new_tag"
        else
            @debug "Scheduler at link $nodeA, $nodeB woke up but no entanglement was generated."
        end
        
        # Evaluate the new operating regime
        queueA = get_queue_state(nodeA, net)
        queueB = get_queue_state(nodeB, net)
        new_rate = prot.capacity

        should_slow = should_slow_down(queueA, queueB, nodeA, net)

        if should_log
            # log the queue states
            log(prot, node=nodeA, queueA=queueA, queueB=queueB, high_rate=!should_slow, time=now(sim))
        end

        if should_slow
            # slow down
            @debug "Link $nodeA, $nodeB slowing down. Queue states: $queueA, $queueB"
            new_rate = prot.capacity * (1 - prot.slack)
        end
    end
end


function setup(nrepeaters::Int, nslots::Int, linkcapacity::AbstractFloat; linklength::AbstractFloat=0.0, slack=0.4, coherencetime::Union{AbstractFloat, Nothing}=nothing, outfolder::String="./out/", outfile::Union{String, Nothing}=nothing, usetempfile::Bool=false)
    net = repeater_chain(nrepeaters, nslots; linklength=linklength, coherencetime=coherencetime)
    sim = get_time_tracker(net)

    # Setup an entanglement tracker at each node
    for node in 1:(nrepeaters+2)
        tracker = get_tracker(sim, net, node)
        @process tracker()
    end

    # Setup the entanglement swapping protocols at each repeater
    for node in 2:(nrepeaters+1)
        swapper = get_swapper(sim, net, node)
        @process swapper()
    end

    # Setup the scheduler protocol at each link
    schedulers = []
    for node in 1:(nrepeaters+1)
        filename = usetempfile ? joinpath(outfolder, "_results_temp.csv") : (outfile === nothing ? joinpath(outfolder, "results.csv") : joinpath(outfolder, outfile))
        scheduler = SchedulerProt(sim, net, node, node+1; capacity=linkcapacity, slack=slack, out_filename=filename)
        push!(schedulers, scheduler)
        @process scheduler()
    end

    # Setup consumers at the end nodes
    consumer = get_consumer(sim, net, 1, nrepeaters+2)
    @process consumer()

    return (sim, net, consumer, schedulers)
end


####################################
# SEQUENTIAL SWAPS SIMULATION SETUP
####################################

include("enhanced_qtcp.jl")

function setup_seq(nrepeaters::Int, nslots::Int, linkcapacity::AbstractFloat; linklength::AbstractFloat=0.0, slack=0.4, coherencetime::Union{AbstractFloat, Nothing}=nothing, outfolder::String="./out/", outfile::Union{String, Nothing}=nothing, usetempfile::Bool=false)
    net = repeater_chain(nrepeaters, nslots; linklength=linklength, coherencetime=coherencetime)
    sim = get_time_tracker(net)

    # No need for entanglement trackers in this setup

    # Setup the entanglement swapping protocols at each repeater
    for node in 1:(nrepeaters+2)
        swapper = EnhancedNetworkNodeController(sim, net, node)
        @process swapper()
    end

    # Setup the LinkControllers at each link
    schedulers = []
    for node in 1:(nrepeaters+1)
        linkcontroller = EnhancedLinkController(sim, net, node, node+1, 0.001*linkcapacity, 0.001)
        push!(schedulers, linkcontroller)
        @process linkcontroller()
    end

    # Setup consumers at the end nodes
    e2etracker = E2EEntanglementTrackerProt(sim, net, 1, nrepeaters+2)
    consumer = get_consumer(sim, net, 1, nrepeaters+2)
    @process e2etracker()
    @process consumer()

    # Setup the sequential rate controller
    seq_rate_controller_alice = EndNodeRateController(sim, net, 1, linkcapacity*(1-slack))
    @process seq_rate_controller_alice()
    seq_rate_controller_bob = EndNodeRateController(sim, net, nrepeaters+2, linkcapacity*(1-slack))
    @process seq_rate_controller_bob()

    flow = QuantumSavory.ProtocolZoo.QTCP.Flow(1, nrepeaters+2, 1e9, 1101)
    put!(net[1], flow)

    return (sim, net, consumer, nothing)
end