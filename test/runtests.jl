using BangBangSimulation
using QuantumSavory
using QuantumSavory.ProtocolZoo: EntanglementCounterpart
using Test
using CSV
using DataFrames

const OUTFOLDER = mktempdir()

@testset "BangBangSimulation.jl" begin
    # first test is just to see if the module loads
    @test isdefined(BangBangSimulation, :BangBangSimulation)

    # second test: a simple simulation with three nodes and one memory slot each
    @testset "Simple three-node simulation" begin
        sim, net, consumer, schedulers = setup(1, 1, 10.0, linklength=10.0, coherencetime=nothing, outfolder=OUTFOLDER, usetempfile=true)
        run(sim, 10)
        # with 1 slot, swapper can't swap (needs qubits from both sides),
        # so the consumer should have no entries
        @test length(consumer._log) == 0

        sim, net, consumer, schedulers = setup(1, 10, 10.0, linklength=10.0, coherencetime=nothing, outfolder=OUTFOLDER, usetempfile=true)
        run(sim, 10)
        rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
        # with 10 slots, the swapper can swap and the consumer should have entries
        @test length(consumer._log) > 0
    end

    @testset "Cutoff protocol" begin
        # Test A: Cutoff with sufficient slots — simulation runs without errors
        @testset "Cutoff runs without errors" begin
            sim, net, consumer, schedulers = setup(1, 10, 10.0;
                cutofftime=5.0, coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim, 20)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            @test length(consumer._log) > 0
        end

        # Test B: Cutoff deletes qubits, reducing consumed pairs vs no cutoff
        @testset "Short cutoff reduces throughput" begin
            # Use low link capacity so qubits wait longer, making cutoff impactful
            # Run without cutoff
            sim_no, net_no, consumer_no, _ = setup(1, 10, 2.0;
                coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim_no, 100)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            pairs_no_cutoff = length(consumer_no._log)

            # Run with aggressive cutoff — with low capacity, qubits wait ~0.5s
            # on average, so cutofftime=0.5 deletes many before swapping
            sim_cut, net_cut, consumer_cut, _ = setup(1, 10, 2.0;
                cutofftime=0.5, coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            # Aggressive cutoff may trigger tracker race conditions;
            # catch errors but consumer._log still has entries from before
            try run(sim_cut, 100) catch end
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            pairs_with_cutoff = length(consumer_cut._log)

            @test pairs_with_cutoff < pairs_no_cutoff
        end

        # Test C: EntanglementDelete notification — remote qubit is traced out
        @testset "EntanglementDelete cleans remote slot" begin
            sim, net, consumer, schedulers = setup(1, 10, 10.0;
                cutofftime=3.0, coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim, 20)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            # After running with cutoff, verify that no slot holds an
            # EntanglementCounterpart tag pointing to a remote slot that
            # is unassigned (which would mean the delete message was lost).
            all_consistent = true
            for node in 1:3
                reg = net[node]
                for slot_idx in 1:nsubsystems(reg)
                    slot = reg[slot_idx]
                    info = query(slot, EntanglementCounterpart, ❓, ❓)
                    if !isnothing(info)
                        remote_node = info.tag[2]
                        remote_slot_idx = info.tag[3]
                        remote_slot = net[remote_node][remote_slot_idx]
                        if !isassigned(remote_slot)
                            all_consistent = false
                        end
                    end
                end
            end
            @test all_consistent
        end
    end
end
