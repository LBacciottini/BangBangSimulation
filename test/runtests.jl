using BangBangSimulation
using QuantumSavory
using QuantumSavory.ProtocolZoo: EntanglementCounterpart
using Test
using CSV
using DataFrames
using Logging
using Random

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
            # Run without cutoff (seeded for determinism)
            Random.seed!(42)
            sim_no, net_no, consumer_no, _ = setup(1, 10, 2.0;
                coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim_no, 100)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            pairs_no_cutoff = length(consumer_no._log)

            # Run with cutoff=1.0 (seeded with same seed for identical entanglement timing)
            Random.seed!(42)
            sim_cut, net_cut, consumer_cut, _ = setup(1, 10, 2.0;
                cutofftime=1.0, coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim_cut, 100)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            pairs_with_cutoff = length(consumer_cut._log)

            @test pairs_with_cutoff > 0
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

    @testset "Scenario utilities" begin
        @testset "bk_link_capacity" begin
            # Zero-length link: no fiber loss, p_det = static_eff
            bk = bk_link_capacity(linklength_km=0.0, excitation_time_s=1e-6, static_eff=0.5)
            @test bk.p_det ≈ 0.5
            @test bk.p_success ≈ 0.5^2 / 2
        end

        @testset "mean_ci" begin
            # Known vector
            m, lo, hi = mean_ci([1.0, 2.0, 3.0])
            @test m ≈ 2.0
            @test lo < 2.0
            @test hi > 2.0
            # CI should be symmetric around the mean
            @test (2.0 - lo) ≈ (hi - 2.0)

            # NaN filtering with warning
            m2, lo2, hi2 = @test_logs (:warn, r"dropped 1 NaN") mean_ci([NaN, 1.0])
            @test m2 ≈ 1.0
            @test lo2 ≈ 1.0
            @test hi2 ≈ 1.0

            # All NaN
            m3, lo3, hi3 = @test_logs (:warn, r"dropped 1 NaN") mean_ci([NaN])
            @test isnan(m3)
            @test isnan(lo3)
            @test isnan(hi3)
        end

        @testset "safe_analyze_consumer_data" begin
            # DataFrame with known obs1 values at time > 100
            df = DataFrame(
                time=[50.0, 150.0, 200.0, 250.0],
                obs1=[0.0, 0.8, 0.9, 0.7],
                obs2=[0.0, 0.0, 0.0, 0.0],
            )
            fid, thr = safe_analyze_consumer_data(df)
            # analyze_consumer_data computes mean(obs1) over steady state (time > 100)
            # then safe_ applies (3*raw + 1)/4
            mean_obs1 = (0.8 + 0.9 + 0.7) / 3
            @test fid ≈ (3 * mean_obs1 + 1) / 4

            # All time < 100 → not enough steady state data
            df_early = DataFrame(
                time=[10.0, 20.0],
                obs1=[0.5, 0.6],
                obs2=[0.0, 0.0],
            )
            fid2, thr2 = safe_analyze_consumer_data(df_early)
            @test isnan(fid2)
            @test isnan(thr2)
        end

        @testset "default_config consistency" begin
            cfg = default_config()
            @test cfg.coherencetime == 2.0
            @test cfg.scenarios[4].cutofftime ≈ cfg.coherencetime / 50.0
            @test length(cfg.scenarios) == 5
        end
    end

    @testset "Sliding window analysis" begin
        NT = @NamedTuple{t::Float64, obs1::Float64, obs2::Float64}

        @testset "Uniform log" begin
            log = NT[(t=Float64(i), obs1=1.0, obs2=1.0) for i in 1:50]
            df = sliding_window_analysis(log; window_size=10.0, step=1.0)
            @test :time in propertynames(df)
            @test :fidelity in propertynames(df)
            @test :throughput in propertynames(df)
            # With obs1=1.0 everywhere, fidelity = (3*1.0+1)/4 = 1.0
            steady = filter(r -> r.time >= 20.0, df)
            @test all(f -> f ≈ 1.0, steady.fidelity)
            # 10 events in a 10s window → throughput = 1.0
            @test all(t -> t ≈ 1.0, steady.throughput)
        end

        @testset "Empty windows" begin
            log = NT[(t=100.0, obs1=0.5, obs2=0.5)]
            df = sliding_window_analysis(log; window_size=10.0, step=1.0, start_time=10.0)
            early = filter(r -> r.time < 90.0, df)
            @test all(isnan, early.fidelity)
            @test all(t -> t == 0.0, early.throughput)
        end

        @testset "Step change" begin
            log_low = NT[(t=Float64(i), obs1=0.0, obs2=0.0) for i in 1:25]
            log_high = NT[(t=Float64(i), obs1=1.0, obs2=1.0) for i in 26:50]
            log = vcat(log_low, log_high)
            df = sliding_window_analysis(log; window_size=10.0, step=1.0)
            early_fid = filter(r -> r.time == 15.0, df).fidelity[1]
            late_fid = filter(r -> r.time == 45.0, df).fidelity[1]
            @test late_fid > early_fid
        end

        @testset "Output schema" begin
            log = NT[]
            df = sliding_window_analysis(log; window_size=5.0, step=1.0)
            @test :time in propertynames(df)
            @test :fidelity in propertynames(df)
            @test :throughput in propertynames(df)
            @test nrow(df) == 0
        end
    end

    @testset "Multi-repeater simulation" begin
        @testset "2 repeaters produce end-to-end pairs" begin
            sim, net, consumer, schedulers = setup(2, 10, 2.0;
                linklength=10.0, coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim, 30)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            @test length(consumer._log) > 0
            # Consumer log entries are (time, obs1, obs2) tuples
            for entry in consumer._log
                @test entry.t > 0.0
                @test -1.0 <= entry.obs1 <= 1.0
                @test -1.0 <= entry.obs2 <= 1.0
            end
        end

        @testset "3 repeaters with bang-bang" begin
            sim, net, consumer, schedulers = setup(3, 10, 2.0;
                linklength=10.0, slack=0.3, coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim, 30)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            @test length(consumer._log) > 0
            # 3 repeaters → 5 nodes, 4 links → 4 schedulers
            @test length(schedulers) == 4
        end

        @testset "3 repeaters with YQF policy" begin
            sim, net, consumer, schedulers = setup(3, 10, 2.0;
                linklength=10.0, policy="YQF", coherencetime=10.0,
                outfolder=OUTFOLDER, usetempfile=true)
            run(sim, 30)
            rm(joinpath(OUTFOLDER, "_results_temp.csv"), force=true)
            @test length(consumer._log) > 0
        end
    end

    @testset "dump_log and CSV output" begin
        @testset "Consumer CSV has correct schema" begin
            tmpdir = mktempdir()
            sim, net, consumer, schedulers = setup(2, 10, 2.0;
                linklength=10.0, coherencetime=10.0,
                outfolder=tmpdir, usetempfile=true)
            run(sim, 30)
            dump_log(consumer, tmpdir, "consumer_test.csv")
            df = CSV.read(joinpath(tmpdir, "consumer_test.csv"), DataFrame)
            @test :time in propertynames(df)
            @test :obs1 in propertynames(df)
            @test :obs2 in propertynames(df)
            @test nrow(df) == length(consumer._log)
            @test all(df.time .> 0.0)
            @test eltype(df.time) <: Float64
            @test eltype(df.obs1) <: Float64
            @test eltype(df.obs2) <: Float64
        end

        @testset "Scheduler CSV is written" begin
            tmpdir = mktempdir()
            sim, net, consumer, schedulers = setup(1, 10, 10.0;
                linklength=10.0, coherencetime=10.0,
                outfolder=tmpdir, outfile="sched_test.csv", usetempfile=false)
            run(sim, 20)
            for s in schedulers
                dump_log(s)
            end
            sched_path = joinpath(tmpdir, "sched_test.csv")
            @test isfile(sched_path)
            sched_df = CSV.read(sched_path, DataFrame)
            @test nrow(sched_df) > 0
        end
    end

    @testset "run_one_sliding" begin
        # Use small scenarios (few slots, short sim) to keep tests fast
        test_scenarios = [
            Scenario("test_no_bb", 10, 0.0, "OQF", nothing),
            Scenario("test_bb", 10, 0.30, "OQF", nothing),
        ]
        for scenario in test_scenarios
            df = run_one_sliding(
                scenario;
                nrepeaters=1,
                linkcapacity=10.0,
                linklength_km=10.0,
                coherencetime=10.0,
                sim_time=20.0,
                window_size=5.0,
                window_step=1.0,
                seed=42,
            )
            @test df isa DataFrame
            @test nrow(df) > 0
            @test :time in propertynames(df)
            @test :fidelity in propertynames(df)
            @test :throughput in propertynames(df)
            @test :scenario in propertynames(df)
            @test all(s -> s == scenario.name, df.scenario)
        end

        @testset "2 repeaters sliding window" begin
            scenario = Scenario("test_2rep", 10, 0.0, "OQF", nothing)
            df = run_one_sliding(
                scenario;
                nrepeaters=2,
                linkcapacity=10.0,
                linklength_km=10.0,
                coherencetime=10.0,
                sim_time=20.0,
                window_size=5.0,
                window_step=1.0,
                seed=42,
            )
            @test df isa DataFrame
            @test nrow(df) > 0
            @test all(s -> s == "test_2rep", df.scenario)
            # Time column should be monotonically increasing
            @test issorted(df.time)
            # Throughput should be non-negative
            @test all(t -> t >= 0.0, df.throughput)
        end
    end

    @testset "run_one" begin
        # Use small scenarios with short sim_time
        test_scenarios = [
            Scenario("test_finite", 10, 0.0, "OQF", nothing),
            Scenario("test_bangbang", 10, 0.30, "OQF", nothing),
        ]
        for scenario in test_scenarios
            out = mktempdir()
            r = run_one(
                1,
                scenario;
                nrepeaters=1,
                linkcapacity=10.0,
                linklength_km=10.0,
                coherencetime=10.0,
                sim_time=20.0,
                out_root=out,
            )
            @test haskey(r, :seed)
            @test haskey(r, :fidelity)
            @test haskey(r, :throughput)
            @test r.seed == 1
            # Check that output files were created
            scenario_dir = joinpath(out, scenario.name)
            @test isdir(scenario_dir)
            @test isfile(joinpath(scenario_dir, "consumer_seed1.csv"))
            @test isfile(joinpath(scenario_dir, "run_seed1.csv"))
            # Verify run CSV has the right content
            run_df = CSV.read(joinpath(scenario_dir, "run_seed1.csv"), DataFrame)
            @test nrow(run_df) == 1
            @test :seed in propertynames(run_df)
            @test :fidelity in propertynames(run_df)
            @test :throughput in propertynames(run_df)
        end

        @testset "2 repeaters run_one" begin
            out = mktempdir()
            scenario = Scenario("test_2rep_run", 10, 0.0, "OQF", nothing)
            r = run_one(
                42,
                scenario;
                nrepeaters=2,
                linkcapacity=10.0,
                linklength_km=10.0,
                coherencetime=10.0,
                sim_time=20.0,
                out_root=out,
            )
            @test haskey(r, :fidelity)
            @test haskey(r, :throughput)
            @test r.seed == 42
            # Consumer CSV should have entries
            consumer_df = CSV.read(joinpath(out, scenario.name, "consumer_seed42.csv"), DataFrame)
            @test nrow(consumer_df) > 0
            @test :time in propertynames(consumer_df)
            @test :obs1 in propertynames(consumer_df)
        end
    end
end
