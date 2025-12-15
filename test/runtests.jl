using BangBangSimulation
using Test
using CSV
using DataFrames

@testset "BangBangSimulation.jl" begin
    # first test is just to see if the module loads
    @test isdefined(BangBangSimulation, :BangBangSimulation)

    # second test: a simple simulation with three nodes and one memory slot each
    @testset "Simple three-node simulation" begin
        sim, net = setup(1, 1, 10.0, linklength=10.0, coherencetime=nothing, outfolder="../out/", usetempfile=true)
        run(sim, 10)
        # read the out file "../out/_results_temp.csv" and check if there are entries
        results = CSV.read("../out/_results_temp.csv", DataFrame)
        # clean up the temp file
        rm("../out/_results_temp.csv", force=true)

        @test size(results, 1) <= 2 # There should be at most 2 entries in the results file because swapping node has no memory slots to swap."

        sim, net = setup(1, 10, 10.0, linklength=10.0, coherencetime=nothing, outfolder="../out/", usetempfile=true)
        run(sim, 10)
        # read the out file "../out/_results_temp.csv" and check if there are entries
        results = CSV.read("../out/_results_temp.csv", DataFrame)
        # clean up the temp file
        rm("../out/_results_temp.csv", force=true)
        @test size(results, 1) > 5 # There should be more than 2 entries in the results file because swapping node has memory slots to swap."
    end
end