using BangBangSimulation
using DataFrames
using CSV

using Logging


global_logger(ConsoleLogger(stderr, Logging.Debug))

function run_batch(nrepeaters::Int, nslots::Int, linkcapacity::AbstractFloat, slacks::Vector{Float64}; linklength::AbstractFloat=0.0, coherencetime::Union{AbstractFloat, Nothing}=nothing, outfolder::String="./out/", should_run::Bool=true, swap_asap=true)

    # setup and run simulation for each slack value
    if should_run
        for slack in slacks
            out_file = "results_slack_$(round(Int, slack*100)).csv"
            consumer_out_file = "consumer_slack_$(round(Int, slack*100)).csv"
            setup_func = swap_asap ? setup : setup_seq
            sim, _, consumer, schedulers = setup_func(nrepeaters, nslots, linkcapacity; linklength=linklength, slack=slack, coherencetime=coherencetime, outfile=out_file, outfolder=outfolder, usetempfile=false)
            run(sim, 5)  # run for 1000 seconds
            # dump logs from schedulers
            if swap_asap
                for scheduler in schedulers
                    dump_log(scheduler)
                end
            end
            dump_log(consumer, outfolder, consumer_out_file)
        end
    end

    # now process the data from each simulation and create a big DataFrame
    all_results = DataFrame(node=Int[], slack=Float64[], throughput=Float64[], sojourn_time=Float64[], fidelity=Float64[])
    for slack in slacks
        out_file = joinpath(outfolder, "results_slack_$(round(Int, slack*100)).csv")
        consumers_out_file = joinpath(outfolder, "consumer_slack_$(round(Int, slack*100)).csv")
        if swap_asap
            results = analyze_data(import_data(out_file), import_data(consumers_out_file))
        else
            results = analyze_data(nothing, import_data(consumers_out_file))
        end
        # The data frame should have columns: node, slack, throughput, sojourn_time
        # so we need to add a slack column with constant value to processed
        for node in keys(results)
            push!(all_results, (node=node, slack=slack, throughput=results[node][1], sojourn_time=results[node][2], fidelity=results[node][3]))
        end

    end

    return all_results
end

link_capacity = 10.0  # entanglements per second
coherence_time = 500/link_capacity  # seconds


# Example: run a batch of simulations with different slack values
slacks = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
slacks = [0.9]
results = run_batch(3, 200, link_capacity, slacks; coherencetime=coherence_time, outfolder="./out_3_rep_seq/")

# store the results in a CSV file
CSV.write("./out_3_rep_seq/batch_results.csv", results)