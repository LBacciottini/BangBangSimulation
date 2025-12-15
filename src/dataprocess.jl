using DataFrames
using Statistics

function import_data(filename::String)
    if !isfile(filename)
        error("File $filename does not exist.")
    end
    df = CSV.read(filename, DataFrame)
    return df
end

function analyze_data(df::Union{DataFrame, Nothing}, consumer_df::DataFrame)
    steady_state = filter(row -> row.time > 100.0, df)
    # we need to compute, for each node: (i) avg throughput, (ii) avg sojourn time
    # Throughput is tricky. We need to count as 1 each time queue_state moves towards 0
    # We can compute avg queue size and use Little's law for sojourn time with queue size and throughput
    results = Dict{Int, Tuple{Float64, Float64, Float64}}()  # node => (throughput, sojourn_time, fidelity)
    fidelity, throughput = analyze_consumer_data(consumer_df)
    fidelity = (3*fidelity + 1)/4  # convert to fidelity of |Φ+>
    for node in unique(steady_state.node)
        node_data = filter(row -> row.node == node, steady_state)
        println("Analyzing data for node $node. The size of node_data is $(size(node_data, 1))")
        avg_queue_size = mean(abs.(node_data.queueA))
        println("Node $node has avg queue size $avg_queue_size and throughput $throughput")
        sojourn_time = avg_queue_size / throughput
        results[node] = (throughput, sojourn_time, fidelity)
    end
    
    
    return results
end

function analyze_consumer_data(df::DataFrame)
    # df has columns: time, obs1, obs2
    steady_state = filter(row -> row.time > 100.0, df)
    fidelity = mean(steady_state.obs1)
    throughput = size(steady_state, 1) / (maximum(steady_state.time) - minimum(steady_state.time))
    
    return fidelity, throughput
end