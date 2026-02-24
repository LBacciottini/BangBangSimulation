module BangBangSimulation

using DataFrames
using CSV
using Revise

export setup, setup_seq, dump_log,
# dataprocess.jl
import_data, analyze_data, analyze_consumer_data,
# scenarios.jl
Scenario, bk_link_capacity, mean_ci, safe_analyze_consumer_data, default_config,
sliding_window_analysis, run_one_sliding, run_one

include("setup.jl")
include("dataprocess.jl")
include("scenarios.jl")



end # module