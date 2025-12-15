module BangBangSimulation

using DataFrames
using CSV
using Revise

export setup, setup_seq, dump_log,
# dataprocess.jl
import_data, analyze_data, analyze_consumer_data

include("setup.jl")
include("dataprocess.jl")



end # module