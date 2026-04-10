#!/usr/bin/env julia
#
# Plot sliding-window throughput and fidelity from out_sliding/sliding_window_all.csv.
# Produces two publication-quality figures saved as PDF.
#
# Usage:
#   julia --project=. plot_sliding.jl
#
# Edit the `scenarios_to_plot` list below to select which scenarios appear in the figures.

using Pkg; Pkg.activate(".")
using CSV, DataFrames, CairoMakie

# ── Configuration ────────────────────────────────────────────────────────

input_file = "out_sliding/sliding_window_all.csv"
output_dir = "out_sliding"

# Select which scenarios to plot and how they appear in the legend.
# Comment/uncomment lines to include/exclude scenarios.
scenarios_to_plot = [
    # (scenario name in CSV,       legend label,             line style, color)
    ("S1_infinite_no_bangbang",     "Swap-ASAP (OQF)",        :solid,    :royalblue),
    # ("S2_finite6_no_bangbang",    "Finite 6 slots",         :dash,     :red),
    ("S3_finite50_no_bangbang",   "Finite 50 slots",        :dash,     :orange),
    ("S4_infinite_yqf_no_bangbang", "Swap-ASAP (YQF)",        :solid,    :crimson),
    # ("S5_infinite_bangbang_slack30","Bang-Bang (slack=0.30)",  :dash,     :forestgreen),
    # ("S6_infinite_bangbang_slack50","Bang-Bang (slack=0.50)",  :dashdot,  :darkorange),
    ("S7_infinite_bangbang_slack15","Bang-Bang (slack=0.15)",  :dot,      :mediumorchid),
]

# ── Load data ────────────────────────────────────────────────────────────

df = CSV.read(input_file, DataFrame)
selected_names = Set(first.(scenarios_to_plot))
df = filter(row -> row.scenario in selected_names, df)

# ── Theme for publication figures ────────────────────────────────────────

publication_theme = Theme(
    fontsize = 18,
    Axis = (
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
        titlesize = 22,
        spinewidth = 1.2,
        xtickwidth = 1.2,
        ytickwidth = 1.2,
        xgridvisible = false,
        ygridvisible = true,
        ygridstyle = :dash,
        ygridwidth = 0.5,
    ),
    Legend = (
        labelsize = 14,
        framevisible = true,
        framewidth = 0.8,
        padding = (8, 8, 6, 6),
    ),
    Lines = (
        linewidth = 2.0,
    ),
)

set_theme!(publication_theme)

# ── Helper: plot one metric ──────────────────────────────────────────────

function plot_metric(df, scenarios_to_plot, metric::Symbol, ylabel::String, title::String, output_path::String; axis_kwargs...)
    fig = Figure(size = (700, 450))
    ax = Axis(fig[1, 1]; xlabel = "Time (s)", ylabel = ylabel, title = title, axis_kwargs...)

    for (name, label, lstyle, color) in scenarios_to_plot
        sdf = filter(row -> row.scenario == name, df)
        isempty(sdf) && continue
        sort!(sdf, :time)
        lines!(ax, sdf.time, sdf[!, metric]; label = label, linestyle = lstyle, color = color)
    end

    Legend(fig[1, 2], ax)

    save(output_path, fig; pt_per_unit = 1)
    println("Saved: $output_path")
    return fig
end

# ── Secret Key Rate ──────────────────────────────────────────────────────

# Binary entropy: h(p) = -p log2(p) - (1-p) log2(1-p), with h(0) = h(1) = 0.
h_binary(p) = (p <= 0 || p >= 1) ? 0.0 : -p * log2(p) - (1 - p) * log2(1 - p)

# BB84 secret key rate: SKR = throughput * max(0, 1 - h(QBER))
# where QBER = 0.5 - 0.5*F (quantum bit error rate derived from fidelity).
function secret_key_rate(throughput, fidelity)
    qber = 0.5 - 0.5 * fidelity
    r = 1.0 - h_binary(qber)
    return throughput * max(0.0, r)
end

# Add SKR column to the dataframe.
df[!, :skr] = secret_key_rate.(df.throughput, df.fidelity)

# ── Generate figures ─────────────────────────────────────────────────────

fig1 = plot_metric(df, scenarios_to_plot, :throughput,
    "Throughput (pairs/s)", "End-to-end throughput",
    joinpath(output_dir, "throughput.pdf"))

fig2 = plot_metric(df, scenarios_to_plot, :fidelity,
    "Fidelity", "End-to-end fidelity",
    joinpath(output_dir, "fidelity.pdf");
    limits = (nothing, (0.4, 1.0)), yticks = 0.4:0.1:1.0)

fig3 = plot_metric(df, scenarios_to_plot, :skr,
    "Secret key rate (bits/s)", "Secret key rate (BB84)",
    joinpath(output_dir, "secret_key_rate.pdf"))

println("Done.")
