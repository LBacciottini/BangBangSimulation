using CSV, DataFrames, CairoMakie

df = CSV.read("out_scenarios/scenario_sweep.csv", DataFrame)

# Scenario display names and colors
scenario_style = Dict(
    "S1_infinite_no_bangbang"      => (label="S1: ∞ slots, OQF",        color=:red,       marker=:circle),
    "S2_finite6_no_bangbang"       => (label="S2: 6 slots, OQF",        color=:blue,      marker=:rect),
    "S3_finite50_no_bangbang"      => (label="S3: 50 slots, OQF",       color=:green,     marker=:diamond),
    "S4_infinite_yqf_no_bangbang"  => (label="S4: ∞ slots, YQF",        color=:orange,    marker=:utriangle),
    "S7_infinite_bangbang_optslack" => (label="S7: ∞ slots, YQF, BB opt.", color=:purple,   marker=:star5),
    "S8_infinite_pyqf_no_bangbang"  => (label="S8: ∞ slots, PYQF",        color=:teal,     marker=:hexagon),
    "S9_infinite_yqf_cutoff02"      => (label="S9: ∞ slots, YQF, cutoff=0.2s", color=:brown, marker=:dtriangle),
    "S10_infinite_bangbang_cutoff02" => (label="S10: ∞ slots, YQF, BB 80%, cutoff=0.2s", color=:magenta, marker=:cross),
)
scenarios_ordered = [
    "S1_infinite_no_bangbang",
    "S3_finite50_no_bangbang",
    "S4_infinite_yqf_no_bangbang",
    "S7_infinite_bangbang_optslack",
    "S8_infinite_pyqf_no_bangbang",
    "S9_infinite_yqf_cutoff02",
    "S10_infinite_bangbang_cutoff02",
]

# BB84 secret key rate: SKR = throughput × max(0, 1 - 2h(QBER))
# For depolarized Bell state with fidelity F:  QBER = 2(1-F)/3
function binary_entropy(x)
    (x ≤ 0 || x ≥ 1) && return 0.0
    return -x * log2(x) - (1 - x) * log2(1 - x)
end

function bb84_skr(fidelity, throughput)
    qber = 2 * (1 - fidelity) / 3
    r = 1 - 2 * binary_entropy(qber)
    return throughput * max(0.0, r)
end

# Compute SKR from mean fidelity/throughput where not already in CSV
if !hasproperty(df, :mean_skr)
    df.mean_skr .= NaN
end
for i in 1:nrow(df)
    if ismissing(df.mean_skr[i]) || isnan(df.mean_skr[i])
        df.mean_skr[i] = bb84_skr(df.mean_fidelity[i], df.mean_throughput[i])
    end
end

# ── Helpers to plot metrics ──
function plot_metric!(ax, df, scenarios_ordered, scenario_style, ycol, ylo, yhi)
    for sname in scenarios_ordered
        sdf = sort(filter(r -> r.scenario == sname, df), :linklength_km)
        st = scenario_style[sname]
        band!(ax, sdf.linklength_km, sdf[!, ylo], sdf[!, yhi];
              color=(st.color, 0.15))
        scatterlines!(ax, sdf.linklength_km, sdf[!, ycol];
                      label=st.label, color=st.color, marker=st.marker,
                      markersize=10, linewidth=2)
    end
end

function plot_metric_noband!(ax, df, scenarios_ordered, scenario_style, ycol)
    for sname in scenarios_ordered
        sdf = sort(filter(r -> r.scenario == sname, df), :linklength_km)
        st = scenario_style[sname]
        scatterlines!(ax, sdf.linklength_km, sdf[!, ycol];
                      label=st.label, color=st.color, marker=st.marker,
                      markersize=10, linewidth=2)
    end
end

# ── Create figure ──
fig = Figure(size=(1200, 1200), fontsize=14)

# Plot 1: Throughput
ax1 = Axis(fig[1, 1];
    xlabel="Link length (km)", ylabel="Throughput (pairs/s)",
    title="End-to-end throughput vs link length",
    yscale=log10)
plot_metric!(ax1, df, scenarios_ordered, scenario_style,
             :mean_throughput, :ci_throughput_low, :ci_throughput_high)
axislegend(ax1; position=:rt)

# Plot 2: Fidelity
ax2 = Axis(fig[2, 1];
    xlabel="Link length (km)", ylabel="Fidelity",
    title="End-to-end fidelity vs link length")
plot_metric!(ax2, df, scenarios_ordered, scenario_style,
             :mean_fidelity, :ci_fidelity_low, :ci_fidelity_high)
axislegend(ax2; position=:rb)

# Plot 3: BB84 SKR
ax3 = Axis(fig[3, 1];
    xlabel="Link length (km)", ylabel="Secret key rate (bits/s)",
    title="BB84 secret key rate vs link length",
    yscale=log10)
plot_metric_noband!(ax3, df, scenarios_ordered, scenario_style, :mean_skr)
axislegend(ax3; position=:rt)

save("out_scenarios/scenario_plots.pdf", fig)
println("Saved to out_scenarios/scenario_plots.pdf")
