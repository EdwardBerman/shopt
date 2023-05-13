

# another Pluto.jl Cell
let
# cf. https://github.com/JuliaPlots/Makie.jl/issues/822#issuecomment-769684652
# with scale argument that is required now
struct LogMinorTicks end

    function MakieLayout.get_minor_tickvalues(::LogMinorTicks, scale, tickvalues, vmin, vmax)
        vals = Float64[]
        for (lo, hi) in zip(
                @view(tickvalues[1:end-1]),
                @view(tickvalues[2:end]))
            interval = hi-lo
            steps = log10.(LinRange(10^lo, 10^hi, 11))
            append!(vals, steps[2:end-1])
        end
        vals
    end

custom_formatter(values) = map(v -> "10" * Makie.UnicodeFun.to_superscript(round(Int64, v)), values)
    data = star
    fig = Figure()
    ax, hm = heatmap(fig[1, 1], log10.(data),
    axis=(; xminorticksvisible=true,
       xminorticks=IntervalsBetween(9)))
    ax.xlabel = "U"
    ax.ylabel = "V"
    cb = Colorbar(fig[1, 2], hm;
    tickformat=custom_formatter,
    minorticksvisible=true,
    minorticks=LogMinorTicks())
    ax.title = "Log Scale Model PSF"
    save(joinpath("outdir", "test.png"), fig)
end
