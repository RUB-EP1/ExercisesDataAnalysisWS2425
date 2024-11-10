function _data_in_support(data, support)
    red_data = filter(x -> support[1] < x < support[2], data)
    if length(data) > length(red_data)
        print("Reduced dataset to domain $support to get normalizations right")
    end
    return red_data
end


"""
    plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlbl="", ylbl="")

Plot a histogram of data with a fit using model overlaid and a pull distribution.

# Arguments
- `data`: A collection of data points.
- `model`: A function that represents the model to be fitted. It should take two arguments: data points and parameters.
- `binning`: The bin edges for the histogram.
- `best_fit_pars`: The best-fit parameters for the model.
- `xlbl`: (Optional) Label for the x-axis. Default is an empty string.
- `ylbl`: (Optional) Label for the y-axis. Default is an empty string.

# Example
```julia
data = log.(1 .+ (exp(1) - 1) .* rand(10_000))
model(x, p) = p[1] * exp(p[2] * x)
binning = 0:0.1:1.0
best_fit_pars = [1.0, 1.0]
plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlbl = "X-axis", ylbl = "Y-axis")
```
"""
function plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlbl = "", ylbl = "")
    # check if data is within limits
    support = (binning[1], binning[end])
    filtered_data = _data_in_support(data, support)
    # plot data
    h = Hist1D(filtered_data; binedges = binning)
    centers = (binning[1:end-1] + binning[2:end]) ./ 2
    scatter(
        centers,
        h.bincounts,
        # seriestype = :,
        mc = :black,
        yerror = sqrt.(h.bincounts),
        xerror = (step(binning)) / 2,
        ms = 1.5,
        legend = false,
        ylimits = (0, :auto),
        xlimits = support,
        lw = 1.5,
        tickfontfamily = "Times",
        ylabel = ylbl,
        guidefontvalign = :top,
        guidefontfamily = "Times",
    )
    # model normalization
    normalization = quadgk(x -> model(x, best_fit_pars), support...)[1]
    scale = step(binning) * length(filtered_data) / normalization
    best_scaled_model(x) = scale * model(x, best_fit_pars)
    # model plot
    plot!(best_scaled_model, support..., lw = 2, lc = :cornflowerblue)
    # add pull
    p = plot!(xaxis = nothing)
    yv_model = best_scaled_model.(centers)
    scatter(
        centers,
        (h.bincounts .- yv_model) ./ sqrt.(h.bincounts),
        ylims = (-7, 7),
        xerror = (step(binning)) / 2,
        yerror = 1.0,
        ms = 1.5,
        mc = :black,
        xlabel = xlbl,
        guidefonthalign = :right,
        tickfontfamily = "Times",
        ylabel = "Pull",
        guidefontfamily = "Times",
    )
    pull = hline!([0], lc = :gray)
    plot(
        p,
        pull,
        layout = grid(2, 1, heights = (0.8, 0.2)),
        link = :x,
        legend = false,
        xlimits = support,
        bottom_margin = [-4.75mm 0mm],
    )
end
