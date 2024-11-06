
"""
    WithData(factor, support)
    WithData(bins, N::Int = 1)
    WithData(h::Hist1D)

A struct that helps plotting a model curve on top of the binned data with correct normalization.
If the model is normalized to 1, the scaling factor accounts for the bin width and number of entries in the histogram.
For a model curve normalized to the total number of events, default option `N=1` is provided.

# Examples:
```julia
julia> data = randn(1000);
julia> h = Hist1D(data; binedges=range(-5,5, 100));
julia> model_fun(x) = length(data) * exp(-x^2 / 2) / sqrt(2π);
julia> #
julia> let
    plot(h, seriestype=:stepbins)
    plot!(model_fun, WithData(h.binedges[1]), lw=2)
end  # displays the plot
```
"""
struct WithData
    factor::Float64
    support::Tuple{Float64, Float64}
end
WithData(bins, N::Int = 1) = WithData(N * diff(bins)[1], (bins[1], bins[end]))
WithData(h::Hist1D) = WithData(h.binedges[1], Int(sum(h.bincounts)))
#
@recipe function f(model_fun::Function, scale::WithData; n_points = 100)
    normalized_function(x) = model_fun(x) * scale.factor
    xv = range(scale.support..., n_points)
    return (normalized_function, xv)
end
