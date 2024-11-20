
"""
    sample_rejection(f, n, support; nbins=1000)

Generates `n` samples using the rejection sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional. The number of equidistant points in `support` to find the maximum of `f`. Default is `1000`.

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
```julia
julia> data = sample_rejection(exp, 10, (0, 4))
```
"""
function sample_rejection(f, n, support; nbins = 1000)
    samples = []
    M = maximum(f, range(support..., nbins))
    while length(samples) < n
        x = rand() * (support[2] - support[1]) + support[1]
        y = rand() * M
        if y <= f(x)
            push!(samples, x)
        end
    end
    return samples
end

function invert_on_grid(y_i; x_grid, y_values)
    i_bin = findfirst(y_values .> y_i) - 1
    f_i = (y_i - y_values[i_bin]) / (y_values[i_bin+1] - y_values[i_bin])
    return x_grid[i_bin] + f_i * (x_grid[i_bin+1] - x_grid[i_bin])
end

"""
    sample_inversion(f, n, support; nbins=1000)

Generates `n` samples using the inversion sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional keyword argument. The number of equidistant points in `support` for which the c.d.f. is pre-computed.

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
```julia
julia> data = sample_inversion(exp, 10, (0, 4))
```
"""
function sample_inversion(f, n, support; nbins = 1000)
    cdf(x) = quadgk(f, support[1], x)[1]
    x_grid = range(support..., nbins + 1)
    y_values = cdf.(x_grid) ./ cdf(support[2])
    #
    y_sample = rand(n)
    x_sample = invert_on_grid.(y_sample; x_grid, y_values)
    return x_sample
end
