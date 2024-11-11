## Fit and NLL

```julia
function fit_with_fixed(objective, initial; numbers_to_fix)
    n = length(initial)
    #
    unit_matrix = Diagonal(I, n)
    eye = reduce(.|, (unit_matrix[i, :] for i in numbers_to_fix))
    to_right_dims = unit_matrix[:, (1:n)[.!(eye)]]
    #
    optimize(to_right_dims' * initial, BFGS()) do p
        full_p = to_right_dims * p .+ initial .* eye
        objective(full_p)
    end
end
```

```julia
extended_nll(model, data;
    support = extrema(data), normalization_call = _quadgk_call)
```

### Automatic differentiation

- Implement a new flag for `fit_nll` function
- Add docstring for `extended_nll` without parameters

### Likelihood profile

Nothing to implement, just use `fit_with_fixed`

### Utils

```julia
function interpolate_to_zero(two_x, two_y)
    w_left = 1 ./ two_y .* [1, -1]
    w_left ./= sum(w_left)
    return two_x' * w_left
end


function find_zero_two_sides(xv, yv)
    yxv = yv .* xv
    _left = findfirst(x -> x > 0, yxv)
    _right = findlast(x -> x < 0, yxv)
    #
    x_left_zero = interpolate_to_zero([xv[_left-1], xv[_left]], [yv[_left-1], yv[_left]])
    x_right_zero =
        interpolate_to_zero([xv[_right], xv[_right+1]], [yv[_right], yv[_right+1]])
    #
    [x_left_zero, x_right_zero]
end
```

## Models

### General

````julia
abstract type SpectrumModel end
Base.collect(model::SpectrumModel) =
    getproperty.(model |> Ref, collect(fieldnames(typeof(model))))
#
# Define a generic constructor for any subtype of SpectrumModel
function (::Type{T})(p_values::Union{AbstractVector,NTuple}) where {T<:SpectrumModel}
    T(; NamedTuple{fieldnames(T)}(p_values)...)
end
````

### Anka

One gaussian + pol1 background

````julia
@with_kw struct Anka{P} <: SpectrumModel
    μ::P
    σ::P
    a::P
    flat::P
    log_slope::P
end
#
function peak1_func(model::Anka, x)
    @unpack μ, σ, a = model
    gaussian_scaled(x; μ, σ, a)
end
background_func(model::Anka, x) = pol1_with_logs_slope(x, model; x0 = sum(support) / 2)
#
"""
    total_func(model::Anka, x)

where

    Anka{P} <: SpectrumModel

evaluation of the pdf for model `Anka`.
is a simple spectral model that has two components,
 - a background described by pol1, and
 - a peaking signal described by the gaussian function
# Example
```julia
julia> model = Anka(; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)
julia> total_func(model, 3.3)
4.1775
```
"""
total_func(model::Anka, x) = peak1_func(model, x) + background_func(model, x)
````

### Frida

Two gaussian functions + pol1 background
