# Tutorial 4: Fits with fixed parameters and abstract models

In this tutorial, you will extend your fitting function to be able to fix parameters and define an abstract class for models.
In addition, the helper functions to numerically find zeros from lecture 5 should be implemented.

The new code tests for this tutorial are given at the bottom. The extension of the generic fitting function, the abstract model
implementation for the `Anka` and `Frida` models and the helper function to find zeros are needed.

Note that you might need to import more packages and add the new function exports to `src/DataAnalysisWS2425.jl`

## Fixing fit parameters

Extend your existing `fit_enll` function with an optional argument that fixes parameters of your model.
You will need to implement the logic for fixing parameters using their indices as shown in lecture 5.
The new function argument, called `fixed_parameter_indices` will thus be a list of integers,
representing the indices of parameters that should be fixed to their initial values.

## Models

Define the abstract type `SpectrumModel` in `src/functions.jl`, and use it to implement the `Anka` model from lecture 5-a.

```julia
abstract type SpectrumModel end
Base.collect(model::SpectrumModel) =
    getproperty.(model |> Ref, collect(fieldnames(typeof(model))))
#
# Define a generic constructor for any subtype of SpectrumModel
function (::Type{T})(p_values::Union{AbstractVector,NTuple}) where {T<:SpectrumModel}
    T(; NamedTuple{fieldnames(T)}(p_values)...)
end
```

The function is supposed to look like this to pass tests:

````julia
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

````

In exercise 3, you have added a second Gaussian peak to fit the new $\Xi_c^+ \rightarrow \Xi^- K^+ \pi^+$ decay process.
It is useful to define a new `SpectrumModel` for this; we'll call it Frida:

````julia
"""
total_func(model::Frida, x)

where

Frida{P} <: SpectrumModel

evaluation of the pdf for model `Frida`.
is a simple spectral model that has two components,
 - a background described by pol1, and
 - two peaking signals described by gaussian functions

# Example
```julia
julia> model = Frida(; μ1 = 2.29, σ1 = 0.005, μ2 = 2.47, σ2 = 0.008, flat = 1.5, log_slope = 2.1, a1 = 5.0, a2 = 1.0)
julia> total_func(model, 3.3)
11.895
```

"""

````

## Utils

Helper functions to find zeros, used in lecture 5-a, will be useful to have in our code-base.
They should have the following form:

````julia
"""
    interpolate_to_zero(two_x, two_y)

Interpolate to zero based on two points.

# Arguments
- `two_x::Vector`: A vector containing two x-values.
- `two_y::Vector`: A vector containing two y-values corresponding to `two_x`.

# Returns
- `Float64`: The interpolated x-value where y is zero.
"""
````

````julia
"""
    find_zero_two_sides(xv, yv)

Find the zero crossings on both sides of the x-axis.

# Arguments
- `xv::AbstractVector`: A vector of x-values.
- `yv::AbstractVector`: A vector of y-values corresponding to `xv`.

# Returns
- `Vector{Float64}`: A vector containing two x-values where the y-values are zero.
"""
````

<details> <summary> To copy and append to `test/runtests.jl`</summary>
Your `test/runtests.jl` file should look like this:

```julia
using Test
using DataAnalysisWS2425

include("test-distributions.jl")
include("test-sampling.jl")
include("test-fitting.jl")
include("test-plotting.jl")
include("test-utils.jl")
```

The following code should be appended to `test-fitting.jl`:

```julia
@testset "Fitting with fixed parameters" begin
    init_pars = (; μ = 0.35, σ = 0.8, a = 1.0)
    support = (-4.0, 4.0)
    Random.seed!(42)
    data = sample_inversion(400, support) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    model(x, pars) = gaussian_scaled(x; pars.μ, pars.σ, pars.a)
    ext_unbinned_fit =
        fit_enll(model, init_pars, data; support = support, fixed_parameter_indices = [2])
    best_pars_extnll = ext_unbinned_fit.minimizer
    @test ext_unbinned_fit.ls_success
    @test best_pars_extnll[1] ≈ 0.4232630179698842
end

```

The following code should be appended to `test-distributions.jl`:

```julia
@testset "Anka" begin
    ankamod = Anka(; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)
    @test total_func(ankamod, 2.3) ≈ 8.745018633265861
end

@testset "Frida" begin
    fridamod = Frida(;
        μ1 = 2.29,
        σ1 = 0.005,
        μ2 = 2.47,
        σ2 = 0.008,
        flat = 1.5,
        log_slope = 2.1,
        a1 = 5.0,
        a2 = 1.0,
    )
    @test total_func(fridamod, 2.3) ≈ 9.421676416183121
end
```

To test the helper function, make a new file `test-utils.jl`:

```julia
using Test
using DataAnalysisWS2425

@testset "Find zeros" begin
    xv = -0.0020005274189482786:0.00044456164865517303:0.0020005274189482786
    yv = [1.4510060494358186, 0.5284120618234738, -0.1098603023965552, -0.4946022192843884, -0.6522705923434842, -0.6060111053911896, -0.3764066548264964, 0.01795773002140777, 0.560081699474722, 1.2342658098241372]
    @test find_zero_two_sides(xv,yv) ≈ [-0.0011879226737373184,0.0010911606149442425]
end
```

</details>
