abstract type SpectrumModel end
Base.collect(model::SpectrumModel) =
    getproperty.(model |> Ref, collect(fieldnames(typeof(model))))
#
# Define a generic constructor for any subtype of SpectrumModel
function (::Type{T})(p_values::Union{AbstractVector,NTuple}) where {T<:SpectrumModel}
    T(; NamedTuple{fieldnames(T)}(p_values)...)
end

# Common to many models


"""
    pol1_with_logs_slope(x, pars; x0::Float64 = 0.0)

Evaluates a linear polynomial with a logarithmic slope at a given `x`, using parameters provided in `pars`.

The polynomial is defined as:

    y = (flat - slope * x0) + slope * x

where:
- `flat` and `log_slope` are unpacked from the `pars` object.
- `slope` is computed as `flat * log_slope`.
- `x0` is an optional shift parameter, defaulting to `0.0`.

This function internally calls `polynomial_scaled` with coefficients derived from the `flat`, `log_slope`, and `x0`.

# Parameters:
- `x`: The value at which the polynomial is evaluated.
- `pars`: A structure containing:
  - `flat`: The base level of the polynomial.
  - `log_slope`: A scaling factor for the slope.
- `x0::Float64`: (Optional) A shift for the polynomial's origin. Defaults to `0.0`.

# Example:
```julia
julia> pars = (flat = 2.0, log_slope = 0.5)
julia> y = pol1_with_logs_slope(1.0, pars; x0 = 0.0)
```
"""
function pol1_with_logs_slope(x, pars; x0::Float64 = 0.0)
    @unpack flat, log_slope = pars
    slope = flat * log_slope
    coeffs = (flat - slope * x0, slope)
    polynomial_scaled(x; coeffs)
end




"""
    Anka{P} <: SpectrumModel

A simple spectral model combining a Gaussian peak with a linear background.

The `Anka` model has two components:
1. A **peaking signal** represented by a Gaussian:
   - Parameters: `μ` (mean), `σ` (standard deviation), and `a` (amplitude).
2. A **background** modeled as a linear polynomial with a logarithmic slope:
   - Parameters: `flat` (intercept) and `log_slope` (slope scaling factor).

# Parameters:
- `μ::P`: Mean of the Gaussian peak.
- `σ::P`: Standard deviation of the Gaussian peak.
- `a::P`: Amplitude of the Gaussian peak.
- `flat::P`: Intercept for the linear background.
- `log_slope::P`: Scaling factor for the slope of the background.

# Functions:
- `peak1_func(model::Anka, x)`: Evaluates the Gaussian component at `x`.
- `background_func(model::Anka, x)`: Evaluates the background component at `x` using a polynomial with logarithmic slope.
- `total_func(model::Anka, x)`: Evaluates the sum of the Gaussian and background components at `x`.

# Example:
```julia
julia> model = Anka(; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)
julia> total_func(model, 3.3)
4.1775
```
"""
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

background_func(model::Anka, x) = pol1_with_logs_slope(x, model)

total_func(model::Anka, x) = peak1_func(model, x) + background_func(model, x)


# Frida comes

"""
    Frida{P} <: SpectrumModel

A spectral model combining two Gaussian peaks with a linear background.

The `Frida` model is composed of:
1. **Two peaking signals** represented by Gaussian functions:
   - Parameters for the first peak: `μ1` (mean), `σ1` (standard deviation), and `a1` (amplitude).
   - Parameters for the second peak: `μ2` (mean), `σ2` (standard deviation), and `a2` (amplitude).
2. **A background** modeled as a linear polynomial with a logarithmic slope:
   - Parameters: `flat` (intercept) and `log_slope` (slope scaling factor).

# Parameters:
- `μ1::P`, `μ2::P`: Means of the two Gaussian peaks.
- `σ1::P`, `σ2::P`: Standard deviations of the two Gaussian peaks.
- `a1::P`, `a2::P`: Amplitudes of the two Gaussian peaks.
- `flat::P`: Intercept for the linear background.
- `log_slope::P`: Scaling factor for the slope of the background.

# Functions:
- `peak1_func(model::Frida, x)`: Evaluates the first Gaussian component at `x`.
- `peak2_func(model::Frida, x)`: Evaluates the second Gaussian component at `x`.
- `background_func(model::Frida, x)`: Evaluates the background component at `x` using a polynomial with logarithmic slope.
- `total_func(model::Frida, x)`: Evaluates the sum of the two Gaussian peaks and the background at `x`.

# Example:
```julia
julia> model = Frida(; μ1 = 2.29, σ1 = 0.005, μ2 = 2.47, σ2 = 0.008, flat = 1.5, log_slope = 2.1, a1 = 5.0, a2 = 1.0)
julia> total_func(model, 3.3)
11.895
```
"""
@with_kw struct Frida{P} <: SpectrumModel
    μ1::P
    σ1::P
    a1::P
    μ2::P
    σ2::P
    a2::P
    flat::P
    log_slope::P
end
#
function peak1_func(model::Frida, x)
    @unpack μ1, σ1, a1 = model
    gaussian_scaled(x; μ = μ1, σ = σ1, a = a1)
end

function peak2_func(model::Frida, x)
    @unpack μ2, σ2, a2 = model
    gaussian_scaled(x; μ = μ2, σ = σ2, a = a2)
end

background_func(model::Frida, x) = pol1_with_logs_slope(x, model)

total_func(model::Frida, x) =
    peak1_func(model, x) + peak2_func(model, x) + background_func(model, x)
