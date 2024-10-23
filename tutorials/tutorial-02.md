# Tutorial 2: Tests for Distributions and samples

For the following exercises and tutorials, you will be building up a codebase in your repository.

During this tutorial, you will copy over some functions from Exercise sheet 1, and modify them to pass the tests that you can find on the bottom of this sheet.

> [!TIP]
> Reminder on how to run tests in [the setup instructions](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/setup.md#back-to-julia-running-tests).
> Also, to make the functions globally available, you need to export the functions in the src file: `export gaussian_scaled`.

## Functions

To describe the particle spectra,
will need set of functions in a form with arbitrary normalization (such that the optimizations during fitting can work faster)

1. Gaussian
2. Polynomial
3. Breit-Wigner
4. Voigt profile

We will use a suffix `_scaled` to indicate that the function is not normalized to identity, but has a normalization factor, given by a parameter `a`.
The docstrings below should be copied before the function definition in the `src` file to enable the language server to discover them.

> [!NOTE]
> The docstring fixes the function signature and provides a description of the function, its arguments, and return values.
> Also, that is what our testset expects from the function.


### Gaussian function

````julia
"""
    gaussian_scaled(x; μ, σ, a=1.0)

Computes the value of a Gaussian function with flexible normalization at `x`, given the mean `μ`, standard deviation `σ`, and scaling factor `a`.

The form of the Gaussian is:

    a * exp(-((x - μ)^2) / (2 * σ^2))

# Example
```julia
julia> y = gaussian_scaled(2.0; μ=0.0, σ=1.0, a=3.0)
```
"""
````

### Polynomial

Next, we have the polynomial function,

````julia
"""
    polynomial_scaled(x; coeffs)

Evaluates a polynomial at `x`, given the coefficients in `coeffs`.
The `coeffs` is an iterable collection of coefficients, where the first element corresponds to the lowest degree term.

The polynomial has the form:

    coeffs[1] * x^0 + coeffs[1] * x^(2) + ... + coeffs[n] * x^(n-1)

where `n-1` is the degree of the polynomial, determined by the length of `coeffs`.

# Example
```julia
julia> y = polynomial_scaled(2.0; coeffs=[1.0, -3.0, 2.0])
```
"""
````

### Breit-Wigner

The relativistic Breit-Wigner function is given by,

````julia
"""
    breit_wigner_scaled(x; M, Γ, a=1.0)

Computes the value of a Breit-Wigner function with flexible normalization at `x`, given the mass `m`, width `Γ`, and scaling factor `a`.

The form of the Breit-Wigner is:

    a / |m^2 - x^2 - imΓ|^2

# Example
```julia
julia> y = breit_wigner_scaled(2.0; m=1.0, Γ=0.5, a=2.0)
```
"""
````

### Voigt

Finally, we have the Voigt profile function,

````julia
"""
    voigt_scaled(x; m=0.0, Γ=1.0, σ=1.0, a=1.0)

Computes the value of a Voigt profile with flexible normalization at `x`, given the peak position `m`, Breit-Wigner width `Γ`, Gaussian width `σ`, and scaling factor `a`.

The Voigt profile is a convolution of a non-relativistic Breit-Wigner function and a Gaussian, commonly used to describe spectral lineshapes.

# Example
```julia
julia> y = voigt_scaled(2.0; m=1.3, Γ=0.15, σ=0.3, a=3.0)
```
"""
````


## Sampling methods

We will also use the sampling methods and generalize them to take any function as input. Their signature looks like this:

````julia
"""
    sample_rejection(f, n, support; nbins=1000)

Generates `n` samples using the rejection sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional keyword argument. The number of equidistant points in `support` used to search for the maximum of `f`

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
```julia
julia> data = sample_rejection(exp, 10, (0, 4))
```
"""
````

````julia
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
````

<details> <summary> To copy to `test/runtests.jl`</summary>
Here is the code you copy over to your `test/runtests.jl` file

```julia
using Test
using DataAnalysisWS2425
using DataAnalysisWS2425.QuadGK
using DataAnalysisWS2425.Random

# test the implementation of gaussian_scaled
@testset "gaussian" begin
    @test gaussian_scaled(1.1; μ = 0.4, σ = 0.7, a = 1.0) ≈ 0.6065306597126333
    @test gaussian_scaled(2268.1; μ = 2286.4, σ = 7.0, a = 1.0) ≈ 0.03280268530267093
end

# test the implementation of polynomial_scaled
@testset "polynomials" begin
    @test polynomial_scaled(1.3; coeffs = (1.1, 0.5)) ≈ 1.75
    @test polynomial_scaled(1.3; coeffs = (0.0, -0.5, 0.3, 1.7)) ≈ 3.5919
end

# test the implementation of breit_wigner_scaled
@testset "Relativistic Breit-Wigner" begin
    @test breit_wigner_scaled(1530.0; M = 1532.0, Γ = 9.0, a = 1532.0^2) ≈ 0.010311498077081241
    @test breit_wigner_scaled(11.3; M = 12.0, Γ = 0.3, a = 144.0) ≈ 0.5161732492496677
end

# test the implementation of voigt_scaled
@testset "Voigt profile" begin
    @test voigt_scaled(1530.0; M = 1532.0, Γ = 9.0, σ = 6.0, a = 1532.0) ≈ 0.10160430090139255
    @test voigt_scaled(4.2; M = 4.3, Γ = 0.1, σ = 0.05, a = 1.0) ≈ 0.1952796435889611
end

# # code for visual inspection of sampling methods
# using Plots
# let
#     f(x) = exp(-x^4)
#     data = sample_rejection(f, 100_000, (-2.0, 2.0), 3)
#     bins = range(-2, 2, 400)
#     stephist(data; bins)
# end

@testset "Rejection sampling" begin
    Random.seed!(1234)
    #
    g(x) = gaussian_scaled(x; μ = 2286.4, σ = 7.0, a = 1.0)
    data_g = sample_rejection(g, 3, (2240.0, 2330.0))
    @test data_g isa Vector
    @test length(data_g) == 3
    #
    v(x) = voigt_scaled(x; M = 1532.0, Γ = 9.0, σ = 6.0, a = 1532)
    data_v = sample_rejection(v, 2, (1500.0, 1560.0))
    @test data_v isa Vector
    @test length(data_v) == 2
end

@testset "Inversion sampling" begin
    data_g = sample_inversion(4, (-4.0, 4.0)) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    @test data_g isa Vector
    @test length(data_g) == 4
end

```

</details>
