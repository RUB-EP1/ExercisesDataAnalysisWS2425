# Tutorial 2: Tests for Distributions and samples

For the following exercises and tutorials, you will be building up a codebase in your repository.

During this tutorial, you will copy over some functions from Exercise sheet 1, and modify them to pass the tests that you can find on the bottom of this sheet.

> [!TIP]
> Reminder on how to run tests in [the setup instructions](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/setup.md#back-to-julia-running-tests).
> Also, to make the functions globally available, you need to export the functions in the src file: `export gaussian_parametrized, poly, ...`.

We will need some p.d.f.s in a form with arbitrary normalization (such that the optimizations during fitting can work faster)

1. Gaussian
2. Polynomial
3. Relativistic Breit-Wigner
4. Voigt profile

To pass the tests, their signatures should look like this:

1. `gaussian_parametrized(x; μ, σ, a)`, where `a` is an arbitrary normalization factor
2. `poly(x; coeffs)`, where `coeffs` is a tuple of coefficients. Hint: implement the polynomials in a loop.
3. `relativistic_breit_wigner(x; M, Γ, a)`
4. `voigt(x; M, Γ, σ, a)`

We will also use the sampling methods and generalize them to take any function as input. Their signature looks like this:

```julia
function sample_rejection(f, n, support, nbins=1000)
"""
Generates `n` samples using the rejection sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional. The number of equidistant points in `support` to find the maximum of `f`. Default is `1000`.

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
data = sample_rejection(exp, 10, (0, 4))
"""
```

```julia
function sample_inversion(f, n, support, nbins=1000)
"""
Generates `n` samples using the inversion sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional. The number of equidistant points in `support` for which the c.d.f. is pre-computed. Default is `1000`.

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
data = sample_inversion(exp, 10, (0, 4))
```

<details> <summary> To copy to `test/runtests.jl`</summary>
Here is the code you copy over to your `test/runtests.jl` file

```julia
using Test
using DataAnalysisWS2425
using DataAnalysisWS2425.QuadGK
using DataAnalysisWS2425.Random

# test the implementation of gaussian_parametrized
@testset "gaussian" begin
    @test gaussian_parametrized(1.1; μ = 0.4, σ = 0.7, a = 1.0) ≈ 0.6065306597126333
    @test gaussian_parametrized(2268.1; μ = 2286.4, σ = 7.0, a = 1.0) ≈ 0.03280268530267093
end

# test the implementation of poly
@testset "polynomials" begin
    @test poly(1.3; coeffs = (1.1, 0.5)) ≈ 1.75
    @test poly(1.3; coeffs = (0.0, -0.5, 0.3, 1.7)) ≈ 3.5919
end

# test the implementation of relativistic_breit_wigner
@testset "relativistic Breit-Wigner" begin
    @test relativistic_breit_wigner(1530.0; M = 1532.0, Γ = 9.0, a = 1532.0^2) ≈ 0.010311498077081241
    @test relativistic_breit_wigner(11.3; M = 12.0, Γ = 0.3, a = 144.0) ≈ 0.5161732492496677
end

# test the implementation of voigt
@testset "Voigt profile" begin
    @test voigt(1530.0; M = 1532.0, Γ = 9.0, σ = 6.0, a = 1532.0) ≈ 0.10160430090139255
    @test voigt(4.2; M = 4.3, Γ = 0.1, σ = 0.05, a = 1.0) ≈ 0.1952796435889611
end

# test the implementation of sample_rejection
@testset "Rejection sampling" begin
    Random.seed!(1234)
    @test sample_rejection(
        x -> gaussian_parametrized(x; μ = 2286.4, σ = 7.0, a = 1.0),
        3,
        (2240.0, 2330.0),
    ) ≈ [2284.4824880201377, 2290.863082333516, 2296.4114519317136]
    @test sample_rejection(
        x -> voigt(x; M = 1532.0, Γ = 9.0, σ = 6.0, a = 1532),
        2,
        (1500.0, 1560.0),
    ) ≈ [1535.3323235714606, 1534.4594091166991]
end

# test the implementation of sample_inversion
@testset "Inversion sampling" begin
    Random.seed!(1234)
    @test sample_inversion(x -> gaussian_parametrized(x; μ = 0.4, σ = 0.7, a = 1.0), 4, (-4.0, 4.0)) ≈
          [0.5438341871307295, 1.733853123918199, 0.4335500428402825, 1.1008379801314545]
    @test sample_inversion(
        x -> relativistic_breit_wigner(x; M = 1532.0, Γ = 9.0, a = 1532),
        3,
        (1500.0, 1560.0),
    ) ≈ [1523.9607479415154, 1532.8935525470029, 1532.8572201887143]
end

```

</details>
