# Tutorial 2: Tests for Distributions and samples

For the following exercises and tutorials, you will be building up a codebase in your repository.

During this tutorial, you will copy over some functions from Exercise sheet 1, and modify them to pass the tests that you can find on the bottom of this sheet.

> [!TIP]
> Reminder on how to run tests in [the setup instructions](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/setup.md#back-to-julia-running-tests).

We will need some of the p.d.f.s in un-normalized form (such that the optimizations during fitting can work faster)

1. Gaussian
2. Polynomial
3. Relativistic Breit-Wigner
4. Voigt profile

To pass the tests, their signatures should look like this:

1. `gauss(x; μ, σ, a)`, where `a` is an arbitrary normalization factor
2. `poly(x; coeffs)`, where `coeffs` is a tuple of coefficients. Hint: implement the polynomials in a loop.
3. `rbw(x; M, Γ, a)`
4. `voigt(x; M, Γ, σ, a)`

We will also use the sampling methods and generalize them to take any function as input. Their signature looks like this:

```julia
function sample_rejection(f, n, support, nbins=1000)
# f is the function,
# n the sample size (passing the method),
# support is a tuple specifying in which range the function will be sampled
# nbins is the number of equidistant points in "support" for finding the maximum of the function (the default value is 1000)
```

```julia
function sample_inversion(f, n, support, nbins=1000)
# Similar to sample_rejection.
# Here, nbins is the number of bins in which the c.d.f. is pre-computed. See lecture-02-a for details.
```

Finally, you will be implementing a method to compute the correlation between two arrays:

```julia
correlation(A, B)
```

<details> <summary> To copy to `test/runtests.jl`</summary>
Here is the code you copy over to your `test/runtests.jl` file

```julia
using Test
using DataAnalysisWS2425
using DataAnalysisWS2425.QuadGK
using DataAnalysisWS2425.Random

# test the implementation of gauss
@testset "gaussian" begin
    @test gauss(1.1; μ=0.4, σ=0.7, a=1.) ≈ 0.6065306597126333
    @test gauss(2268.1; μ=2286.4, σ=7.0, a=1.0) ≈ 0.03280268530267093
end

# test the implementation of poly
@testset "polynomials" begin
    @test poly(1.3; coeffs=(1.1,0.5)) ≈ 1.75
    @test poly(1.3; coeffs=(0.0, -0.5, 0.3, 1.7)) ≈ 3.5919000000000008
end

# test the implementation of rbw
@testset "relativistic Breit-Wigner" begin
    @test rbw(1530.0; M=1532., Γ=9.0, a=1532.0^2) ≈ 0.010311498077081241
    @test rbw(11.3; M=12.0, Γ=0.3, a=144.0) ≈ 0.5161732492496677
end

# test the implementation of voigt
@testset "Voigt profile" begin
    @test voigt(1530.; M=1532.0, Γ=9.0, σ=6.0, a=1532.0) ≈ 0.10160430090139255
    @test voigt(4.2; M=4.3, Γ=0.1, σ=0.05, a=1.) ≈ 0.1952796435889611
end

# test the implementation of sample_rejection
@testset "Rejection sampling" begin
    Random.seed!(1234)
    @test sample_rejection(x->gauss(x; μ=2286.4, σ=7.,a=1.),3,(2240.,2330.)) ≈
        [2284.4824880201377, 2290.863082333516, 2296.4114519317136]
    @test sample_rejection(x->voigt(x; M=1532., Γ=9., σ=6., a=1532),2,(1500.,1560.)) ≈
        [1535.3323235714606, 1534.4594091166991]
end

# test the implementation of sample_inversion
@testset "Inversion sampling" begin
    Random.seed!(1234)
    @test sample_inversion(x->gauss(x; μ=0.4,σ=0.7,a=1.),4,(-4.,4.)) ≈
        [0.5438341871307295, 1.733853123918199, 0.4335500428402825, 1.1008379801314545]
    @test sample_inversion(x->rbw(x; M=1532., Γ=9., a=1532),3,(1500.,1560.)) ≈
        [1523.9607479415154, 1532.8935525470029, 1532.8572201887143]
end

# test the implementation of correlation
@testset "correlation" begin
    Random.seed!(1234)
    xv = rand(3)
    yv = randn(3)
    @test correlation(xv,yv) ≈ 0.6008493488865081
    @test correlation(xv,2*yv) ≈ correlation(xv,yv)
    @test correlation(xv,xv+yv) ≈ 0.7197187369786976
end
```

</details>
