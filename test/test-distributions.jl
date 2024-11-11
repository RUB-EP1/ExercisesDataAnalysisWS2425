using Test
using DataAnalysisWS2425

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
    @test breit_wigner_scaled(1530.0; M = 1532.0, Γ = 9.0, a = 1532.0^2) ≈
          0.010311498077081241
    @test breit_wigner_scaled(11.3; M = 12.0, Γ = 0.3, a = 144.0) ≈ 0.5161732492496677
end

# test the implementation of voigt_scaled
@testset "Voigt profile" begin
    @test voigt_scaled(1530.0; M = 1532.0, Γ = 9.0, σ = 6.0, a = 1532.0) ≈
          0.10160430090139255
    @test voigt_scaled(4.2; M = 4.3, Γ = 0.1, σ = 0.05, a = 1.0) ≈ 0.1952796435889611
end

# # code for visual instpection of sampling methods
# using Plots
# let
#     f(x) = exp(-x^4)
#     data = sample_rejection(f, 100_000, (-2.0, 2.0), 3)
#     bins = range(-2, 2, 400)
#     stephist(data; bins)
# end
