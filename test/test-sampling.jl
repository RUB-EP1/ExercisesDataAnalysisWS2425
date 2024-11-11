using Test
using DataAnalysisWS2425
using DataAnalysisWS2425.Random

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
