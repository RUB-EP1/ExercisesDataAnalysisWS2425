using DataAnalysisWS2425
using Test

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
