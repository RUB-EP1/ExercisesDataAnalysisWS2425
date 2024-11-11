using Test
using DataAnalysisWS2425
using DataAnalysisWS2425.FHist

data = randn(1000)
h = Hist1D(data; binedges = range(-5, 5, 101))
@testset "WithData struct" begin
    wd = WithData(h)
    @test wd.factor ≈ 100
    @test wd.support == (-5.0, 5.0)
    wd = WithData(h.binedges[1], 100)
    @test wd.factor ≈ 10
end
