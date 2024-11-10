using Test
using DataAnalysisWS2425

@testset "Simple fitting" begin
    init_pars = (; μ = 0.35, σ = 0.8, a = 1.0)
    support = (-4.0, 4.0)
    data = sample_inversion(400, support) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    model(x, pars) = gaussian_scaled(x; pars.μ, pars.σ, pars.a)
    ext_unbinned_fit = fit_enll(model, init_pars, data; support = support)
    best_pars_extnll = typeof(init_pars)(ext_unbinned_fit.minimizer)
    @test ext_unbinned_fit.ls_success
    @test 0.36 < best_pars_extnll.μ < 0.44
end
