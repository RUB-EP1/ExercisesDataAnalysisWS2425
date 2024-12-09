### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 77ea602c-d9a7-4956-b702-331566c1b2cc
# ╠═╡ show_logs = false
begin
    using Pkg: Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    #
    using Plots
    using Parameters
    using QuadGK
    using ForwardDiff
    using Statistics
    using Optim
    using FHist
    using Measurements
    using DataAnalysisWS2425
    using DataFrames
end

# ╔═╡ a78f1cc1-9fe0-4969-9a13-32c9113a0b28
md"""
# Lecture 10a: Data Selection. MVA
"""

# ╔═╡ 2d463187-f2e4-44dc-b8df-70deba7c363b
theme(
    :wong2,
    grid = false,
    frame = :box,
    ylims = (0, :auto),
    xlims = (:auto, :auto),
    lab = "",
    linealpha = 1,
)

# ╔═╡ 86ee3836-2359-4eac-a66b-f714e3b5da72
const support = (0.5, 4.5);

# ╔═╡ f3affd46-4ca6-47cc-b188-b82396d6600f
const m_bins = range(support..., 100);

# ╔═╡ 63e12730-6d8c-4c92-9512-613ec878303d
const τ_bins = range(-4, 4, 50);

# ╔═╡ 049d77de-be06-4096-b94b-284ef77b5104
md"""
## Generate data
"""

# ╔═╡ 4a4a3bf4-d1f0-464d-b252-8e6dfd875acc
function generate_signal(nSignal)
    μ, σ = 2.3, 0.3
    m_signal = randn(nSignal) .* σ .+ μ
    τ_signal = (randn(nSignal) .* 2 .+ 1.0) .^ 3
    #
    NamedTuple{(:m, :τ)}.(zip(m_signal, τ_signal))
end

# ╔═╡ f35872fd-ba96-47a4-b8bd-1a5054d76db6
function generate_background(nBackground)
    m_background = rand(nBackground) .* (support[2] - support[1]) .+ support[1]
    τ_background = randn(nBackground) .- 1.0
    #
    NamedTuple{(:m, :τ)}.(zip(m_background, τ_background))
end

# ╔═╡ 5f959d38-e252-46eb-bd28-27965858a003
data = let
    nBackground = 10000
    nSignal = 700
    vcat(generate_signal(nSignal), generate_background(nBackground))
end |> DataFrame;

# ╔═╡ f1c73feb-7d48-4dcd-9ff5-20127fdf90be
data

# ╔═╡ 55f39b5c-9d6d-4fe0-849e-3481b2abfc1c
let
    bins = range(support..., 50)
    stephist(data.m; bins, xlab = "m [GeV]", title = "variable of interest")
end

# ╔═╡ ad356c5d-7c35-4531-9471-ff431b81b289
stephist(data.τ; bins = τ_bins, xlab = "τ", title = "discriminating variable")

# ╔═╡ fd517340-4938-4ba9-8fcf-3760a96e1528
md"""
## Training samples
"""

# ╔═╡ aed3e416-8173-4aa4-b47b-e15df842cdb3
nTrain = 10000

# ╔═╡ c30b5d69-a5cd-4ed8-84d1-bfb88c389d21
reference_τ_background = generate_background(nTrain) |> DataFrame;

# ╔═╡ fe9879c0-8ca5-4197-81fe-74c7a1bd11ac
reference_τ_signal = generate_signal(nTrain) |> DataFrame;

# ╔═╡ 98c19990-76a6-412f-8a8b-e7d4a0b4c853
begin
    plot(xlab = "τ")
    stephist!(
        reference_τ_background.τ,
        fill = 0;
        bins = τ_bins,
        α = 0.3,
        c = 2,
        lab = "Background",
    )
    stephist!(reference_τ_signal.τ, fill = 0; bins = τ_bins, α = 0.3, c = 3, lab = "Signal")
end

# ╔═╡ c6868022-2cab-48be-b2fc-cf03000223dc
md"""
## Receiver-operating characteristics (ROC) curve
"""

# ╔═╡ 65fac633-f17e-46f3-b1d8-71fd72f6fc1d
function p_value(sample, cut)
    sum(sample .> cut) / length(sample)
end

# ╔═╡ 6c00f2f4-ad50-4951-b504-022134042b55
efficiency(signal_sample, cut) = p_value(signal_sample, cut)

# ╔═╡ fd74415f-58a4-469d-82c3-17dd0682cb5f
rejection(background_sample, cut) = 1 - p_value(background_sample, cut)

# ╔═╡ 2e6cbf7a-05c4-4934-a0fd-abfe079fed42
ROC_values = map(range(-4, 4, 50)) do cut
    ϵ = efficiency(reference_τ_signal.τ, cut)
    r = rejection(reference_τ_background.τ, cut)
    (; ϵ, r, cut)
end |> DataFrame;

# ╔═╡ 8a046d2a-d9c8-4124-b467-ffede91ab482
let
    plot(
        ROC_values.ϵ,
        ROC_values.r,
        xlab = "signal efficiency",
        ylab = "background rejection",
        m = (3, :o),
        title = "ROC curve",
    )
    for cut in (-0.5, 0, 1)
        ϵ = efficiency(reference_τ_signal.τ, cut)
        r = rejection(reference_τ_background.τ, cut)
        scatter!([ϵ], [r], lab = "c=$cut", m = (10, :d))
    end
    plot!()
end

# ╔═╡ b1f30c83-c653-4c78-b53b-a32bb96b7ed5
md"""
## Figure of Merit
"""

# ╔═╡ 0153c3ad-e066-4367-9f46-4182303b8ce9
signal_significance(s, b) = s / sqrt(s + b)

# ╔═╡ bf9d1a36-6b34-48be-9db3-45cc49eeb780
purity(s, b) = s / (s + b)

# ╔═╡ a00d1416-6546-4929-b57b-dfa16da0e553
md"""
### Unoptimized sample
"""

# ╔═╡ b7c77080-659d-470d-9c87-a6a3ae948d1d
const h0 = Hist1D(data.m; binedges = m_bins);

# ╔═╡ 1dada9a7-d13c-4ad3-83aa-0fcaaf6a92b5
md"""
### Scan over the cut
"""

# ╔═╡ 1dc1d901-220f-4bf7-a1d9-eba9d8e68fd5
md"""
### Validation: fit parameters
"""

# ╔═╡ 3273ca89-e1b0-40f6-8426-8795d276c036
md"""
## Fitting utils
"""

# ╔═╡ 2e07e2fb-eb5b-4853-b0cb-612a1ee82ad7
function chi2(model_func, xv, yv)
    #
    δy = map(yv) do y
        y > 0 ? sqrt.(y) : 1.0
    end
    model_y = model_func.(xv)
    Δ = yv .- model_y
    χ² = sum(Δ .^ 2 ./ δy .^ 2)
    #
    return χ² / 1000
end

# ╔═╡ fe2feec5-6e4d-47cd-969c-550ed6a2f7a2
function fit_chi2(model, init_pars, h)
    xv, yv = bincenters(h), h.bincounts
    objective(p) = chi2(x -> model(x, p), xv, yv)
    fit_res = optimize(objective, init_pars, BFGS(); autodiff = :forward)
    pars = fit_res.minimizer
    H = ForwardDiff.hessian(objective, pars)
    (; minimizer = pars, H)
end

# ╔═╡ f7770a47-47a1-494e-ba0f-228efb93808e
function model(x, pars)
    μ, σ, aS, aB = pars
    return aS * exp(-(μ - x)^2 / (2σ^2)) + aB
end

# ╔═╡ ce3d879d-9262-41c7-86ec-0bbdf14c329b
fit_res0 = fit_chi2(model, [2.3, 0.6, 2.2, 1.2], h0)

# ╔═╡ 23688d29-e1bc-4de3-99a8-9723c0298f3a
begin
    plot(h0, c = 1, alpha = 0.2, seriestype = :stepbins)
    plot!(support...) do x
        model(x, fit_res0.minimizer)
    end
    best_pars0 = NamedTuple{(:μ, :σ, :aS, :aB)}(fit_res0.minimizer)
    vspan!(best_pars0.μ .+ [-2, 2] .* abs(best_pars0.σ), alpha = 0.3)
end

# ╔═╡ dc653645-f6ec-4bb0-aa48-caa6f2ca0f97
function computations(data, cut)
    _selected_data = data[data.τ.>cut, :]
    #
    _h = Hist1D(_selected_data.m; binedges = m_bins)
    mean_aB = mean(_h.bincounts[1:10])
    mean_aS = mean(_h.bincounts[40:50]) - mean_aB
    # @show mean_aB
    _fit_res = fit_chi2(model, [2.3, 0.3, mean_aS, mean_aB], _h)
    _best_pars = NamedTuple{(:μ, :σ, :aS, :aB)}(_fit_res.minimizer)
    (; cut, _selected_data, _best_pars, H = _fit_res.H)
end

# ╔═╡ 3c0d31d2-9f17-4616-ac7f-f4283532056e
computations_scan = map(-4.0:0.1:6) do cut
    computations(data, cut)
end |> DataFrame;

# ╔═╡ 3e9d58f6-5dd0-498c-a9ef-f79c62ee77b7
let
    μv = map(eachrow(computations_scan)) do row
        row._best_pars.μ ± inv(row.H)[1, 1]
    end
    σv = map(eachrow(computations_scan)) do row
        row._best_pars.σ ± inv(row.H)[2, 2]
    end
    plot(xlab = "τ critical", ylim = (0, 3))
    plot!(μv, m = (2, :o), markerstrokewidth = 1, mc = 2, c = 2, lab = "μ")
    plot!(σv, m = (2, :o), markerstrokewidth = 1, mc = 3, c = 3, lab = "σ")
end

# ╔═╡ 8740bc30-e293-4a6d-9a7f-a685f4af267b
function integral_signal_region(model, best_pars)
    μ, σ = best_pars
    σ_abs = abs(σ)
    return quadgk(x -> model(x, best_pars), μ - σ_abs, μ + σ_abs)[1]
end

# ╔═╡ d7752625-dee8-4590-81e8-3706daf3c7b7
begin
    nS_and_nB0 = integral_signal_region(model, best_pars0)
    nB0 = integral_signal_region(model, (; best_pars0..., aS = 0.0))
    nS0 = integral_signal_region(model, (; best_pars0..., aB = 0.0))
    @assert nS0 + nB0 ≈ nS_and_nB0
    (; nS = nS0, nB = nB0, significance = signal_significance(nS0, nB0))
end

# ╔═╡ 83539285-3fc3-45bc-9576-995adf518b95
function FOM(pars)
    nS_and_nB = integral_signal_region(model, pars)
    nB = integral_signal_region(model, (; pars..., aS = 0.0))
    nS = integral_signal_region(model, (; pars..., aB = 0.0))
    @assert nS + nB ≈ nS_and_nB
    signal_significance(nS, nB)
end

# ╔═╡ 753f1568-26f2-4d55-9882-2b3773809f23
begin
    xv = computations_scan.cut
    yv = FOM.(computations_scan._best_pars)
    plot(
        xv,
        yv,
        title = "optimization for FOM",
        xlab = "τ critical",
        ylab = "significance",
        ylim = (:auto, :auto),
    )
    #
    vline!([-0.5 0 1.0])
    plot!()
end

# ╔═╡ Cell order:
# ╟─a78f1cc1-9fe0-4969-9a13-32c9113a0b28
# ╠═77ea602c-d9a7-4956-b702-331566c1b2cc
# ╠═2d463187-f2e4-44dc-b8df-70deba7c363b
# ╠═86ee3836-2359-4eac-a66b-f714e3b5da72
# ╠═f3affd46-4ca6-47cc-b188-b82396d6600f
# ╠═63e12730-6d8c-4c92-9512-613ec878303d
# ╟─049d77de-be06-4096-b94b-284ef77b5104
# ╠═4a4a3bf4-d1f0-464d-b252-8e6dfd875acc
# ╠═f35872fd-ba96-47a4-b8bd-1a5054d76db6
# ╠═5f959d38-e252-46eb-bd28-27965858a003
# ╠═f1c73feb-7d48-4dcd-9ff5-20127fdf90be
# ╠═55f39b5c-9d6d-4fe0-849e-3481b2abfc1c
# ╠═ad356c5d-7c35-4531-9471-ff431b81b289
# ╟─fd517340-4938-4ba9-8fcf-3760a96e1528
# ╠═aed3e416-8173-4aa4-b47b-e15df842cdb3
# ╠═c30b5d69-a5cd-4ed8-84d1-bfb88c389d21
# ╠═fe9879c0-8ca5-4197-81fe-74c7a1bd11ac
# ╠═98c19990-76a6-412f-8a8b-e7d4a0b4c853
# ╟─c6868022-2cab-48be-b2fc-cf03000223dc
# ╠═65fac633-f17e-46f3-b1d8-71fd72f6fc1d
# ╠═6c00f2f4-ad50-4951-b504-022134042b55
# ╠═fd74415f-58a4-469d-82c3-17dd0682cb5f
# ╠═2e6cbf7a-05c4-4934-a0fd-abfe079fed42
# ╠═8a046d2a-d9c8-4124-b467-ffede91ab482
# ╟─b1f30c83-c653-4c78-b53b-a32bb96b7ed5
# ╠═0153c3ad-e066-4367-9f46-4182303b8ce9
# ╠═bf9d1a36-6b34-48be-9db3-45cc49eeb780
# ╟─a00d1416-6546-4929-b57b-dfa16da0e553
# ╠═b7c77080-659d-470d-9c87-a6a3ae948d1d
# ╠═ce3d879d-9262-41c7-86ec-0bbdf14c329b
# ╠═23688d29-e1bc-4de3-99a8-9723c0298f3a
# ╠═d7752625-dee8-4590-81e8-3706daf3c7b7
# ╟─1dada9a7-d13c-4ad3-83aa-0fcaaf6a92b5
# ╠═dc653645-f6ec-4bb0-aa48-caa6f2ca0f97
# ╠═3c0d31d2-9f17-4616-ac7f-f4283532056e
# ╠═83539285-3fc3-45bc-9576-995adf518b95
# ╠═753f1568-26f2-4d55-9882-2b3773809f23
# ╟─1dc1d901-220f-4bf7-a1d9-eba9d8e68fd5
# ╠═3e9d58f6-5dd0-498c-a9ef-f79c62ee77b7
# ╟─3273ca89-e1b0-40f6-8426-8795d276c036
# ╠═fe2feec5-6e4d-47cd-969c-550ed6a2f7a2
# ╠═2e07e2fb-eb5b-4853-b0cb-612a1ee82ad7
# ╠═f7770a47-47a1-494e-ba0f-228efb93808e
# ╠═8740bc30-e293-4a6d-9a7f-a685f4af267b
