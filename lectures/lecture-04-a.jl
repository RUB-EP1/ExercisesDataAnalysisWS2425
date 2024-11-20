### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 2d141b9d-09bb-4074-bf01-4c4b6099585d
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    #
    using DataAnalysisWS2425
    #
    using Measurements
    using Plots
    using ForwardDiff
    using Plots.PlotMeasures: mm
    using Statistics
    using LinearAlgebra
    using Parameters
    using QuadGK
    using Random
    Random.seed!(1234)
    using FHist
    using Optim
    using DataFrames
end

# ╔═╡ 9da55708-8792-4b26-984f-5795a981bf2c
md"""
# Lecture 4a: Parameter uncertainties

In this lecture we discuss methods to estimate statistical uncertainty of parameters
"""

# ╔═╡ 39ee4bcd-8d01-443f-9714-103ab6d7f7d6
theme(
    :wong2,
    xlims = (:auto, :auto),
    ylims = (0, :auto),
    frame = :box,
    grid = false,
    lab = "",
    lw = 1.5,
)

# ╔═╡ 9b6b7d99-9f92-4b0a-b617-4111317e8271
const support = (2.2, 2.7); # fixed
# used in the notebook as a global constant
# will be const-propagated by a compiler

# ╔═╡ ace5e914-f516-438e-ae04-012573ad3586
md"""
## Model
"""

# ╔═╡ c29b061b-e19d-4107-b42a-ce906200456d
begin
    function signal_func(x, pars)
        @unpack μ, σ, a = pars
        gaussian_scaled(x; μ, σ, a)
    end
    function background_func(x, pars)
        @unpack flat, log_slope = pars
        x0 = sum(support) / 2
        #
        # flat is a value of bgd at support/2
        # make the slope independent of normalization
        # log_slope = dy/dx  / y = d log(y) / y
        #
        slope = flat * log_slope
        coeffs = (flat * 1 - slope * x0, slope)
        polynomial_scaled(x; coeffs)
    end
    model_func(x, pars) = signal_func(x, pars) + background_func(x, pars)
end

# ╔═╡ 54629edc-12ed-402d-bfc4-33cbb0b71848
const default = (; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)

# ╔═╡ 5e4a13fa-0929-46bc-a997-a303f5e44122
plot(x -> model_func(x, default), support...)

# ╔═╡ 8878c8be-21ba-4fc2-aa59-76754ebfbf0b
md"""
## Data
"""

# ╔═╡ 02b194bf-661f-44ee-94b8-9f2a21b3d219
pseudodata(pars, n) =
    sample_inversion(n, support) do x
        model_func(x, pars)
    end

# ╔═╡ d1b7527c-4375-4773-99b4-d039462b5044
const nData = 1_000

# ╔═╡ d1fdce14-0768-4838-b244-8b7fb101b137
const data = pseudodata(default, nData);

# ╔═╡ ca70935b-46e6-4e3b-9e53-0eb9d32c0fa7
md"""
Introduce a type that holds the model parameters
"""

# ╔═╡ 5f23fd6a-5950-45df-8136-3aa4a2bb1cf7
const ModelPars = typeof(default)

# ╔═╡ 20b68451-a639-44f0-97dd-28a89cf517cd
md"""
## Get Data
"""

# ╔═╡ 255e5019-f8e0-49ed-8efa-ba21004008aa
let
    bins = range(support..., 100)
    stephist(data; bins)
end

# ╔═╡ b26dae47-0eb7-4de3-aacd-844d866889ab
md"""
## Fitting
"""

# ╔═╡ ee73daf7-a8cd-42dc-a928-bd8f8bcf25a5
md"""
### Extended Maximum Likelihood method


The extended likelihood is defined with a poisson constrained for pdf integral,
```math
\mathcal{L}(p) = \frac{e^{-μ} μ^{n}}{n!} \prod_{i=1}^n P(x_i|p) = \frac{1}{(\int)^n} \prod_{i=1}^n f(x_i|p)
```

where $\mu = \int f(x|p) dx$.

The extended negative log likelihood (NLL) is similar to nll before, but without `log` for normalization
```math
\text{NNL}(p) = -\log \mathcal{L}(p) = - \sum_{i=1}^n \log f(x_i|p) + \int f(x|p) dx
```

"""

# ╔═╡ 28bf9f8e-567b-4857-91ac-1c6646556076
quadgk_call(f) = quadgk(f, support...)[1]

# ╔═╡ 22442c7f-f5b7-4b65-afcf-78a15585bdc3
function extended_nll(pars, data; normalization_call = quadgk_call)
    #
    minus_sum_log = -sum(data) do x
        value = model_func(x, pars)
        value > 0 ? log(value) : -1e10
    end
    #
    n = length(data)
    normalization = normalization_call() do x
        model_func(x, pars)
    end
    nll = minus_sum_log + normalization
    return nll
end

# ╔═╡ c72f2357-3d08-49ae-aae6-0f5f157db030
md"""
## Build initial guess
"""

# ╔═╡ 755a09d9-c491-439d-b933-1cd7cc29089e
initial_estimate = let
    μ = 2.35
    σ = 0.03
    #
    f_back = 0.7
    #
    I_back = nData * f_back
    flat = I_back / (support[2] - support[1])
    log_slope = 2.0
    #
    I_sig = nData * (1 - f_back)
    a = I_sig / sqrt(2π) / σ
    #
    ModelPars((; μ, σ, a, flat, log_slope))
end

# ╔═╡ 7f9e7439-59fe-4402-9738-ca8747299f84
let
    bins = range(support..., 50)
    h = Hist1D(data; binedges = bins)
    #
    plot(h, seriestype = :stepbins)
    #
    normalization = quadgk(support...) do x
        model_func(x, initial_estimate)
    end[1]
    dx = bins[2] - bins[1]
    scale = dx * nData / normalization
    #
    scaled_model(x) = scale * model_func(x, initial_estimate)
    p = plot!(scaled_model, support...)
end

# ╔═╡ b42b7e2e-df09-4273-8282-c294b30f095a
function fit_enll(data, initial_estimate; normalization_call = quadgk_call)
    objective(p) = extended_nll(ModelPars(p), data; normalization_call)
    optimize(objective, collect(initial_estimate), BFGS())
end

# ╔═╡ d7f6c8ab-232a-480f-b076-c2ee8fe83dd4
ext_unbinned_fit = fit_enll(data, initial_estimate)

# ╔═╡ f22353f4-ba2c-4808-b00e-028dfdd8a0c4
md"""
## Looking at the Results
"""

# ╔═╡ 1089442c-cfa3-4666-9270-999fc084ffe5
best_pars_extnll = ModelPars(ext_unbinned_fit.minimizer)

# ╔═╡ 2465c8bd-4f47-4f93-b48c-42bd04bc23b9
let
    bins = range(support..., 70)
    h = Hist1D(data; binedges = bins)
    plot(h, seriestype = :stepbins)
    #
    normalization = quadgk(support...) do x
        model_func(x, best_pars_extnll)
    end[1]
    dx = bins[2] - bins[1]
    n = length(data)
    scale = dx * n / normalization
    #
    scaled_model(x) = scale * model_func(x, best_pars_extnll)
    plot!(scaled_model, support...)
    plot!(x -> scale * signal_func(x, best_pars_extnll), support..., fill = 0, alpha = 0.4)
    #
    # add pull
    p = plot!(xaxis = nothing)
    centers = (bins[1:end-1] + bins[2:end]) ./ 2
    yv_model = scaled_model.(centers)
    scatter(
        centers,
        h.bincounts .- yv_model,
        ylims = (:auto, :auto),
        xerror = (bins[2] - bins[1]) / 2,
        yerror = sqrt.(h.bincounts),
        ms = 2,
    )
    pull = hline!([0], lc = 2)
    plot(
        p,
        pull,
        layout = grid(2, 1, heights = (0.8, 0.2)),
        link = :x,
        bottom_margin = [-4mm 0mm],
    )
end

# ╔═╡ a05ea1e6-9667-4f73-b4ca-fc5ffccbb84b
# extended nll
best_yields = let # model integral is fixed to N_data
    nSignal = quadgk(x -> signal_func(x, best_pars_extnll), support...)[1]
    nBackground = quadgk(x -> background_func(x, best_pars_extnll), support...)[1]
    (; nSignal, nBackground)
end

# ╔═╡ dc0082f0-546f-425d-8fc3-fcf969b5fade
md"""
## Running pseudoexperiments
"""

# ╔═╡ 133057a5-64cb-4eeb-885b-07fec21c745d
toys = map(1:100) do _
    _data = pseudodata(default, nData)
    _fit_result = ext_unbinned_fit = fit_enll(_data, initial_estimate)
    #
    return ModelPars(_fit_result.minimizer)
end |> DataFrame

# ╔═╡ 1be765bf-7c78-4bfd-9079-1a3182a9b4ad
toys.nSignal = map(eachrow(toys)) do pars
    quadgk(x -> signal_func(x, pars), support...)[1]
end;

# ╔═╡ 44d04cbc-695f-45f9-b31b-90c28d606891
transform!(toys, :σ => ByRow(abs) => :σ);

# ╔═╡ 8bcdff59-9157-498b-8e83-f271360ee7ff
toys

# ╔═╡ a3bd550e-a940-46c5-95a7-2c92baa4e8e0
normal_bins(s) = range((mean(s) .+ std(s) .* [-3, 3])..., 30)

# ╔═╡ e5482cbf-87f6-4942-80ce-6b4c9ebfb3ab
default_nSignal = let
    Is = quadgk(x -> signal_func(x, default), support...)[1]
    Im = quadgk(x -> model_func(x, default), support...)[1]
    Is / Im * nData
end

# ╔═╡ 8e0d5554-eacd-40cc-9e26-ab6c4fed8cef
let
    plot(
        title = ["μ" "σ" "nSignal"],
        stephist(toys.μ, bins = normal_bins(toys.μ)),
        stephist(toys.σ .|> abs, bins = normal_bins(toys.σ)),
        stephist(toys.nSignal, bins = normal_bins(toys.nSignal)),
    )
    #
    vline!(
        sp = 1,
        [default.μ best_pars_extnll.μ],
        lab = ["default" "fit"],
        lc = [:gray :red],
        lw = [1 2],
    )
    #
    vline!(
        sp = 2,
        [default.σ abs(best_pars_extnll.σ)],
        lab = ["default" "fit"],
        lc = [:gray :red],
        lw = [1 2],
    )
    #
    vline!(
        sp = 3,
        [default_nSignal best_yields.nSignal],
        lab = ["default" "fit"],
        lc = [:gray :red],
        lw = [1 2],
    )
end

# ╔═╡ 21b32675-48f3-40a9-b086-f4842b9fae0e
stat_toys = combine(toys, All() .=> std .=> "δ" .* names(toys)) |> first |> NamedTuple

# ╔═╡ 5e997750-3f04-49e1-a083-dc47338be149
md"""
## MC technique to compute normalization
"""

# ╔═╡ 6d36c5fb-591e-468d-b1b7-e3e8f5727b1c
const nMC = nData * 10;

# ╔═╡ 334ab41b-fb0e-4a1f-8b30-3bafd281d525
const data_mc = support[1] .+ rand(nMC) .* (support[2] - support[1]);

# ╔═╡ bd1417a4-6101-495b-a47f-b1c045ea8fb8
mc_call(f) = (support[2] - support[1]) * mean(f, data_mc)

# ╔═╡ 36584517-4416-4c39-8816-75db05b9a495
# notice the time, compare to `quadgk_call`
ext_unbinned_fit_mc = fit_enll(data, initial_estimate; normalization_call = mc_call)

# ╔═╡ 1ba5cacd-ce13-4707-b27d-dedf2b51ccbc
best_pars_extnll_mc = ModelPars(ext_unbinned_fit_mc.minimizer);

# ╔═╡ 8fafefef-940e-4b7e-998f-57773a3f79f5
[
    (; normalization_call = :quadgk, best_pars_extnll...),
    (; normalization_call = :mc, best_pars_extnll_mc...),
] |> DataFrame

# ╔═╡ 29d39a85-9ea6-4fe8-b71a-ac099e323ef4
md"""
## Matrix of derivatives

Using the second order (gaussian) approximation of the nll minimum,
covariance matrix can be computed as,

```math
V = H^{-1} = \begin{pmatrix}
σ_1^2 & v_{1,2} & \dots \\
v_{1,2} & σ_2^2 & \dots \\
\dots & \dots & \dots
\end{pmatrix}
```

the diagonal of the matrix gives the statistical errors.
"""

# ╔═╡ 0a0c877a-6d10-4cbb-aa61-baf4a991627d
const AnyModelPars = NamedTuple{(fieldnames(ModelPars))}

# ╔═╡ 6bf3164c-8ffa-44c3-a27e-5450232a7003
## Gradient should be zero in the proper minimum
▽nll = ForwardDiff.gradient(
    p -> extended_nll(AnyModelPars(p), data; normalization_call = mc_call),
    collect(best_pars_extnll_mc),
)

# ╔═╡ 555ccd0e-a14a-40ab-976f-19e26665312c
H_mc = ForwardDiff.hessian(
    p -> extended_nll(AnyModelPars(p), data; normalization_call = mc_call),
    collect(best_pars_extnll_mc),
)

# ╔═╡ 758ab556-38c8-4279-8f76-1e02ffce15b5
from_hesse = let
    names = fieldnames(ModelPars)
    delta_names = "δ" .* string.(names)
    NamedTuple{Symbol.(delta_names)}(sqrt.(diag(inv(H_mc))))
end

# ╔═╡ 3e2a4cef-684e-4de0-a5c2-5569bee217b9
md"""
Here is a comparison of the statistical errors from pseudoexperiments, and these coming from hessian.
"""

# ╔═╡ 785039e8-9079-495e-b9b7-1515f2857435
[(method = :toys, stat_toys...), (method = :hesse, from_hesse..., δnSignal = missing)] |>
DataFrame

# cspell:disable

# ╔═╡ Cell order:
# ╟─9da55708-8792-4b26-984f-5795a981bf2c
# ╠═2d141b9d-09bb-4074-bf01-4c4b6099585d
# ╠═39ee4bcd-8d01-443f-9714-103ab6d7f7d6
# ╠═9b6b7d99-9f92-4b0a-b617-4111317e8271
# ╟─ace5e914-f516-438e-ae04-012573ad3586
# ╠═c29b061b-e19d-4107-b42a-ce906200456d
# ╠═54629edc-12ed-402d-bfc4-33cbb0b71848
# ╠═5e4a13fa-0929-46bc-a997-a303f5e44122
# ╟─8878c8be-21ba-4fc2-aa59-76754ebfbf0b
# ╠═02b194bf-661f-44ee-94b8-9f2a21b3d219
# ╠═d1b7527c-4375-4773-99b4-d039462b5044
# ╠═d1fdce14-0768-4838-b244-8b7fb101b137
# ╟─ca70935b-46e6-4e3b-9e53-0eb9d32c0fa7
# ╠═5f23fd6a-5950-45df-8136-3aa4a2bb1cf7
# ╟─20b68451-a639-44f0-97dd-28a89cf517cd
# ╠═255e5019-f8e0-49ed-8efa-ba21004008aa
# ╟─b26dae47-0eb7-4de3-aacd-844d866889ab
# ╟─ee73daf7-a8cd-42dc-a928-bd8f8bcf25a5
# ╠═28bf9f8e-567b-4857-91ac-1c6646556076
# ╠═22442c7f-f5b7-4b65-afcf-78a15585bdc3
# ╟─c72f2357-3d08-49ae-aae6-0f5f157db030
# ╠═755a09d9-c491-439d-b933-1cd7cc29089e
# ╠═7f9e7439-59fe-4402-9738-ca8747299f84
# ╠═b42b7e2e-df09-4273-8282-c294b30f095a
# ╠═d7f6c8ab-232a-480f-b076-c2ee8fe83dd4
# ╠═2465c8bd-4f47-4f93-b48c-42bd04bc23b9
# ╟─f22353f4-ba2c-4808-b00e-028dfdd8a0c4
# ╠═1089442c-cfa3-4666-9270-999fc084ffe5
# ╠═a05ea1e6-9667-4f73-b4ca-fc5ffccbb84b
# ╟─dc0082f0-546f-425d-8fc3-fcf969b5fade
# ╠═133057a5-64cb-4eeb-885b-07fec21c745d
# ╠═1be765bf-7c78-4bfd-9079-1a3182a9b4ad
# ╠═44d04cbc-695f-45f9-b31b-90c28d606891
# ╠═8bcdff59-9157-498b-8e83-f271360ee7ff
# ╠═a3bd550e-a940-46c5-95a7-2c92baa4e8e0
# ╠═e5482cbf-87f6-4942-80ce-6b4c9ebfb3ab
# ╠═8e0d5554-eacd-40cc-9e26-ab6c4fed8cef
# ╠═21b32675-48f3-40a9-b086-f4842b9fae0e
# ╟─5e997750-3f04-49e1-a083-dc47338be149
# ╠═6d36c5fb-591e-468d-b1b7-e3e8f5727b1c
# ╠═334ab41b-fb0e-4a1f-8b30-3bafd281d525
# ╠═bd1417a4-6101-495b-a47f-b1c045ea8fb8
# ╠═36584517-4416-4c39-8816-75db05b9a495
# ╠═1ba5cacd-ce13-4707-b27d-dedf2b51ccbc
# ╟─8fafefef-940e-4b7e-998f-57773a3f79f5
# ╟─29d39a85-9ea6-4fe8-b71a-ac099e323ef4
# ╠═0a0c877a-6d10-4cbb-aa61-baf4a991627d
# ╠═6bf3164c-8ffa-44c3-a27e-5450232a7003
# ╠═555ccd0e-a14a-40ab-976f-19e26665312c
# ╠═758ab556-38c8-4279-8f76-1e02ffce15b5
# ╟─3e2a4cef-684e-4de0-a5c2-5569bee217b9
# ╠═785039e8-9079-495e-b9b7-1515f2857435
