### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2d141b9d-09bb-4074-bf01-4c4b6099585d
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__,"..", "ReferenceDataAnalysisWS2425"))
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
	using LikelihoodProfiler
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
		slope = flat*log_slope
		coeffs = (flat*1-slope*x0, slope)
        polynomial_scaled(x; coeffs)
    end
    model_func(x, pars) = signal_func(x, pars) + background_func(x, pars)
end

# ╔═╡ 54629edc-12ed-402d-bfc4-33cbb0b71848
const default = (; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)

# ╔═╡ 5e4a13fa-0929-46bc-a997-a303f5e44122
plot(x->model_func(x,default), support...)

# ╔═╡ 8878c8be-21ba-4fc2-aa59-76754ebfbf0b
md"""
## Data
"""

# ╔═╡ 02b194bf-661f-44ee-94b8-9f2a21b3d219
pseudodata(pars, n) = sample_inversion(n, support) do x
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

# ╔═╡ 0a0c877a-6d10-4cbb-aa61-baf4a991627d
const AnyModelPars = NamedTuple{(fieldnames(ModelPars))}

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
	I_back = nData*f_back
	flat = I_back / (support[2]-support[1])
	log_slope = 2.0
	# 
	I_sig = nData*(1-f_back)
	a = I_sig/sqrt(2π)/σ
	# 
	ModelPars((; μ, σ, a, flat, log_slope))
end

# ╔═╡ 7f9e7439-59fe-4402-9738-ca8747299f84
let
    bins = range(support..., 50)
    h = Hist1D(data; binedges=bins)
	# 
	plot(h, seriestype=:stepbins)
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

# ╔═╡ 5e997750-3f04-49e1-a083-dc47338be149
md"""
## MC technique to compute normalization
"""

# ╔═╡ 6d36c5fb-591e-468d-b1b7-e3e8f5727b1c
const nMC = nData * 10;

# ╔═╡ 334ab41b-fb0e-4a1f-8b30-3bafd281d525
const data_mc = support[1] .+ rand(nMC) .* (support[2]-support[1]);

# ╔═╡ bd1417a4-6101-495b-a47f-b1c045ea8fb8
mc_call(f) = (support[2]-support[1]) * mean(f, data_mc)

# ╔═╡ 22442c7f-f5b7-4b65-afcf-78a15585bdc3
function extended_nll(pars, data; normalization_call = mc_call)
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

# ╔═╡ 926fb15c-7f10-44af-aafd-c6a9c065502e
extended_nll(p::Vector) = extended_nll(
	AnyModelPars(p), data; normalization_call=mc_call)

# ╔═╡ b42b7e2e-df09-4273-8282-c294b30f095a
function fit_enll(data, initial_estimate; normalization_call=mc_call)
    objective(p) = extended_nll(ModelPars(p), data; normalization_call)
    optimize(
        objective,
        collect(initial_estimate),
        BFGS(),
    )
end

# ╔═╡ 36584517-4416-4c39-8816-75db05b9a495
ext_unbinned_fit_mc = fit_enll(data, initial_estimate)

# ╔═╡ 1ba5cacd-ce13-4707-b27d-dedf2b51ccbc
best_pars_extnll_mc = ModelPars(ext_unbinned_fit_mc.minimizer);

# ╔═╡ 2465c8bd-4f47-4f93-b48c-42bd04bc23b9
let
    bins = range(support..., 70)
    h = Hist1D(data; binedges=bins)
	plot(h, seriestype=:stepbins)
    #
    normalization = quadgk(support...) do x
        model_func(x, best_pars_extnll_mc)
    end[1]
    dx = bins[2] - bins[1]
    n = length(data)
    scale = dx * n / normalization
	# 
	scaled_model(x) = scale * model_func(x, best_pars_extnll_mc)
    plot!(scaled_model, support...)
	plot!(x -> scale * signal_func(x, best_pars_extnll_mc), support...,
		fill = 0, alpha = 0.4)
	#
	# add pull
	p = plot!(xaxis=nothing)
	centers = (bins[1:end-1] + bins[2:end]) ./ 2
	yv_model = scaled_model.(centers)
	scatter(centers, h.bincounts .- yv_model, ylims=(:auto,:auto),
		xerror=(bins[2]-bins[1])/2, yerror=sqrt.(h.bincounts), ms=2)
	pull = hline!([0], lc=2)
	plot(p, pull,
		layout=grid(2,1, heights=(0.8,0.2)),
		link=:x, bottom_margin=[-4mm 0mm])
end

# ╔═╡ 29d39a85-9ea6-4fe8-b71a-ac099e323ef4
md"""
## Hessian and Varaince

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

# ╔═╡ 6bf3164c-8ffa-44c3-a27e-5450232a7003
▽nll = ForwardDiff.gradient(
	p->extended_nll(AnyModelPars(p), data),
	collect(best_pars_extnll_mc))

# ╔═╡ 555ccd0e-a14a-40ab-976f-19e26665312c
H_mc = ForwardDiff.hessian(
	p->extended_nll(AnyModelPars(p), data),
	collect(best_pars_extnll_mc))

# ╔═╡ f974f215-2a41-4e40-8075-746591fbf859
md"""
## Correlation matrix
"""

# ╔═╡ 77c3c4c6-2768-4bab-a3eb-d6e7472b2fc7
correlations = let
	V = inv(H_mc)
	D = diag(V)
	matrix_of_σ = diagm(sqrt.(D))
	inv(matrix_of_σ) * V * inv(matrix_of_σ)
end;

# ╔═╡ b7d7425c-78ae-40c2-9aa6-9e24e7de5290
begin
	heatmap(correlations, clim=(-1,1))
	for ij in CartesianIndices(correlations)
		i,j = ij[1], ij[2]
		ρ = correlations[ij]
		annotate!((i,j, round(ρ; digits=2)))
	end
	par_labels = fieldnames(ModelPars)
	ticks=(1:length(par_labels), par_labels)
	plot!(; title="correlation matrix", aspect_ratio=1, ticks)
end

# ╔═╡ 758ab556-38c8-4279-8f76-1e02ffce15b5
const from_hesse =let
	names = fieldnames(ModelPars)
	delta_names = "δ" .* string.(names)
	NamedTuple{Symbol.(delta_names)}(sqrt.(diag(inv(H_mc))))
end

# ╔═╡ 18d037fa-1a98-4e2f-ade9-0b58d90b3618
const theta_bounds = map(x->(-1,1) .* 2x, collect(from_hesse))

# ╔═╡ 1d2c6460-e85c-467b-a8ca-5003f35a0d50
md"""
## Likelihood profile
"""

# ╔═╡ 0ff2a4e1-3479-40e8-bd34-fe8bef53416c
(1:3)[[false, true,true]]

# ╔═╡ 83e77656-cf22-46de-b0ee-799564a3bed9
function profile_enll(theta_num, data, initial_estimate)
	n = length(initial_estimate)
	# 
	EM = Diagonal(I, length(initial_estimate))
	eye = EM[theta_num,:]
	to_right_dims = EM[:,(1:n)[.!(eye)]]
	# 
	p0 = collect(initial_estimate)
	# 
    optimize(to_right_dims' * p0, BFGS()) do p
		full_p = to_right_dims * p .+ p0 .* eye
		_y = extended_nll(ModelPars(full_p), data)
		_y
	end
end

# ╔═╡ 9a588dce-f0bc-4767-aace-5dd66c042c92
best_pars_extnll_mc

# ╔═╡ 52ea98d9-2166-4544-9916-8bc484c592ea
plot(μ->profile_enll(1, data, (; best_pars_extnll_mc..., μ)).minimum, 2.3505, 2.3506)

# ╔═╡ 155f671b-449a-4e77-9786-2bdf9cea5c30
const p0 = collect(best_pars_extnll_mc)

# ╔═╡ 5a056cd7-af1c-43b5-bdfc-783f4a55fede
const NLL0 = extended_nll(p0)

# ╔═╡ 07fa4e08-995c-4b6a-bae7-6b5ddd4e859f
Δextended_nll(p::Vector) = extended_nll(p) - NLL0

# ╔═╡ deac4d87-9f9a-4015-be64-68ae3a67e187
r1 = let theta_num = 5
	_r = get_interval(
        zeros(length(p0)),
        theta_num,
        x->Δextended_nll(x+p0),
        :QUADR_EXTRAPOL;
		scan_tol = 0.0001,
        loss_crit = 0.5,
		theta_bounds,
		scan_bounds = theta_bounds[theta_num] ./ 1.2
    )
	update_profile_points!(_r)
	_r
end

# ╔═╡ 0a5a862f-6e0f-4a3a-b608-303bfc288a8b
begin 
	theta_num = r1.input.theta_num
	# 
	denll(δ,i) = Δextended_nll(p0 + Diagonal(I, 5)[i,:] .* δ)
	plot(δ->denll(δ,theta_num), theta_bounds[theta_num]...)
	hline!([0.5], leg=:top)
	# 
	plot!(r1, xlims=theta_bounds[theta_num] ./ 2, ylims=(0, 0.9))
end

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
# ╠═22442c7f-f5b7-4b65-afcf-78a15585bdc3
# ╠═0a0c877a-6d10-4cbb-aa61-baf4a991627d
# ╠═926fb15c-7f10-44af-aafd-c6a9c065502e
# ╟─c72f2357-3d08-49ae-aae6-0f5f157db030
# ╠═755a09d9-c491-439d-b933-1cd7cc29089e
# ╠═7f9e7439-59fe-4402-9738-ca8747299f84
# ╠═b42b7e2e-df09-4273-8282-c294b30f095a
# ╠═2465c8bd-4f47-4f93-b48c-42bd04bc23b9
# ╟─5e997750-3f04-49e1-a083-dc47338be149
# ╠═6d36c5fb-591e-468d-b1b7-e3e8f5727b1c
# ╠═334ab41b-fb0e-4a1f-8b30-3bafd281d525
# ╠═bd1417a4-6101-495b-a47f-b1c045ea8fb8
# ╠═36584517-4416-4c39-8816-75db05b9a495
# ╠═1ba5cacd-ce13-4707-b27d-dedf2b51ccbc
# ╟─29d39a85-9ea6-4fe8-b71a-ac099e323ef4
# ╠═6bf3164c-8ffa-44c3-a27e-5450232a7003
# ╠═555ccd0e-a14a-40ab-976f-19e26665312c
# ╟─f974f215-2a41-4e40-8075-746591fbf859
# ╠═77c3c4c6-2768-4bab-a3eb-d6e7472b2fc7
# ╠═b7d7425c-78ae-40c2-9aa6-9e24e7de5290
# ╠═758ab556-38c8-4279-8f76-1e02ffce15b5
# ╠═18d037fa-1a98-4e2f-ade9-0b58d90b3618
# ╟─1d2c6460-e85c-467b-a8ca-5003f35a0d50
# ╠═0ff2a4e1-3479-40e8-bd34-fe8bef53416c
# ╠═83e77656-cf22-46de-b0ee-799564a3bed9
# ╠═9a588dce-f0bc-4767-aace-5dd66c042c92
# ╠═52ea98d9-2166-4544-9916-8bc484c592ea
# ╠═07fa4e08-995c-4b6a-bae7-6b5ddd4e859f
# ╠═155f671b-449a-4e77-9786-2bdf9cea5c30
# ╠═5a056cd7-af1c-43b5-bdfc-783f4a55fede
# ╠═0a5a862f-6e0f-4a3a-b608-303bfc288a8b
# ╠═deac4d87-9f9a-4015-be64-68ae3a67e187
