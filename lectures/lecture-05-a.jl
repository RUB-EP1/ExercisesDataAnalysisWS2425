### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2d141b9d-09bb-4074-bf01-4c4b6099585d
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "ReferenceDataAnalysisWS2425"))
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
# Lecture 5a: Parameter uncertainties

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

# ╔═╡ 30eef5b0-1b1b-4711-9d0d-06679c84b23e
function pol1_with_logs_slope(x, pars)
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

# ╔═╡ 8ff5e326-1e5c-431b-a21c-83171af7d879
begin
	abstract type SpectrumModels end
	Base.collect(model::SpectrumModels) =
		getproperty.(model |> Ref, collect(fieldnames(typeof(model))))
	# 
	# Define a generic constructor for any subtype of SpectrumModels
	function (::Type{T})(p_values::Union{AbstractVector,NTuple}) where {T <: SpectrumModels}
	    T(; NamedTuple{fieldnames(T)}(p_values)...)
	end
	# 
	@with_kw struct Anka{P} <: SpectrumModels
		μ::P
		σ::P
		a::P
		flat::P
		log_slope::P
	end
	# 
	function peak1_func(model::Anka, x)
		@unpack μ, σ, a = model
	    gaussian_scaled(x; μ, σ, a)
	end
	background_func(model::Anka, x) = pol1_with_logs_slope(x, model)
	#
	"""
		model_func(model::Anka, x)

	where
	
		Anka{P} <: SpectrumModels
	
	evaluation of the pdf for model `Anka`.
	is a simple spectral model that has two components,
	 - a background described by pol1, and
	 - a peaking signal described by the gaussian function

	# Example
	```julia
	julia> model = Anka(; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)
	julia> model_func(model, 3.3)
	4.1775
	```
	"""
	model_func(model::Anka, x) = peak1_func(model, x) + background_func(model, x)
	# 
end;

# ╔═╡ 54629edc-12ed-402d-bfc4-33cbb0b71848
const default_model = Anka(; μ = 2.35, σ = 0.01, flat = 1.5, log_slope = 2.1, a = 5.0)

# ╔═╡ a78de1dc-dc59-4296-a18d-a08a8186fb7d
const model_fieldnames = fieldnames(Anka)

# ╔═╡ 8878c8be-21ba-4fc2-aa59-76754ebfbf0b
md"""
## Data
"""

# ╔═╡ d1b7527c-4375-4773-99b4-d039462b5044
const nData = 1_000

# ╔═╡ d1fdce14-0768-4838-b244-8b7fb101b137
const data = sample_inversion(nData, support) do x
        model_func(default_model, x)
    end

# ╔═╡ 255e5019-f8e0-49ed-8efa-ba21004008aa
let
    binedges = range(support..., 100)
    h = Hist1D(data; binedges)
	plot(h, seriestype=:stepbins)
	# I have to compute normalization, it is neither nData, nor 1
	norm = quadgk(support...) do x
		model_func(default_model, x)
	end[1]
	plot!(WithData(h); n_points=300) do x
		model_func(default_model, x) / norm
	end
end

# ╔═╡ 156e44f7-90a8-434d-a90a-38355b630c01
md"""
## Fitting
"""

# ╔═╡ 71f3919c-92c6-47a4-b6cb-d1d2195cb685
initial_guess = 
	Anka(; μ = 2.35, σ = 0.01, flat = 1175.17, log_slope = 2.1, a = 5000.0)

# ╔═╡ 2d97dcb6-78fe-4fda-acd9-d1b72f680862
fit_result = fit_enll(collect(initial_guess), data; support) do x, pars
	_model = Anka(pars)
	model_func(_model, x)
end

# ╔═╡ 8844e5d2-b779-49de-9177-dcec11d73e8d
best_pars_extnll = fit_result.minimizer |> Anka

# ╔═╡ 45e8e2df-4287-41d5-8568-7e78898cd587
let
	binedges = range(support..., 100)
	h = Hist1D(data; binedges)
	# 
	curvedfitwithpulls(h, xlab = "X-axis", ylab = "Y-axis";
		data_scale_curve = false, n_points=1000) do x
		model_func(best_pars_extnll, x)
	end
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

# ╔═╡ 87711274-3ecc-4887-a470-8b9b1fef07fe
function local_extended_nll(p_values::AbstractArray)
	_model = Anka(p_values)
	enll = extended_nll(data) do x
		model_func(_model, x)
	end
	enll
end

# ╔═╡ 6bf3164c-8ffa-44c3-a27e-5450232a7003
▽nll = ForwardDiff.gradient(
	local_extended_nll,
	collect(best_pars_extnll) |> collect)

# ╔═╡ 555ccd0e-a14a-40ab-976f-19e26665312c
H_mc = ForwardDiff.hessian(local_extended_nll, collect(best_pars_extnll))

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

# ╔═╡ 758ab556-38c8-4279-8f76-1e02ffce15b5
const from_hesse = let
    delta_names = "δ" .* string.(model_fieldnames)
    NamedTuple{Symbol.(delta_names)}(sqrt.(diag(inv(H_mc))))
end

# ╔═╡ b7d7425c-78ae-40c2-9aa6-9e24e7de5290
begin
    heatmap(correlations, clim = (-1, 1))
    for ij in CartesianIndices(correlations)
        i, j = ij[1], ij[2]
        ρ = correlations[ij]
        annotate!((i, j, round(ρ; digits = 2)))
    end
    ticks = (1:length(model_fieldnames), model_fieldnames)
    plot!(; title = "correlation matrix", aspect_ratio = 1, ticks)
end

# ╔═╡ 1d2c6460-e85c-467b-a8ca-5003f35a0d50
md"""
## Likelihood profile
"""

# ╔═╡ 85380f97-f586-4f42-ba1b-a88af67f0766
const p0 = collect(best_pars_extnll)

# ╔═╡ 5a056cd7-af1c-43b5-bdfc-783f4a55fede
const NLL0 = local_extended_nll(p0)

# ╔═╡ 18d037fa-1a98-4e2f-ade9-0b58d90b3618
const theta_bounds = map(x -> (-1, 1) .* 2x, collect(from_hesse))

# ╔═╡ 83e77656-cf22-46de-b0ee-799564a3bed9
function fit_with_fixed(objective, initial; numbers_to_fix)
    n = length(initial)
    #
    unitm = Diagonal(I, n)
    eye = reduce(.|, (unitm[i, :] for i in numbers_to_fix))
    to_right_dims = unitm[:, (1:n)[.!(eye)]]
    #
    optimize(to_right_dims' * initial, BFGS()) do p
        full_p = to_right_dims * p .+ initial .* eye
        objective(full_p)
    end
end

# ╔═╡ 0a5a862f-6e0f-4a3a-b608-303bfc288a8b
likelihood_profiling = let theta_num = 2
    #
    p(δ) = p0 + Diagonal(I, length(p0))[theta_num, :] .* δ
    #
    grid = range(theta_bounds[theta_num]..., 10)
    projecting = map(grid) do δ
        local_extended_nll(p(δ)) - NLL0
    end
    profiling = map(grid) do δ
        res = fit_with_fixed(local_extended_nll, p(δ); numbers_to_fix=[theta_num])
        res.minimum - NLL0
    end
    (; theta_num, grid, projecting, profiling)
end;

# ╔═╡ b1f721ff-fa4c-42a3-8a50-3003d9be731b
function interpolate_to_zero(two_x,two_y)
	w_left = 1 ./ two_y .* [1, -1]
	w_left ./= sum(w_left)
	return two_x' * w_left
end

# ╔═╡ 499edc2f-9273-4bc6-9260-8aebef555f24
function findzeros_two_sides(xv,yv)
	yxv = yv .* xv
	_left = findfirst(x->x>0, yxv)
	_right = findlast(x->x<0,yxv)
	# 
	x_left_zero = interpolate_to_zero(
		[xv[_left-1], xv[_left]], [yv[_left-1], yv[_left]])
	x_right_zero = interpolate_to_zero(
		[xv[_right], xv[_right+1]], [yv[_right], yv[_right+1]])
	# 
	[x_left_zero, x_right_zero]
end

# ╔═╡ c17f58db-7790-49c6-b121-cafef370ef5d
let
	@unpack projecting, profiling, grid, theta_num = likelihood_profiling
    #
	xlab = "δ$(model_fieldnames[theta_num])"
	plot(title = ["studies of parameter $(model_fieldnames[theta_num])" ""];
        xlab, ylab = "ΔNLL")
	# 
    plot!(grid, profiling, lab = "profile likelihood", c=2)
	plot!(grid, projecting, lab = "project likelihood", c=3)
	hline!([0.5], leg = :top, c=1)
    # 
	vspan!(findzeros_two_sides(grid, profiling .- 1/2), α=0.2, c=2)
	vspan!(findzeros_two_sides(grid, projecting .- 1/2), α=0.2, c=3)
end

# ╔═╡ 02f9de05-3137-436b-b38f-c4456204bde4
md"""
### 2D profiling
"""

# ╔═╡ bc7e2edb-d9ca-4267-8a07-ad1e3ed147a7
likelihood_profiling_2d = let theta_num = 2, theta_num′ = 3
    #
	d = Diagonal(I, length(p0))
    p(δ,δ′) = p0 + d[theta_num, :] .* δ + d[theta_num′, :] .* δ′
    #
    grid = range(theta_bounds[theta_num]..., 10)
    grid′ = range(theta_bounds[theta_num′]..., 10)
	# 
    projecting = map(Iterators.product(grid,grid′)) do (δ,δ′)
        local_extended_nll(p(δ,δ′)) - NLL0
    end
    profiling = map(Iterators.product(grid,grid′)) do (δ,δ′)
        res = fit_with_fixed(local_extended_nll, p(δ,δ′);
			numbers_to_fix=[theta_num, theta_num′])
        res.minimum - NLL0
    end
    (; theta_num, theta_num′, grid, grid′, projecting, profiling)
end;

# ╔═╡ 663765c7-c8f6-4c6e-89b5-f6a78d443d60
chi2_ndf2_quantile(α) = -2log(1-α)

# ╔═╡ f2c1cc33-54cd-4024-8811-8c3cc9d872ea
levels = chi2_ndf2_quantile.([0.68, 0.95]) ./ 2

# ╔═╡ 1eafb617-c63d-4d2b-b30a-9876d33990b4
let
	@unpack grid, grid′, projecting, profiling = likelihood_profiling_2d
	plot(colorbar=false)
	contour!(grid, grid′, projecting; levels, c=:orange)
	contour!(grid, grid′, profiling; levels, c=:purple)
	# 
	@unpack theta_num, theta_num′ = likelihood_profiling_2d
	scatter!([0], [0], m=(:red,:+, 10))
	plot!(xlab=model_fieldnames[theta_num], ylab=model_fieldnames[theta_num′])
	# 
	vspan!(from_hesse[theta_num] .* [-1,1], α=0.1)
	hspan!(from_hesse[theta_num′] .* [-1,1], α=0.1)
end

# ╔═╡ 3b748eaf-446b-42a6-b643-5cb6af823419
# cspell:disable

# ╔═╡ Cell order:
# ╟─9da55708-8792-4b26-984f-5795a981bf2c
# ╠═2d141b9d-09bb-4074-bf01-4c4b6099585d
# ╠═39ee4bcd-8d01-443f-9714-103ab6d7f7d6
# ╠═9b6b7d99-9f92-4b0a-b617-4111317e8271
# ╟─ace5e914-f516-438e-ae04-012573ad3586
# ╠═30eef5b0-1b1b-4711-9d0d-06679c84b23e
# ╠═8ff5e326-1e5c-431b-a21c-83171af7d879
# ╠═54629edc-12ed-402d-bfc4-33cbb0b71848
# ╠═a78de1dc-dc59-4296-a18d-a08a8186fb7d
# ╟─8878c8be-21ba-4fc2-aa59-76754ebfbf0b
# ╠═d1b7527c-4375-4773-99b4-d039462b5044
# ╠═d1fdce14-0768-4838-b244-8b7fb101b137
# ╠═255e5019-f8e0-49ed-8efa-ba21004008aa
# ╟─156e44f7-90a8-434d-a90a-38355b630c01
# ╠═71f3919c-92c6-47a4-b6cb-d1d2195cb685
# ╠═2d97dcb6-78fe-4fda-acd9-d1b72f680862
# ╠═8844e5d2-b779-49de-9177-dcec11d73e8d
# ╠═45e8e2df-4287-41d5-8568-7e78898cd587
# ╟─29d39a85-9ea6-4fe8-b71a-ac099e323ef4
# ╠═87711274-3ecc-4887-a470-8b9b1fef07fe
# ╠═6bf3164c-8ffa-44c3-a27e-5450232a7003
# ╠═555ccd0e-a14a-40ab-976f-19e26665312c
# ╟─f974f215-2a41-4e40-8075-746591fbf859
# ╠═77c3c4c6-2768-4bab-a3eb-d6e7472b2fc7
# ╠═758ab556-38c8-4279-8f76-1e02ffce15b5
# ╠═b7d7425c-78ae-40c2-9aa6-9e24e7de5290
# ╟─1d2c6460-e85c-467b-a8ca-5003f35a0d50
# ╠═85380f97-f586-4f42-ba1b-a88af67f0766
# ╠═5a056cd7-af1c-43b5-bdfc-783f4a55fede
# ╠═18d037fa-1a98-4e2f-ade9-0b58d90b3618
# ╠═83e77656-cf22-46de-b0ee-799564a3bed9
# ╠═0a5a862f-6e0f-4a3a-b608-303bfc288a8b
# ╠═c17f58db-7790-49c6-b121-cafef370ef5d
# ╠═b1f721ff-fa4c-42a3-8a50-3003d9be731b
# ╠═499edc2f-9273-4bc6-9260-8aebef555f24
# ╟─02f9de05-3137-436b-b38f-c4456204bde4
# ╠═bc7e2edb-d9ca-4267-8a07-ad1e3ed147a7
# ╠═1eafb617-c63d-4d2b-b30a-9876d33990b4
# ╠═663765c7-c8f6-4c6e-89b5-f6a78d443d60
# ╠═f2c1cc33-54cd-4024-8811-8c3cc9d872ea
# ╠═3b748eaf-446b-42a6-b643-5cb6af823419
