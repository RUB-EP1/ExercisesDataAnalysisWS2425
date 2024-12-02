### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ a526fa3e-49ed-462d-9d21-7a5b841a2602
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	Pkg.instantiate()
	# 
	using Plots
	using SpecialFunctions
	using FHist
	using Parameters
	using QuadGK
	using Statistics
	using DataFrames
	using DataAnalysisWS2425
end

# ╔═╡ 3cec85c8-b0aa-11ef-03fd-adfbfe78fee3
md"""
# Lecture 9a: Interval construction

In this lecture we discuss the Neyman method for constructing confidence intervals.
- The data are distributed following a gaussian function, with a known mean μ.
- Variance is unknown and has to be determined from the sample.
- Numerical simulations are supportend by analytical expressions
- First we proceed with a single pseudoexperiment given parameter `pars0`, then, scale to a scan.
"""

# ╔═╡ fd0cfafb-084e-4e81-bc83-2d2efc2f6006
theme(:wong2, fill=0, alpha=0.3, linealpha=1.0, markeralpha=1.0,
	lab="",
	xlims=(:auto, :auto),
	ylims=(:auto, :auto), grid=false, frame=:box)

# ╔═╡ 7fe26504-5d6f-483b-9dfc-02b7ab55cdb9
md"""
## Setup

The `μ0` is fixed, play with `n_data`, try to make sense of the estimator distribution.
"""

# ╔═╡ 124e34f2-420c-4216-93ae-11949523420f
begin
	const μ0 = 2.2
	const pars0 = (σ=0.4,)
end;

# ╔═╡ dbdf19be-ef55-4a00-a758-06f422873da0
const n_data = 7

# ╔═╡ 24e17ceb-1408-4923-9a01-9169f7eedf59
const n_sample = 1_000

# ╔═╡ 6b909f72-7663-4d2a-b400-96b190f537f5
function generate(pars, n_data)
	@unpack σ = pars
	return randn(n_data) .* σ .+ μ0
end

# ╔═╡ bcf6f9d4-cfc0-4949-b474-b476d23b378b
md"""
$\hat{\sigma^2} = \frac{1}{n} \sum_i^n(x_i-\mu)^2$
"""

# ╔═╡ 12d82160-49f8-4700-ad04-bf3e38b2f380
function estimator_variance(data; μ=μ0)
	n = length(data)
	return sum(data) do x
		(x-μ)^2
	end / n
end

# ╔═╡ 00a77358-9b13-4c0e-841f-4c6e081e8da9
variance_pseudodata = map(1:n_sample) do _
	estimator_variance(generate(pars0, n_data))
end;

# ╔═╡ 4d3b7275-7b76-469e-9728-757549a5f032
md"""
## Simulations for CL
"""

# ╔═╡ 43b18c22-9bef-4ae4-a393-947c151f4849
central_CL_simulations = map(range(0.05,0.45, 10)) do σ_sq_true
	_pars = (; σ = sqrt(σ_sq_true))
	_ensamble = map(1:n_sample) do _
		_data = generate(_pars, n_data)
		estimator_variance(_data)
	end
	interval = quantile(_ensamble, [0.16,0.84])
	(; σ_sq_true, interval)
end |> DataFrame

# ╔═╡ 699f85c1-1a25-4a9a-91d4-a61f238b0b9c
md"""
Questions:
- do controur curve meet at 0?
- are the courtour curve straight lines?
- why blue points walk around the line
"""

# ╔═╡ ddca78d4-7ed1-475d-9d7c-35d547ecc218
md"""
## Analytic functions
"""

# ╔═╡ 877c17ab-d285-4aae-b9f0-26835c6fc739
function chi2_pdf(x; ν::Int)
    x < 0 && return 0.0
    coeff = 1 / (2^(ν / 2) * gamma(ν / 2))
    return coeff * x^(ν / 2 - 1) * exp(-x / 2)
end

# ╔═╡ fb0c96a9-01b4-4a96-9f1a-25b96f873c8a
σ_obs_pdf(pars, x; ν) = chi2_pdf(x/pars.σ^2*n_data; ν) / pars.σ^2*n_data

# ╔═╡ 7acbe296-d5f6-44c9-812a-a8f07a453624
function σ_obs_quantile(pars, p; ν)
	@unpack σ = pars
	x = 2 * gamma_inc_inv(ν / 2, p, 1 - p)
	x * σ^2 / n_data
end

# ╔═╡ ce1447b9-a351-444e-8236-1eb66c6b60b2
begin
	h = Hist1D(variance_pseudodata; binedges=range(0,0.5,100))
	# pdf numerically (histogram)
	plot(h, seriestype=:stepbins)
	Δx = diff(h.binedges[1])[1]
	scale = length(variance_pseudodata) * Δx
	# pdf analytically (curve)
	plot!(WithData(h), fill=false, lw=2, lab="ana") do x
		σ_obs_pdf(pars0, x; ν=n_data)
	end
	# quantile numerically
	x1,x2 = quantile(variance_pseudodata, [0.16,0.84])
	plot!([x1,x2], maximum(h.bincounts)/10 .* [1,1], m=(10, :o), lab="CL num", fill=false, lw=10)
	# quantile analytically
	x1,x2 = σ_obs_quantile.(Ref(pars0), [0.16,0.84]; ν=n_data)
	plot!(x1,x2, lw=2, lab="CL ana") do x
		σ_obs_pdf(pars0, x; ν=n_data) * WithData(h).factor
	end
	plot!(title="Distribution of estimator",
		xlab="hat{σ²}")
end

# ╔═╡ d4c840bc-676b-4bf9-80f4-17def0372aee
begin
	plot()
	σhat_vals = range(0, 0.5, 30)
	σhat_1 = map(σ_sq->σ_obs_quantile((; σ=sqrt(σ_sq)), 0.16; ν=n_data), σhat_vals)
	σhat_2 = map(σ_sq->σ_obs_quantile((; σ=sqrt(σ_sq)), 0.84; ν=n_data), σhat_vals)
	plot(Plots.Shape(
		vcat(σhat_2, reverse(σhat_1)),
		vcat(σhat_vals, reverse(σhat_vals))
	), lab="belt ana", c=2)
	map(eachrow(central_CL_simulations)) do s
		@unpack σ_sq_true, interval = s
		plot!(collect(interval), σ_sq_true .* [1,1],
			m=(4,:o), color=3, fill=false, lab="")
	end
	vline!([0.15], lab="σ² obs", l=(4, :green), fill=false)
	plot!(title="central CL", xlab="hat{σ²}", ylab="σ²")
end

# ╔═╡ Cell order:
# ╟─3cec85c8-b0aa-11ef-03fd-adfbfe78fee3
# ╠═a526fa3e-49ed-462d-9d21-7a5b841a2602
# ╠═fd0cfafb-084e-4e81-bc83-2d2efc2f6006
# ╟─7fe26504-5d6f-483b-9dfc-02b7ab55cdb9
# ╠═124e34f2-420c-4216-93ae-11949523420f
# ╠═dbdf19be-ef55-4a00-a758-06f422873da0
# ╠═24e17ceb-1408-4923-9a01-9169f7eedf59
# ╠═6b909f72-7663-4d2a-b400-96b190f537f5
# ╟─bcf6f9d4-cfc0-4949-b474-b476d23b378b
# ╠═12d82160-49f8-4700-ad04-bf3e38b2f380
# ╠═00a77358-9b13-4c0e-841f-4c6e081e8da9
# ╠═ce1447b9-a351-444e-8236-1eb66c6b60b2
# ╟─4d3b7275-7b76-469e-9728-757549a5f032
# ╠═43b18c22-9bef-4ae4-a393-947c151f4849
# ╠═d4c840bc-676b-4bf9-80f4-17def0372aee
# ╟─699f85c1-1a25-4a9a-91d4-a61f238b0b9c
# ╟─ddca78d4-7ed1-475d-9d7c-35d547ecc218
# ╠═877c17ab-d285-4aae-b9f0-26835c6fc739
# ╠═fb0c96a9-01b4-4a96-9f1a-25b96f873c8a
# ╠═7acbe296-d5f6-44c9-812a-a8f07a453624
