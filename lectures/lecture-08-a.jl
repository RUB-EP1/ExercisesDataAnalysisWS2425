### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ ba7c1cc4-aa79-11ef-2d0f-8bcd779516ef
# ╠═╡ show_logs = false
begin
    using Pkg: Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    #
    using DataAnalysisWS2425
    using FHist
    using Statistics
    using Parameters
    using SpecialFunctions
	# 
    using Random
    Random.seed!(1000)
	# 
	using Plots
	theme(:wong2,
		grid = false, frame = :box,
		ylims = (0, :auto), xlims = (:auto, :auto),
		lab="", linealpha=1)
end

# ╔═╡ b4a9b6fa-fdfb-483f-9d46-407a443dae7e
using BenchmarkTools

# ╔═╡ 038132f0-9147-4917-9ab2-411ebf1c181b
using Plots.PlotMeasures: mm

# ╔═╡ 24985ad9-ed3a-4b70-99b7-d8377542d562
md"""
# Lecture 8a: Scan, confidence inversion

In this notebook, we will figure out how statistical methods has been used to search for Higgs boson

![](https://cds.cern.ch/record/1471031/files/CombinedResults.png)

Short list of references:
- A good [CLs discussion](https://inspirehep.net/literature/599622) by Alexander L. Read. How on interprete the Brazil plot
- [ROOT Demo](https://root.cern/doc/master/StandardHypoTestInvDemo_8C.html): not a typo. X-axis should be CL, rather then a p-value
"""

# ╔═╡ 4931cdb4-635a-4484-b823-2f7c419d502c
const support = (0.5, 4.5);

# ╔═╡ 411d1afc-24c0-496c-9b7b-7bd0bda0004d
const background_parameters = (flat=1.0, log_slope=0.8);

# ╔═╡ 5e8302c2-6ebd-4070-9e00-be2fa36a779d
const resolution = (; σ=0.1);

# ╔═╡ 6b08b202-4762-420b-ac45-8891efcd7df0
const default_model = Anka(; μ=3.3, a=1.4, resolution..., background_parameters...)

# ╔═╡ 0f997dc7-2401-4c59-82da-ccb9b5ddb9a9
begin
	plot()
	plot!(x->total_func(default_model,x), support...)
	plot!(x->peak1_func(default_model,x), support..., fill=0, α=0.3)
end

# ╔═╡ 84df0452-3888-4c74-958a-c8a6dcc9f4b6
const nData = 10_000

# ╔═╡ 5f2cd93f-07cb-47ee-b1cd-a152a1fda29a
const data = sample_inversion(x->total_func(default_model,x), nData, support);

# ╔═╡ 8caceda7-43ab-460e-92f2-119ea560360a
let
	bins=range(support..., 50)
	stephist(data; bins)
end

# ╔═╡ d9d6f03b-5d27-4db9-b5cb-a45f11ee001e
md"""
## Hypothesis testing

Let's assume that we search of a signal of known width of `σ=0.1GeV` (resolution).
- We do not knonw the location of the peak, it will be a scann variable.
- We do not know a strength of the peak and will scan for it as well.
We start from scanning the stregth for a given mass.
"""

# ╔═╡ f24b965e-9e8c-4576-83a7-e9e4bb7ecb19
const μ_test = 2.2;

# ╔═╡ 6ba312c4-b2c8-463a-846c-6786bda846cd
md"""
## Defining hypothesis
"""

# ╔═╡ 0c33bd56-f1eb-411f-ac19-553a7295ac09
_quadgk_call_on_model(model,support) =
	DataAnalysisWS2425._quadgk_call(x->total_func(model, x), support)

# ╔═╡ f3141719-2f62-41b2-94c2-b97708a5ba05
function nll(model, data;
    support = extrema(data),
    normalization_call = _quadgk_call_on_model)
    #
    minus_sum_log = -sum(data) do x
        value = total_func(model, x)
        value > 0 ? log(value) : -1e10
    end
    #
    n = length(data)
    normalization = normalization_call(model, support)
    nll = minus_sum_log + n * log(normalization)
    return nll
end

# ╔═╡ 4bc454ca-7361-4b2e-ac22-34f7dcc224d0
_integral(model, support) = DataAnalysisWS2425._quadgk_call(
	x->total_func(model, x), support)

# ╔═╡ b00fb9ba-8b64-49b1-819f-ee0c1e093b5d
md"""
## Benchmarking

```
Mac Pro M3
nData = 10_000
(μ=1.1, a=0.3)
```

Start by saying, `using BenchmarkTools`

### Data generation
```julia
julia> @btime sample_inversion(x->total_func(H0_model, x), nData, support)
2.081 ms (30035 allocations: 2.30 MiB)
julia> @btime sample_rejection(x->total_func(H0_model, x), nData, support)
311.959 μs (10014 allocations: 482.72 KiB)
# even faster implementation, no push!
julia> @btime better_sample_rejection(x->total_func(H0_model, x), Int(1.8nData), support)[1:nData]
150.708 μs (38 allocations: 1014.84 KiB)
```

### NLL call
```julia
julia> @btime nll(H1_model(;μ=1.1, a=0.3), data; normalization_call=fast_normalization, support)
119.541 μs (0 allocations: 0 bytes)

julia> nll(H1_model(;μ=1.1, a=0.3), data) # quadgk
152.667 μs (3 allocations: 384 bytes)
```
"""

# ╔═╡ 10a76edc-fcda-44f0-bad6-7a4b04c31c2c
function better_sample_rejection(f, n_proposal, support)
    a, b = support
    sample = a .+ rand(n_proposal) .* (b - a)
    values = f.(sample)
    values[values .> rand(n_proposal) .* maximum(values)]
end

# ╔═╡ 6552100c-d6e9-4612-a0df-e0f04f239e71
const H0_model = Anka(; μ=1.0, a=0.0, resolution..., background_parameters...);

# ╔═╡ 7cb3b19b-ec7f-4318-b452-c7f03838fb0b
H1_model(; μ,a) = Anka(; μ, a, resolution..., background_parameters...)

# ╔═╡ 5b691c65-13f9-4e01-8748-d5eb3e76cfcb
function expectations(support, nData = 1; μ, a=1.0)
	fB = _integral(H0_model, μ .+ (-1,1) .* resolution.σ) / 
		_integral(H0_model, support)
	_H1_model = H1_model(; μ, a)
	fSB = _integral(_H1_model, μ .+ (-1,1) .* resolution.σ) /
		_integral(_H1_model, support)
	#
	nB, nSB = nData .* (fB, fSB)
	(; nB, nSB)
end

# ╔═╡ 395762bf-5fe5-4e00-aaab-d1f2010b4fdf
expectations(support, nData; μ=μ_test)

# ╔═╡ 4aba026b-a6d9-480c-805d-7d49b7049689
test_statitics(data; μ, a, kw...) = 2*(nll(H0_model, data; kw...) - nll(H1_model(; μ, a), data; kw...))

# ╔═╡ 0e398379-ed9b-4b31-987f-5523479db943
const nSample_H0_test = 1_000;

# ╔═╡ 608b55c2-e01d-47d1-9acf-22a4b32a3add
const nSample_H1_test = 1_000;

# ╔═╡ 5059a6eb-afd4-4ef0-af58-93b2ac66f2cf
const a_test = 0.2;

# ╔═╡ c9315753-951b-4e72-960b-3388815fffcd
expectations(support, nData; μ=μ_test, a=a_test)

# ╔═╡ 305e9819-04b0-4930-82ab-22df856ea4ca
md"""
One can either generate all needed pseudodata at once, using
```julia
H0_sample_test = let
	n = nData*nSample_H0_test
	pseudo_data = sample_rejection(x->total_func(H0_model,x), n, support)
	pseudo_data_matrix = reshape(pseudo_data, (nSample_H0_test, nData))
	mapslices(pseudo_data_matrix, dims=2) do _data
		test_statitics(_data; μ=μ_test, a=a_test) # try also ; nomalization_call=fast_normalization
	end[:,1]
end
```
"""

# ╔═╡ 3b0ce27c-6952-45e8-947a-d5d64b40bd1f
md"""
or generate it in a lookup. The speed is almost the same (The second option is marginally faster)
"""

# ╔═╡ 5c2f4961-ffaa-4122-a60a-e3370120fea9
H0_sample_test = map(1:nSample_H0_test) do _
	μ=μ_test
	a=a_test
	pseudo_data = sample_rejection(x->total_func(H0_model, x), nData, support)
	test_statitics(pseudo_data; μ, a, support)
end;

# ╔═╡ 9adf880e-61a8-473b-9bef-ac3967aa6db5
H1_sample_test = map(1:nSample_H1_test) do _
	μ=μ_test
	a=a_test
	pseudo_data = sample_rejection(x->total_func(H1_model(; μ, a),x), nData, support)
	test_statitics(pseudo_data; μ, a)
end;

# ╔═╡ 417d3a65-9a19-44b5-9482-bceb86bf6999
md"""
## Asymptotics
Let's look at the tail of the H0 distribution
"""

# ╔═╡ 6cd41d11-9462-4b8e-989d-d032a6f2bd62
let
	lims = extrema(vcat(H0_sample_test, H1_sample_test))
	bins = range(lims..., 30)
	plot()
	stephist!(H0_sample_test; bins, fill=0, alpha=0.3, lab="T(H0)")
	stephist!(H1_sample_test; bins, fill=0, alpha=0.3, lab="T(H1)")
	let
		μ = mean(H0_sample_test)
		σ = std(H0_sample_test)
		plot!(x->gaussian_scaled(x; a=1/sqrt(2π)/σ, μ, σ),
			WithData(bins, nSample_H0_test), l=(1, :black))
	end
	plot!()
end

# ╔═╡ 456c69fb-4903-4362-a999-efeb75ff3bc2
log10_ranged(x) = x<1e-5 ? -5 : log10(x)

# ╔═╡ da305f5f-76c5-4b5f-a51f-c7ccfa6592b8
md"""
## Scan over the signal strength

clearly we need to speed up things.
One way to do it is  `normalization_integral`
"""

# ╔═╡ 3bfdcbd5-663a-476a-9ab4-0287974439a1
cdf_gauss(x) = 0.5*(1+erf(x/sqrt(2)))

# ╔═╡ df49e3ef-0b35-4047-b0e9-9bc9da61712f
p_values_test = let
	cuts = quantile(H0_sample_test, cdf_gauss.([-2,-1,0,1,2]))
	map(cuts) do cut
		sum(H1_sample_test .> cut) / nSample_H0_test
	end .- 1e-6
end

# ╔═╡ 40ad2d8b-d895-4f4e-ac2b-0fee34e68cdf
begin
	plot(ylab="CL", xlab="a", title="scan of a")
	# plot()
	plot!([a_test, a_test], 1 .- p_values_test[[1,5]], l=(:yellow, 30))
	plot!([a_test, a_test], 1 .- p_values_test[[2,4]], l=(:green, 20))
	scatter!([a_test], 1 .- p_values_test[[3]], m=(:d, 10, :black))
	plot!(xlim=(0.8,1.3).* a_test)
	plot!(ylim=(1e-5, 1), yscale=:log10)
	hline!([0.05], l=(:gray, 1), 
		ann=(a_test*1.1, 0.05, text("95%", :bottom)))
end

# ╔═╡ 6638d724-e2a7-4ad7-b162-545ce87ab994
begin
	struct HypothesisSimulations{T}
		T_H0::Vector{Float64}
		T_H1::Vector{Float64}
		pars::T
	end
	function HypothesisSimulations(; a,μ, nSample, nData, kw...)
		f0(x) = total_func(H0_model, x)
		f1(x) = total_func(H1_model(; μ, a), x)
		# 
		T_H0 = map(1:nSample) do _
			pseudo_data = sample_rejection(f0, nData, support)
			test_statitics(pseudo_data; μ, a, kw...)
		end
		T_H1 = map(1:nSample) do _
			pseudo_data = sample_rejection(f1, nData, support)
			test_statitics(pseudo_data; μ, a, kw...)
		end
		HypothesisSimulations(T_H0, T_H1, (; a,μ))
	end
	pvalue(sample::Vector, value::Float64) =
		sum(sample .> value) / length(sample)
	CLb(hs::HypothesisSimulations, value::Float64) = 1 - pvalue(hs.T_H0, value) + 1e-10
	CLsb(hs::HypothesisSimulations, value::Float64) = 1 - pvalue(hs.T_H1, value) + 1e-10
	CLs(hs::HypothesisSimulations, value::Float64) = CLsb(hs, value) / CLb(hs, value)
	# 
	five_CLs(hs) = CLs.(Ref(hs), quantile(hs.T_H0, cdf_gauss.([-2,-1,0,1,2])))
end

# ╔═╡ 17c7c463-1abc-4fab-b202-b419a6e2ffcc
let 
	d = randn(100000)
	quantile(d, cdf_gauss.([-2,-1,0,1,2]))
end

# ╔═╡ 6133b706-92dc-4270-b789-9d71b2ff97ea
const _int_b = _quadgk_call_on_model(H0_model, support)

# ╔═╡ 5cc45898-475e-422e-b912-0222c21738d5
# compare to sqrt(2π)*resolution.σ
const _int_s = _quadgk_call_on_model(H1_model(; μ=2.2, a=1.0), support)-_int_b

# ╔═╡ 7494b3f0-189a-4dfe-b2ff-06d357c197dd
fast_normalization(model, support) = model.a * _int_s + _int_b;

# ╔═╡ 578459f4-c461-4d5b-9982-40ddad1965ef
ht_a_scan = map([0.15, 0.2, 0.25, 0.3, 0.35, 0.4,]) do a
	HypothesisSimulations(; a, μ=μ_test, nSample=5000, nData=nData,
		normalization_call=fast_normalization)
end;

# ╔═╡ 122c7e23-2553-4f3b-90f2-7bce60f86f9a
expectations(support, nData; μ=μ_test, a=0.3)

# ╔═╡ 75bb816d-3f5f-4ea0-b020-1b3f217ce82b
let
	# plot(ylab="CLs", xlab="a", title="scan of a")
	plot()
	for ht in ht_a_scan
		_CLs = five_CLs(ht)
		a = ht.pars.a
		plot!([a, a], _CLs[[1,5]], l=(:yellow, 30))
		plot!([a, a], _CLs[[2,4]], l=(:green, 20))
		scatter!([a], _CLs[[3]], m=(:d, 10, :black))
	end
	# 
	plot!(xlim=(-0.01, 0.55))
	plot!(ylims=(1e-4, 1.4), yscale=:log10)
	hline!([0.05], l=(:gray, 1), 
		ann=(a_test*1.1, 0.05, text("5%", :bottom)))
end

# ╔═╡ Cell order:
# ╟─24985ad9-ed3a-4b70-99b7-d8377542d562
# ╠═ba7c1cc4-aa79-11ef-2d0f-8bcd779516ef
# ╠═4931cdb4-635a-4484-b823-2f7c419d502c
# ╠═411d1afc-24c0-496c-9b7b-7bd0bda0004d
# ╠═5e8302c2-6ebd-4070-9e00-be2fa36a779d
# ╠═6b08b202-4762-420b-ac45-8891efcd7df0
# ╠═0f997dc7-2401-4c59-82da-ccb9b5ddb9a9
# ╠═84df0452-3888-4c74-958a-c8a6dcc9f4b6
# ╠═5f2cd93f-07cb-47ee-b1cd-a152a1fda29a
# ╠═8caceda7-43ab-460e-92f2-119ea560360a
# ╟─d9d6f03b-5d27-4db9-b5cb-a45f11ee001e
# ╠═f24b965e-9e8c-4576-83a7-e9e4bb7ecb19
# ╟─6ba312c4-b2c8-463a-846c-6786bda846cd
# ╠═0c33bd56-f1eb-411f-ac19-553a7295ac09
# ╠═f3141719-2f62-41b2-94c2-b97708a5ba05
# ╠═4bc454ca-7361-4b2e-ac22-34f7dcc224d0
# ╠═5b691c65-13f9-4e01-8748-d5eb3e76cfcb
# ╠═395762bf-5fe5-4e00-aaab-d1f2010b4fdf
# ╟─b00fb9ba-8b64-49b1-819f-ee0c1e093b5d
# ╠═b4a9b6fa-fdfb-483f-9d46-407a443dae7e
# ╠═10a76edc-fcda-44f0-bad6-7a4b04c31c2c
# ╠═6552100c-d6e9-4612-a0df-e0f04f239e71
# ╠═7cb3b19b-ec7f-4318-b452-c7f03838fb0b
# ╠═4aba026b-a6d9-480c-805d-7d49b7049689
# ╠═0e398379-ed9b-4b31-987f-5523479db943
# ╠═608b55c2-e01d-47d1-9acf-22a4b32a3add
# ╠═5059a6eb-afd4-4ef0-af58-93b2ac66f2cf
# ╠═c9315753-951b-4e72-960b-3388815fffcd
# ╟─305e9819-04b0-4930-82ab-22df856ea4ca
# ╟─3b0ce27c-6952-45e8-947a-d5d64b40bd1f
# ╠═5c2f4961-ffaa-4122-a60a-e3370120fea9
# ╠═9adf880e-61a8-473b-9bef-ac3967aa6db5
# ╟─417d3a65-9a19-44b5-9482-bceb86bf6999
# ╠═6cd41d11-9462-4b8e-989d-d032a6f2bd62
# ╠═df49e3ef-0b35-4047-b0e9-9bc9da61712f
# ╠═456c69fb-4903-4362-a999-efeb75ff3bc2
# ╠═40ad2d8b-d895-4f4e-ac2b-0fee34e68cdf
# ╟─da305f5f-76c5-4b5f-a51f-c7ccfa6592b8
# ╠═6638d724-e2a7-4ad7-b162-545ce87ab994
# ╠═17c7c463-1abc-4fab-b202-b419a6e2ffcc
# ╠═3bfdcbd5-663a-476a-9ab4-0287974439a1
# ╠═6133b706-92dc-4270-b789-9d71b2ff97ea
# ╠═5cc45898-475e-422e-b912-0222c21738d5
# ╠═7494b3f0-189a-4dfe-b2ff-06d357c197dd
# ╠═578459f4-c461-4d5b-9982-40ddad1965ef
# ╠═122c7e23-2553-4f3b-90f2-7bce60f86f9a
# ╠═038132f0-9147-4917-9ab2-411ebf1c181b
# ╠═75bb816d-3f5f-4ea0-b020-1b3f217ce82b
