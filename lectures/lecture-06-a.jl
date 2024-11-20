### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ a13e09e4-9c86-11ef-11e6-c5d0de162eac
# ╠═╡ show_logs = false
begin
    using Pkg: Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    #
    using DataAnalysisWS2425
    using Plots
    using Random
    using FHist
    using SpecialFunctions
    Random.seed!(1000)
end

# ╔═╡ 85a7a141-79f6-4e75-9b97-0056e2d4a1c8
md"""
# Lecture 6a: $\chi^2$ distribution

This lecture demonstrate how $\chi^2$ distribution comes about:
# - how to generate it using the gaussian distribution, and
- how to describe using an analytic expression.
"""

# ╔═╡ 90f71c33-e5f3-4378-9bca-e044b5e3256c
theme(:wong2, grid = false, frame = :box, ylims = (0, :auto), xlims = (:auto, :auto))

# ╔═╡ f559fd65-e5dc-43d6-b099-1ba62ab75f0e
const n = 10_000

# ╔═╡ 48c28514-b325-4d77-95d7-770e7572ef18
begin
    binedges = range(0, 10, 100)
    h1 = Hist1D(sum(randn(n, 1) .^ 2, dims = 2)[:, 1]; binedges)
    h2 = Hist1D(sum(randn(n, 2) .^ 2, dims = 2)[:, 1]; binedges)
    h3 = Hist1D(sum(randn(n, 3) .^ 2, dims = 2)[:, 1]; binedges)
    h4 = Hist1D(sum(randn(n, 4) .^ 2, dims = 2)[:, 1]; binedges)
end;

# ╔═╡ c24e989d-62fa-49f2-885e-8c631d63b4d8
f(χ², n) = 1 / 2^(n / 2) / gamma(n / 2) * χ²^((n - 2) / 2) * exp(-χ² / 2)

# ╔═╡ 5207d33a-e9cf-412f-86e4-6f0e0ca53d75
begin
    plot(title = "chi2 distribution", xlab = "χ²")
    plot!(h1, seriestype = :stepbins, lab = "")
    plot!(h2, seriestype = :stepbins, lab = "")
    plot!(h3, seriestype = :stepbins, lab = "")
    plot!(h4, seriestype = :stepbins, lab = "")
    #
    plot!(χ² -> f(χ², 1), WithData(h1), lc = 1, lab = "")
    plot!(χ² -> f(χ², 2), WithData(h2), lc = 2, lab = "")
    plot!(χ² -> f(χ², 3), WithData(h3), lc = 3, lab = "")
    plot!(χ² -> f(χ², 4), WithData(h4), lc = 4, lab = "")
end

# cspell:disable

# ╔═╡ Cell order:
# ╟─85a7a141-79f6-4e75-9b97-0056e2d4a1c8
# ╠═a13e09e4-9c86-11ef-11e6-c5d0de162eac
# ╠═90f71c33-e5f3-4378-9bca-e044b5e3256c
# ╠═f559fd65-e5dc-43d6-b099-1ba62ab75f0e
# ╠═48c28514-b325-4d77-95d7-770e7572ef18
# ╠═5207d33a-e9cf-412f-86e4-6f0e0ca53d75
# ╠═c24e989d-62fa-49f2-885e-8c631d63b4d8
