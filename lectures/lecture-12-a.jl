### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 79fab7f8-d195-11ef-06b5-6da83b68b6dd
# ╠═╡ show_logs = false
begin
	using Pkg: Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
	# 
	using Plots
    using QuadGK
	using Statistics
    using DataAnalysisWS2425
end

# ╔═╡ c5c9a6ed-8ea4-4136-96f1-a9bc9189699b
md"""
# Lecture 12a: weights

In this notebook, we discuss how weights can change distributions.
How one distribution can be turned into the other one using the weighting technique.

A concept of the _effective sample size (ESS)_ is introduced. We look how the effective size correlates with the efficiency of the **importance sampling**.
"""

# ╔═╡ 10a2217f-88e1-47f8-ab5a-0e61460c794d
theme(
    :wong2,
    grid = false,
    frame = :box,
    ylims = (0, :auto),
    xlims = (:auto, :auto),
    lab = "", colorbar=false,
    linealpha = 1,
);

# ╔═╡ 48aa08f4-260c-4ef7-be33-0d234ed8e32e
target(x) = exp(-x^4)

# ╔═╡ b4e77def-47e1-448a-ae4a-a5fc9df34f5d
n = 10_000

# ╔═╡ b35fb151-28b8-4eb4-80bd-82295f4e1d80
md"""
### Method 1
"""

# ╔═╡ dc525d92-4041-4e99-916e-754654c7a435
f1(x) = 0.5

# ╔═╡ 710d57ad-b042-44d5-b189-aabdcf20fc2e
data1 = 4 .* rand(n) .- 2;

# ╔═╡ 946b6550-eafd-403f-b7b2-4570fd2422cb
md"""
### Method 2
"""

# ╔═╡ 327f1bab-5ed1-464e-ae00-60abe8993047
const σ_gauss = 0.7;

# ╔═╡ e4ef8c63-8704-4604-aa94-055a60e0f8fc
f2(x) = exp(-x^2/(2*σ_gauss^2))

# ╔═╡ 0316daba-6ee1-4d03-adfc-b0c3fb0529a5
begin
	plot(target, -2, 2, lab="target")
	plot!(f1, -2, 2, fill=0, alpha=0.3, lab="flat")
	plot!(f2, -2, 2, fill=0, alpha=0.3, lab="gauss")
	
end

# ╔═╡ bbb54ea2-76a4-43a9-b02f-c98867927f1f
data2 = randn(n) .* σ_gauss;

# ╔═╡ c4d62ef6-4c2b-4521-a162-dc5cfc8e0e68
md"""
### Visualization
"""

# ╔═╡ 1abd5688-c1a3-4e25-a4f7-0a85ecedd90f
let
	bins=range(-2,2,50)
	plot(title = "original samples")
	stephist!(data1; bins, fill=0, alpha=0.3, lab="flat")
	stephist!(data2; bins, fill=0, alpha=0.3, lab="gauss")
end

# ╔═╡ c5ebe76d-1798-4e33-a5d1-f379ee5ea260
quadgk(target, -2, 2)[1]

# ╔═╡ 8331fe66-4d5f-4a0e-836d-e04532922bd1
md"""
### Comparison of ESS
"""

# ╔═╡ 6e8e0f44-6515-4a47-ba01-19dcefde5379
w1 = target.(data1) ./ f1.(data1);

# ╔═╡ 16747dba-6bcf-48dc-a304-f7a779200b7f
w2 = target.(data2) ./ f2.(data2);

# ╔═╡ 383f28c7-2e4f-41c9-b0a4-1d9654a6cbf6
let
	bins=range(-2,2,50)
	# 
	plot(title = "weighted samples")
	stephist!(data1, weights=w1 ./ sum(w1); bins, fill=0, alpha=0.3, lab="from flat")
	stephist!(data2, weights=w2 ./ sum(w2); bins, fill=0, alpha=0.3, lab="from gauss")
	# 
	integral = quadgk(target, bins[1], bins[end])[1]
	plot!(x->target(x)/integral, WithData(bins), l=(:black, 2), lab="target")
end

# ╔═╡ af9ebdca-316d-4845-86f4-21156d57ecf4
ESS(w) = sum(w)^2 / sum(w .^ 2) 

# ╔═╡ 8d01dc29-2158-4f6a-84cf-1b021ec4dc58
ESS(w1) / n # flat

# ╔═╡ 23c932dc-f3a4-47fb-aee4-afb64475d718
ESS(w2) / n # gauss

# ╔═╡ 9cf1afd0-b396-4c58-93e0-6c0a65ec38de
md"""
## Errors of the weighted histogram

To formalize a problem, we consider a rand variable $W$ computed as 

$W = \sum^\text{bin}_{i} w_i,$

We pull from a bag two numbers $(x_i,y_i)$. The weight is a function of the two variables: $w_i = w(x_i, y_i)$.
Depending on the variable $x$,
decide either this event falls into bin ($x_i\in\text{bin}$), or not .
The variable $y$ refers to other dimentions that the weight can depend upon.
"""

# ╔═╡ 36bb27f4-de32-45a4-90f8-f7643507d54d
md"""
In practice,

$\sqrt{\sum_i^{\text{bin}} w_i^2}$

is used for uncertainty of bin content with weighted distributions.
It give a conservative estimate of the variance of the weight sample scaled by expect event rate. Indeed, one can derive that

```math
\begin{align}
E(W) &= N p \,E(w)\,\\
V(W) &= N p \,V(w)\,\\
\end{align}
```
where E, and V stand for expectation and variance, respectively. They are computed for the sample within the $\text{bin}$. $N$ is the total number of event, and $p$ give the probability to fall into the $\text{bin}$.
"""

# ╔═╡ 460f0b4e-95d3-4e6a-81a9-70f75e4d7f35
md"""
## Importance sampling

ESS correlates with the efficiency of the importance sampling.
"""

# ╔═╡ 9a21b5f0-6542-41bd-8e56-5168cab9a73e
count(w1 .> rand(n) .* maximum(w1)) / n # for flat

# ╔═╡ c2677f1d-cad5-465a-b481-4ffc1a7e7473
count(w2 .> rand(n) .* maximum(w2)) / n # for gauss

# ╔═╡ Cell order:
# ╟─c5c9a6ed-8ea4-4136-96f1-a9bc9189699b
# ╠═79fab7f8-d195-11ef-06b5-6da83b68b6dd
# ╠═10a2217f-88e1-47f8-ab5a-0e61460c794d
# ╠═48aa08f4-260c-4ef7-be33-0d234ed8e32e
# ╠═0316daba-6ee1-4d03-adfc-b0c3fb0529a5
# ╠═b4e77def-47e1-448a-ae4a-a5fc9df34f5d
# ╟─b35fb151-28b8-4eb4-80bd-82295f4e1d80
# ╠═dc525d92-4041-4e99-916e-754654c7a435
# ╠═710d57ad-b042-44d5-b189-aabdcf20fc2e
# ╟─946b6550-eafd-403f-b7b2-4570fd2422cb
# ╠═327f1bab-5ed1-464e-ae00-60abe8993047
# ╠═e4ef8c63-8704-4604-aa94-055a60e0f8fc
# ╠═bbb54ea2-76a4-43a9-b02f-c98867927f1f
# ╟─c4d62ef6-4c2b-4521-a162-dc5cfc8e0e68
# ╟─1abd5688-c1a3-4e25-a4f7-0a85ecedd90f
# ╠═383f28c7-2e4f-41c9-b0a4-1d9654a6cbf6
# ╠═c5ebe76d-1798-4e33-a5d1-f379ee5ea260
# ╟─8331fe66-4d5f-4a0e-836d-e04532922bd1
# ╠═6e8e0f44-6515-4a47-ba01-19dcefde5379
# ╠═16747dba-6bcf-48dc-a304-f7a779200b7f
# ╠═af9ebdca-316d-4845-86f4-21156d57ecf4
# ╠═8d01dc29-2158-4f6a-84cf-1b021ec4dc58
# ╠═23c932dc-f3a4-47fb-aee4-afb64475d718
# ╟─9cf1afd0-b396-4c58-93e0-6c0a65ec38de
# ╟─36bb27f4-de32-45a4-90f8-f7643507d54d
# ╟─460f0b4e-95d3-4e6a-81a9-70f75e4d7f35
# ╠═9a21b5f0-6542-41bd-8e56-5168cab9a73e
# ╠═c2677f1d-cad5-465a-b481-4ffc1a7e7473
