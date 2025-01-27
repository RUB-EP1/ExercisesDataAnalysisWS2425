### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 7b877f4a-dcf1-11ef-15e9-f56f02622ad0
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	Pkg.instantiate()
	
	using Statistics
    using DataFrames
	using StatsBase: Weights
    using PlutoUI
    using XGBoost
	using Random
    using Plots
end

# ╔═╡ 1dd996a9-097d-4e89-910e-83eb78288c38
md"""
# Lecture 14b: Rewighting with GB

This notebook implements a simple algorithm that computes weights to make the _source_ distibution look like the _target_ distribution.

- A GBDT classifier is trained on combination of source/target distribution, using label 1 for _target_, and 0 for _source_.
- Then, the weight for ith event in _source_ distribution is computed as follows $p_i/(1-p_i)$.
"""

# ╔═╡ 9fbffbbe-9b29-4ba0-b5a5-ad5b3563c6f1
theme(:boxed)

# ╔═╡ 7204604e-1024-46f9-90a6-1ea9833e43ae
md"""
## Generate data
"""

# ╔═╡ 18105b59-2404-4d1e-bd21-63cc344f1210
begin
	Random.seed!(42)
	n_original = 20_000
	n_target = 40_000
end;

# ╔═╡ 53ed960e-447d-484d-8bf8-33e2770d0e70
# Original distribution: N(0, 1)
original = randn(n_original, 2) .* [1.0 1.5] .+ [0.0 0.0];

# ╔═╡ 1cd0cf89-e28b-492b-8e07-5304acf461e6
# Target distribution: N(1, 1.5)
target = randn(n_target, 2) .* [0.7 0.9] .+ [1.0 0.5];

# ╔═╡ fc22713e-e85f-436b-995e-a0c93732ea49
begin
	# Create combined dataset with labels
	X = vcat(original, target)
	y = vcat(zeros(n_original), ones(n_target))
end;

# ╔═╡ 706ef079-07e7-4ab4-84c2-db5644bf5407
# Create class-balanced sample weights
sample_weights = let
	w = (n_original+n_target)/2
	vcat(
	    fill(w/n_original, n_original),
	    fill(w/n_target, n_target)
	);
end;

# ╔═╡ edc35449-ad95-463d-b4ab-eadbfffd7ef4
begin
	# Convert to DMatrix with weights
	dtrain = DMatrix(X, label=y)
	XGBoost.setinfo!(dtrain, :weight, sample_weights)
end;

# ╔═╡ 7ae778cc-2baa-4f29-ba4e-de66ba0d65b0
# Configure XGBoost parameters
params = (
    max_depth = 5,
    η = 0.1,
    objective = "binary:logistic",
    eval_metric = "logloss",
    lambda = 1.0,  # L2 regularization
    alpha = 0.1,   # L1 regularization
    subsample = 0.8
);

# ╔═╡ 1799f468-0a78-47db-b93c-7a4b0e9f152f
# Train the model
model = xgboost(dtrain; num_round=100, verbosity=0, params...)

# ╔═╡ b6504b20-9e75-448a-a962-9d6c3791d43e
# Predict probabilities for original samples
p_original = predict(model, original)

# ╔═╡ d5645e18-b146-4252-96bb-448c317de60c
weights = let # Calculate importance weights
	_weights = p_original ./ (1 .- p_original .+ 1e-8)  # Add small epsilon for numerical stability
	# Normalize weights to target sample size
	_weights *= n_target / sum(_weights)
	_weights
end

# ╔═╡ 787e17be-95c6-4918-93e5-ed2bcbb3de73
function compare_distributions(original, target, weights)
    println("Feature 1:")
    println("- Original weighted mean: ", mean(original[:, 1], Weights(weights)))
    println("- Target mean:            ", mean(target[:, 1]))
    
    println("\nFeature 2:")
    println("- Original weighted mean: ", mean(original[:, 2], Weights(weights)))
    println("- Target mean:            ", mean(target[:, 2]))

    println("\nFeature 3:")
    println("- Original weighted std: ", std(original[:, 1], Weights(weights)))
    println("- Target std:            ", std(target[:, 1]))
end

# ╔═╡ 4037d1cc-395f-44da-b319-578eeca24bf5
compare_distributions(original, target, weights)

# ╔═╡ 2c7967ef-984e-46b6-bfab-d1d194a87a1f
begin
	bins = range(-5,5,70)
	# Optional visualization
	plot(layout=grid(1,2), size=(800,300))
	stephist!(original[:, 2]; sp=1, bins, label="Original", alpha=0.6)
	stephist!(original[:, 2]; sp=1, weights, bins, label="Reweighted Original", alpha=0.6)
	stephist!(target[:, 2]; sp=1, bins, label="Target", alpha=0.6)
	title!(sp=1, "Feature 1")
	# 
	stephist!(original[:, 1]; sp=2, bins, label="Original", alpha=0.6)
	stephist!(original[:, 1]; sp=2, weights, bins, label="Reweighted Original", alpha=0.6)
	stephist!(target[:, 1]; sp=2, bins, label="Target", alpha=0.6)
	title!(sp=2, "Feature 2")
end

# ╔═╡ Cell order:
# ╟─1dd996a9-097d-4e89-910e-83eb78288c38
# ╠═7b877f4a-dcf1-11ef-15e9-f56f02622ad0
# ╠═9fbffbbe-9b29-4ba0-b5a5-ad5b3563c6f1
# ╟─7204604e-1024-46f9-90a6-1ea9833e43ae
# ╠═18105b59-2404-4d1e-bd21-63cc344f1210
# ╠═53ed960e-447d-484d-8bf8-33e2770d0e70
# ╠═1cd0cf89-e28b-492b-8e07-5304acf461e6
# ╠═fc22713e-e85f-436b-995e-a0c93732ea49
# ╠═706ef079-07e7-4ab4-84c2-db5644bf5407
# ╠═edc35449-ad95-463d-b4ab-eadbfffd7ef4
# ╠═7ae778cc-2baa-4f29-ba4e-de66ba0d65b0
# ╠═1799f468-0a78-47db-b93c-7a4b0e9f152f
# ╠═b6504b20-9e75-448a-a962-9d6c3791d43e
# ╠═d5645e18-b146-4252-96bb-448c317de60c
# ╠═787e17be-95c6-4918-93e5-ed2bcbb3de73
# ╠═4037d1cc-395f-44da-b319-578eeca24bf5
# ╟─2c7967ef-984e-46b6-bfab-d1d194a87a1f
