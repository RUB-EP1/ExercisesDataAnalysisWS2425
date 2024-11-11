### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ a1f98e16-9db3-11ef-3017-d7ffb849f806
begin
    using ForwardDiff
    using QuadGK
end

# ╔═╡ c18fdcf1-ce8e-4b43-9f08-7f5415ec2888
md"""
# Forward Diff and Dual numbers

In this notebook, we explore how the automatic differentiation works in the forward mode. We will use `ForwardDiff.jl` as an reference implementation.
"""

# ╔═╡ 44d61456-ec9c-4a06-9ea2-9b838c629ef1
x0 = MathConstants.e |> Float64

# ╔═╡ bf742e73-cf6d-498f-9336-88eaba68ac39
x0_dual = ForwardDiff.Dual(x0, 1.0)

# ╔═╡ ef9806df-c423-43f1-9b35-06bc3c543776
md"""
## Mathrmatical operations
"""

# ╔═╡ 1283bbd2-e7e6-4cb1-a35e-955cecaf6793
x0_dual + 5 # summation

# ╔═╡ 4526305d-9266-48dd-ad2c-777342467f5d
x0_dual * 2.2 # multiplication

# ╔═╡ bb030fc8-b2b0-42eb-82fc-fd907c17587a
sin(x0_dual) # to make it work, someone had to define `sin(x::Dual)`

# ╔═╡ 82ee357e-77cb-4af9-a70c-3401a78a3c21
x0_dual + x0_dual # adding several dial numbers

# ╔═╡ f5fb3db5-a2ac-4311-905b-8cf7465c46b4
md"""
## Computations with Dual
"""

# ╔═╡ b87372c7-b858-4672-b84f-b293d0d02aeb
f(x) = x^5 + 3x * log(x)

# ╔═╡ d7305bf4-b3c4-43d1-a2ab-1563907f4abb
analytic_f′(x) = 5x^4 + 3 + 3log(x)

# ╔═╡ 77973c4c-f114-4efd-8dd6-7993c1b8d202
md"""
### Compare to analytic value
"""

# ╔═╡ 99851d5a-3f94-45c3-a462-0fc9da8951ca
analytic_derivative = analytic_f′(x0)

# ╔═╡ e09e2d10-c928-4c9d-9615-feeacfb0e5e4
f(x0_dual).partials[1]

# ╔═╡ c2091f87-4ead-4db7-aab2-8205013ee736
from_dual_operations = let
	g(x) = f(x)
	g(x0_dual).partials[1]
end

# ╔═╡ 728ba2d1-f00b-417a-9bbe-5d9e310e514c
md"""
That is the number that gets returned can calling `ForwardDiff.gradient`
"""

# ╔═╡ 5458e12e-a6bd-4a73-a297-3eedab4f60cb
ForwardDiff.gradient(x -> f(x[1]), [x0])[1]

# ╔═╡ 29b21242-b58f-41cb-888b-d380baa3fd3b
from_dual_operations - analytic_derivative

# ╔═╡ 9dbf6f99-103e-4526-b94e-df1852d799ec
md"""
## Missing implementation
"""

# ╔═╡ c2782f7b-00f9-4a1f-9cc5-2d0e74a06563
md"""
$g(a) = \int_{1}^{2} f(x+a) dx$
"""

# ╔═╡ 6b622bd8-2644-4fb0-b7af-3865ee4f6bcd
g(a) = quadgk(x -> f(x + a), 1, 2)[1]

# ╔═╡ 3c54cd95-119b-4eef-b7a6-afcc3413023a
g(ForwardDiff.Dual(0, 1))

# ╔═╡ 04e118b3-17a7-4a02-b9c9-61ea02817d82
html"""
<span style="font-size: 30px;">😲 </span>
"""

# ╔═╡ 4460a0e2-4f2e-4827-aa7c-8151110f9dd8
f(2) - f(1)

# ╔═╡ db9592f3-b813-42b5-b38a-9751eeb129ab
md"""
### Non implemented interface
"""

# ╔═╡ f2cca306-93cf-4bd7-ae70-b1484d7fd7e7
h(a) = quadgk(x -> f(x), 1, a)[1]

# ╔═╡ 755c6301-52fa-4aab-b650-0a0e42552f5a
# h(ForwardDiff.Dual(2,1))

# ╔═╡ a764f59b-e791-43ab-a6bf-efbfe0531c0e
md"""
### Impossible type propagation
"""

# ╔═╡ 883eaea0-e9ff-4355-acee-43a7867d9180
struct MyModelPars
    m::Float64
    b::Float64
end

# ╔═╡ 52ddbb24-5bf6-476b-94a5-3c268988311c
function model(p)
    p.m^2 + log(p.b)
end

# ╔═╡ 07af670c-ad38-4aab-a38d-f087111777a6
t_fixed(x) = let b=2
	model(MyModelPars(x, 2))
end

# ╔═╡ d7eb7e7a-b976-415f-9e90-75728f0aadba
# gives the errors
# 
# let b = 2
# 	ForwardDiff.gradient(x->t_fixed(x[1]), [1.1])
# end

# ╔═╡ 8711e365-ee7b-4af5-a9f7-4ac874b032c5
# gives the errors
# MyModelPars(x0_dual,2.0)

# ╔═╡ 4d1b86e0-ea90-450d-8c4b-eacb36c2ba34
md"""
How to fix? - use parametric types
"""

# ╔═╡ df889849-a6c2-47de-85b5-d8fac9368d2f
struct MyModelPars2{M, B}
    m::M
    b::B
end

# ╔═╡ e910d00c-ddce-4471-acf2-61976ac3f681
t_parametric(x) = let b=2
	model(MyModelPars2(x, b))
end

# ╔═╡ de2fcc1c-af77-4257-ab6d-f8ea0b7fab9b
ForwardDiff.gradient(x -> t_parametric(x[1]), [1.1])

# ╔═╡ 6295b28f-3d12-4094-a2ad-67aa625a2a9e
# cspell:disable

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"

[compat]
ForwardDiff = "~0.10.37"
QuadGK = "~2.11.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "c5d57612251a6d7a3b7ba3f1ecbbdcf380c0c13d"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a9ce73d3c827adab2d70bf168aaece8cce196898"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.37"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─c18fdcf1-ce8e-4b43-9f08-7f5415ec2888
# ╠═a1f98e16-9db3-11ef-3017-d7ffb849f806
# ╠═44d61456-ec9c-4a06-9ea2-9b838c629ef1
# ╠═bf742e73-cf6d-498f-9336-88eaba68ac39
# ╟─ef9806df-c423-43f1-9b35-06bc3c543776
# ╠═1283bbd2-e7e6-4cb1-a35e-955cecaf6793
# ╠═4526305d-9266-48dd-ad2c-777342467f5d
# ╠═bb030fc8-b2b0-42eb-82fc-fd907c17587a
# ╠═82ee357e-77cb-4af9-a70c-3401a78a3c21
# ╟─f5fb3db5-a2ac-4311-905b-8cf7465c46b4
# ╠═b87372c7-b858-4672-b84f-b293d0d02aeb
# ╠═d7305bf4-b3c4-43d1-a2ab-1563907f4abb
# ╟─77973c4c-f114-4efd-8dd6-7993c1b8d202
# ╠═99851d5a-3f94-45c3-a462-0fc9da8951ca
# ╠═e09e2d10-c928-4c9d-9615-feeacfb0e5e4
# ╠═c2091f87-4ead-4db7-aab2-8205013ee736
# ╟─728ba2d1-f00b-417a-9bbe-5d9e310e514c
# ╠═5458e12e-a6bd-4a73-a297-3eedab4f60cb
# ╠═29b21242-b58f-41cb-888b-d380baa3fd3b
# ╟─9dbf6f99-103e-4526-b94e-df1852d799ec
# ╟─c2782f7b-00f9-4a1f-9cc5-2d0e74a06563
# ╠═6b622bd8-2644-4fb0-b7af-3865ee4f6bcd
# ╠═3c54cd95-119b-4eef-b7a6-afcc3413023a
# ╟─04e118b3-17a7-4a02-b9c9-61ea02817d82
# ╠═4460a0e2-4f2e-4827-aa7c-8151110f9dd8
# ╟─db9592f3-b813-42b5-b38a-9751eeb129ab
# ╠═f2cca306-93cf-4bd7-ae70-b1484d7fd7e7
# ╠═755c6301-52fa-4aab-b650-0a0e42552f5a
# ╟─a764f59b-e791-43ab-a6bf-efbfe0531c0e
# ╠═883eaea0-e9ff-4355-acee-43a7867d9180
# ╠═52ddbb24-5bf6-476b-94a5-3c268988311c
# ╠═07af670c-ad38-4aab-a38d-f087111777a6
# ╠═d7eb7e7a-b976-415f-9e90-75728f0aadba
# ╠═8711e365-ee7b-4af5-a9f7-4ac874b032c5
# ╟─4d1b86e0-ea90-450d-8c4b-eacb36c2ba34
# ╠═df889849-a6c2-47de-85b5-d8fac9368d2f
# ╠═e910d00c-ddce-4471-acf2-61976ac3f681
# ╠═de2fcc1c-af77-4257-ab6d-f8ea0b7fab9b
# ╠═6295b28f-3d12-4094-a2ad-67aa625a2a9e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
