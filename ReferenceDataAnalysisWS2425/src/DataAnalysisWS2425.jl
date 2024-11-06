module DataAnalysisWS2425
using QuadGK, Random, Statistics
using FHist
using RecipesBase

export gaussian_scaled, polynomial_scaled, breit_wigner_scaled, voigt_scaled
include("functions.jl")

export sample_rejection, sample_inversion
include("sampling.jl")

export WithData
include("plotting_recipe.jl")

end # module DataAnalysisWS2425
