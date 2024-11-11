module DataAnalysisWS2425

using QuadGK, Parameters
using Random, Statistics
using FHist
using Optim
#
using Plots, RecipesBase
using Plots.PlotMeasures: mm

export gaussian_scaled, polynomial_scaled, breit_wigner_scaled, voigt_scaled
include("functions.jl")

export sample_rejection, sample_inversion
include("sampling.jl")

export fit_enll, extended_nll
include("fitting.jl")

export WithData, curvehistpulls
include("plotting_recipe.jl")

end # module DataAnalysisWS2425
