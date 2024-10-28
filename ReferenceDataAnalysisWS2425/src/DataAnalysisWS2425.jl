module DataAnalysisWS2425
using QuadGK, Random, Statistics

export gaussian_scaled, polynomial_scaled, breit_wigner_scaled, voigt_scaled
include("functions.jl")

export sample_rejection, sample_inversion
include("sampling.jl")

end # module DataAnalysisWS2425
