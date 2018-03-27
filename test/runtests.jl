using NLSolversBase
using Compat, Compat.Test, Compat.Random, Compat.LinearAlgebra
using OptimTestProblems
MVP = OptimTestProblems.MultivariateProblems

include("objective_types.jl")
include("interface.jl")
include("incomplete.jl")
include("constraints.jl")
