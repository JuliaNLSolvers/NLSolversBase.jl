using NLSolversBase
using Base.Test
using OptimTestProblems
using RecursiveArrayTools
MVP = OptimTestProblems.MultivariateProblems

# TODO: Use OptimTestProblems
# TODO: MultivariateProblems.UnconstrainedProblems.exampples["Exponential"]

# Test example
function exponential(x)
    return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
end

function exponential_gradient!(storage, x)
    storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
end


function exponential_value_gradient!(storage, x)
    storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
end

function exponential_hessian!(storage, x)
    storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    storage[1, 2] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = 2.0 * exp((3.0 - x[1])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
end

include("objective_types.jl")
include("interface.jl")
include("incomplete.jl")
include("constraints.jl")
include("abstractarrays.jl")
include("sparse.jl")
