using NLSolversBase, Test
using Random
using LinearAlgebra: Diagonal, I, mul!
using ComponentArrays
using StaticArrays
using SparseArrays
using OptimTestProblems
using RecursiveArrayTools
using ADTypes
MVP = OptimTestProblems.MultivariateProblems

import Aqua
import ExplicitImports
import JET

# TODO: Use OptimTestProblems (but it does not have exponential_gradient_hession etc.)
# TODO: MultivariateProblems.UnconstrainedProblems.examples["Exponential"]

# Test example
function exponential(x)
    return exp((2 - x[1])^2) + exp((3 - x[2])^2)
end

function exponential_gradient!(G, x)
    G[1] = -2 * (2 - x[1]) * exp((2 - x[1])^2)
    G[2] = -2 * (3 - x[2]) * exp((3 - x[2])^2)
    G
end


function exponential_gradient(x)
    G = similar(x)
    G[1] = -2 * (2 - x[1]) * exp((2 - x[1])^2)
    G[2] = -2 * (3 - x[2]) * exp((3 - x[2])^2)
    G
end


function exponential_value_gradient!(storage, x)
    storage[1] = -2 * (2 - x[1]) * exp((2 - x[1])^2)
    storage[2] = -2 * (3 - x[2]) * exp((3 - x[2])^2)
    return exp((2 - x[1])^2) + exp((3 - x[2])^2)
end

function exponential_value_gradient(x)
    return exponential(x), exponential_gradient(x)
end

function exponential_gradient_hessian(x)
    F = similar(x)
    F[1] = -2 * (2 - x[1]) * exp((2 - x[1])^2)
    F[2] = -2 * (3 - x[2]) * exp((3 - x[2])^2)

    nx = length(x)
    J = fill!(x*x', 0)
    J[1, 1] = 2 * exp((2 - x[1])^2) * (2 * x[1]^2 - 8 * x[1] + 9)
    J[1, 2] = 0
    J[2, 1] = 0
    J[2, 2] = 2 * exp((3 - x[1])^2) * (2 * x[2]^2 - 12 * x[2] + 19)
    F, J
end

function exponential_hessian(x)
    nx = length(x)
    J = fill!(x*x', 0)
    J[1, 1] = 2 * exp((2 - x[1])^2) * (2 * x[1]^2 - 8 * x[1] + 9)
    J[1, 2] = 0
    J[2, 1] = 0
    J[2, 2] = 2 * exp((3 - x[1])^2) * (2 * x[2]^2 - 12 * x[2] + 19)
    J
end

function exponential_hessian!(J, x)
    J[1, 1] = 2 * exp((2 - x[1])^2) * (2 * x[1]^2 - 8 * x[1] + 9)
    J[1, 2] = 0
    J[2, 1] = 0
    J[2, 2] = 2 * exp((3 - x[1])^2) * (2 * x[2]^2 - 12 * x[2] + 19)
    J
end

function exponential_hessian_product!(J, x)
    J[1, 1] = 2 * exp((2 - x[1])^2) * (2 * x[1]^2 - 8 * x[1] + 9)
    J[1, 2] = 0
    J[2, 1] = 0
    J[2, 2] = 2 * exp((3 - x[1])^2) * (2 * x[2]^2 - 12 * x[2] + 19)
    J
end

@testset verbose=true "NLSolversBase.jl" begin
    include("objective_types.jl")
    include("interface.jl")
    include("incomplete.jl")
    include("constraints.jl")
    include("abstractarrays.jl")
    include("autodiff.jl")
    include("sparse.jl")
    include("kwargs.jl")
    include("utils.jl")
    include("qa.jl")
end
