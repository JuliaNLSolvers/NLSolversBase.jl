__precompile__(true)

module NLSolversBase

using Compat

import Base: gradient
export AbstractObjective,
       UninitializedObjective,
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
       UninitializedNonDifferentiable,
       UninitializedOnceDifferentiable,
       UninitializedTwiceDifferentiable,
       value,
       value!,
       value_gradient!,
       gradient,
       gradient!,
       hessian,
       hessian!

include("objective_types.jl")
include("interface.jl")

end # module
