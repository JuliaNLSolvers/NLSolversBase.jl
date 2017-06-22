__precompile__(true)

module NLSolversBase

using Compat

import Base: gradient
export AbstractObjective,
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
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
