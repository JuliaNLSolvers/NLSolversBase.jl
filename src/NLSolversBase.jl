module NLSolversBase

using Compat
using Calculus
using ForwardDiff
using ReverseDiff

import Base: gradient
export NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
       AbstractObjective,
       value,
       value!,
       value_gradient!,
       gradient,
       gradient!,
       hessian,
       hessian!

include("objective_types.jl")

end # module
