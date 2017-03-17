module NLSolversBase

using Compat

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
include("interface.jl")

end # module
