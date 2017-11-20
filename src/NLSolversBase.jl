__precompile__(true)

module NLSolversBase

import Base: gradient
export AbstractObjective,
       NonDifferentiable,
       NonDifferentiableCache,
       OnceDifferentiable,
       OnceDifferentiableCache,
       TwiceDifferentiable,
       TwiceDifferentiableCache,
       iscomplex,
       real_to_complex,
       complex_to_real,
       value,
       value!,
       value_gradient!,
       gradient,
       gradient!,
       hessian,
       hessian!

include("complex_real.jl")
include("objective_types.jl")
include("interface.jl")

end # module
