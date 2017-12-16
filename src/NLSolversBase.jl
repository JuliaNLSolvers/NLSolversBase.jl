__precompile__(true)

module NLSolversBase

import Base: gradient
export AbstractObjective,
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
       iscomplex,
       real_to_complex,
       complex_to_real,
       value,
       value!,
       value_gradient!,
       value_jacobian!,
       gradient,
       gradient!,
       jacobian,
       jacobian!,
       hessian,
       hessian!,
       value!!,
       value_gradient!!,
       value_jacobian!!,
       hessian!!

x_of_nans(x) = convert(typeof(x), fill(eltype(x)(NaN), size(x)...))

include("complex_real.jl")
include("objective_types/abstract.jl")
include("objective_types/nondifferentiable.jl")
include("objective_types/oncedifferentiable.jl")
include("objective_types/twicedifferentiable.jl")
include("interface.jl")

end # module
