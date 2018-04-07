__precompile__(true)

module NLSolversBase

import Base: gradient, clear!
export AbstractObjective,
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
       TwiceDifferentiableHV,
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
       hessian!!,
       hv_product,
       hv_product!,
       only_fg!,
       only_fgh!,
       only_fj!,
       only_fg,
       only_fj,
       only_g_and_fg,
       only_j_and_fj,
       clear!

export AbstractConstraints, OnceDifferentiableConstraints,
    TwiceDifferentiableConstraints, ConstraintBounds

x_of_nans(x) = copy(x).=(eltype(x))(NaN)

include("objective_types/abstract.jl")
include("objective_types/nondifferentiable.jl")
include("objective_types/oncedifferentiable.jl")
include("objective_types/twicedifferentiable.jl")
include("objective_types/twicedifferentiablehv.jl")
include("objective_types/incomplete.jl")
include("objective_types/constraints.jl")
include("interface.jl")

end # module
