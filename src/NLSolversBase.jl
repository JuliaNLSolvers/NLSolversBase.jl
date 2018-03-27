__precompile__(true)

module NLSolversBase

using Compat

import Compat.Distributed: clear!
import Compat.LinearAlgebra: gradient
export AbstractObjective,
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
       TwiceDifferentiableHV,
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
       hessian!!,
       hv_product,
       hv_product!,
       only_fg!,
       only_fj!,
       only_fg,
       only_fj,
       only_g_and_fg,
       only_j_and_fj,
       clear!

export AbstractConstraints, OnceDifferentiableConstraints,
    TwiceDifferentiableConstraints, ConstraintBounds

function x_of_nans(x)   # https://github.com/JuliaLang/julia/issues/26516
    out = copy(x)       # if above issue is fixed, revert.
    (out.=(eltype(x))(NaN); nothing)
    out
end

include("complex_real.jl")
include("objective_types/abstract.jl")
include("objective_types/nondifferentiable.jl")
include("objective_types/oncedifferentiable.jl")
include("objective_types/twicedifferentiable.jl")
include("objective_types/twicedifferentiablehv.jl")
include("objective_types/incomplete.jl")
include("objective_types/constraints.jl")
include("interface.jl")

end # module
