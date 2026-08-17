module NLSolversBase

using ADTypes: AbstractADType, AutoFiniteDiff
import DifferentiationInterface as DI
using FiniteDiff: FiniteDiff
using LinearAlgebra: LinearAlgebra

export AbstractObjective,
       NonDifferentiable,
       OnceDifferentiable,
       TwiceDifferentiable,
       value,
       value!,
       value_gradient!,
       value_gradient_hessian!,
       value_jacobian!,
       value_jvp!,
       gradient,
       gradient!,
       gradient_hessian!,
       jacobian,
       jacobian!,
       jvp!,
       hessian,
       hessian!,
       value!!,
       value_gradient!!,
       value_jacobian!!,
       hessian!!,
       hvp!,
       only_fg!,
       only_fgh!,
       only_fj!,
       only_fg,
       only_fj,
       only_g_and_fg,
       only_j_and_fj,
       only_fg_and_hvp!,
       only_fghvp!,
       clear!,
       f_calls,
       g_calls,
       jvp_calls,
       h_calls,
       hvp_calls

export AbstractConstraints, OnceDifferentiableConstraints,
    TwiceDifferentiableConstraints, ConstraintBounds

x_of_nans(x::AbstractArray, ::Type{Tf}=float(eltype(x))) where {Tf} = fill!(similar(x, Tf), NaN)

# Differentiation is prepared once, at construction, and then reused for every evaluation. Preparation has to
# see the array type the evaluations will use, and that is the type of the x caches, not necessarily the type
# of the array the caller handed us: the caches come from `similar`, which can change the container (a
# ReinterpretArray becomes an Array). The caller's values are carried over because preparation evaluates the
# objective, so a cache full of NaNs is not a safe point to prepare at.
x_of_values(x::AbstractArray) = copyto!(x_of_nans(x), x)

include("objective_types/inplace_factory.jl")
include("objective_types/abstract.jl")
include("objective_types/nondifferentiable.jl")
include("objective_types/oncedifferentiable.jl")
include("objective_types/twicedifferentiable.jl")
include("objective_types/incomplete.jl")
include("objective_types/constraints.jl")
include("interface.jl")

NonDifferentiable(f::OnceDifferentiable, x::AbstractArray) = NonDifferentiable(f.f, x, copy(f.F))
NonDifferentiable(f::TwiceDifferentiable, x::AbstractArray) = NonDifferentiable(f.f, x, copy(f.F))

end # module
