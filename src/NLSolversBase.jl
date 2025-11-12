module NLSolversBase

using ADTypes: AbstractADType, AutoFiniteDiff
import DifferentiationInterface as DI
using FiniteDiff: FiniteDiff
using LinearAlgebra: LinearAlgebra

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
       only_fg_and_hv!,
       only_fghv!,
       clear!,
       f_calls,
       g_calls,
       h_calls,
       hv_calls

export AbstractConstraints, OnceDifferentiableConstraints,
    TwiceDifferentiableConstraints, ConstraintBounds

x_of_nans(x::AbstractArray, ::Type{Tf}=float(eltype(x))) where {Tf} = fill!(similar(x, Tf), NaN)

include("objective_types/inplace_factory.jl")
include("objective_types/abstract.jl")
include("objective_types/nondifferentiable.jl")
include("objective_types/oncedifferentiable.jl")
include("objective_types/twicedifferentiable.jl")
include("objective_types/twicedifferentiablehv.jl")
include("objective_types/incomplete.jl")
include("objective_types/constraints.jl")
include("interface.jl")

NonDifferentiable(f::OnceDifferentiable, x::AbstractArray) = NonDifferentiable(f.f, x, copy(f.F))
NonDifferentiable(f::TwiceDifferentiable, x::AbstractArray) = NonDifferentiable(f.f, x, copy(f.F))
NonDifferentiable(f::TwiceDifferentiableHV, x::AbstractArray) = NonDifferentiable(f.f, x, copy(f.F))
end # module
