# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{TF,TX,Tcplx} <: AbstractObjective where {TF, TX,
                                                            Tcplx<:Union{Val{true},Val{false}}  #if true, must convert back on every f call
                                                            }
    f
    F::TF
    x_f::TX
    f_calls::Vector{Int}
end
iscomplex(obj::NonDifferentiable{T,A,Val{true}}) where {T,A} = true
iscomplex(obj::NonDifferentiable{T,A,Val{false}}) where {T,A} = false
NonDifferentiable(f, F::TF, x_f::TX, f_calls::Vector{Int}) where {TF, TX} =
    NonDifferentiable{TF, TX, Val{false}}(f, F, x_f, f_calls)

function NonDifferentiable(f, x::TX, F::TF = zero(eltype(x))) where {TX, TF}
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
    end
    NonDifferentiable{TF,TX,Val{iscomplex}}(f, F, x_of_nans(x), [0,])
end

NonDifferentiable(f, g,        x::AbstractArray{T}, F = real(zero(T))) where T = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, h,     x::TX, F) where TX  = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, fg, h, x::TX, F) where TX  = NonDifferentiable(f, x, F)
