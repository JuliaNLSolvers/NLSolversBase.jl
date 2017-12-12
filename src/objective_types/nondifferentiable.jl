# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{T,A<:AbstractArray{T},Tcplx} <: AbstractObjective where {T<:Real,
                                                            Tcplx<:Union{Val{true},Val{false}}  #if true, must convert back on every f call
                                                            }
    f
    F::T
    x_f::A
    f_calls::Vector{Int}
end
iscomplex(obj::NonDifferentiable{T,A,Val{true}}) where {T,A} = true
iscomplex(obj::NonDifferentiable{T,A,Val{false}}) where {T,A} = false
NonDifferentiable(f,f_x::T, x_f::AbstractArray{T}, f_calls::Vector{Int}) where {T} = NonDifferentiable{T,typeof(x_f),Val{false}}(f,f_x,x_f,f_calls) #compatibility with old constructor

function NonDifferentiable(f, F::T, x::AbstractArray{T}) where T
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
    end
    NonDifferentiable{eltype(x),typeof(x),Val{iscomplex}}(f, F, x_of_nans(x), [0])
end

NonDifferentiable(f, g, F::T, x::AbstractArray{T}) where T = NonDifferentiable(f, F, x)
NonDifferentiable(f, g, fg, h, F::T, x::AbstractArray{T}) where T = NonDifferentiable(f, F, x)
NonDifferentiable(f, g, h, F::T, x::AbstractArray{T}) where T = NonDifferentiable(f, F, x)
