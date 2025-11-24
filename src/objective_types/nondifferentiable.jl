# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{TF<:Union{AbstractArray,Real},TX<:AbstractArray} <: AbstractObjective
    const f
    F::TF
    const x_f::TX
    f_calls::Int
end

# These could be the same if we could restrict g below not to be an AbstractArray
function NonDifferentiable(f, x::AbstractArray, F::Real = real(zero(eltype(x))); inplace = true)
    xnans = x_of_nans(x)
    NonDifferentiable{typeof(F),typeof(xnans)}(f, F, xnans, 0)
end
function NonDifferentiable(f, x::AbstractArray, F::AbstractArray; inplace = true)
    f = inplace ? f : f!_from_f(f, F, inplace)
    xnans = x_of_nans(x)
    NonDifferentiable{typeof(F),typeof(xnans)}(f, F, xnans, 0)
end

# this is the g referred to above!
NonDifferentiable(f, g,        x::AbstractArray, F::Union{AbstractArray, Real} = real(zero(eltype(x)))) = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, h,     x::TX, F) where TX  = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, fg, h, x::TX, F) where TX  = NonDifferentiable(f, x, F)
