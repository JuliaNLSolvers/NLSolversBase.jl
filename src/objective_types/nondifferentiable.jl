# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{TF,TX} <: AbstractObjective where {TF, TX}
    f
    F::TF
    x_f::TX
    f_calls::Vector{Int}
end

# These could be the same if we could restrict g below not to be an AbstractArray
function NonDifferentiable(f, x::AbstractArray, F::Real = real(zero(eltype(x))))
    NonDifferentiable{typeof(F),typeof(x)}(f, F, x_of_nans(x), [0,])
end
function NonDifferentiable(f, x::AbstractArray, F::AbstractArray)
    NonDifferentiable{typeof(F),typeof(x)}(f, F, x_of_nans(x), [0,])
end

# this is the g referred to above!
NonDifferentiable(f, g,        x::AbstractArray, F::Union{AbstractArray, Real} = real(zero(eltype(x)))) = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, h,     x::TX, F) where TX  = NonDifferentiable(f, x, F)
NonDifferentiable(f, g, fg, h, x::TX, F) where TX  = NonDifferentiable(f, x, F)
