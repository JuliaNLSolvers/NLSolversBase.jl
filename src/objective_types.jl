abstract type AbstractObjective end
abstract type AbstractObjectiveCache end
real_to_complex(c::AbstractObjectiveCache, x) = iscomplex(c) ? real_to_complex(x) : x
complex_to_real(c::AbstractObjectiveCache, x) = iscomplex(c) ? complex_to_real(x) : x

# Used for objectives and solvers where no gradient is available/exists
struct NonDifferentiable <: AbstractObjective
    f
end

mutable struct NonDifferentiableCache{T,A<:AbstractArray{T},Tcplx} <: AbstractObjectiveCache where {T<:Real,
                                                                    Tcplx<:Union{Val{true},Val{false}} # true is complex x; must convert back on every f call
                                                                    }
    f_x::T
    last_x_f::A
    f_calls::Vector{Int}
end
iscomplex(obj::NonDifferentiableCache{T,A,Val{true}}) where {T,A} = true
iscomplex(obj::NonDifferentiableCache{T,A,Val{false}}) where {T,A} = false
function NonDifferentiableCache(f, x_seed::AbstractArray)
    x_complex = eltype(x_seed) <: Complex
    if x_complex
        x_seed = complex_to_real(x_seed)
    end
    NonDifferentiableCache{eltype(x_seed),typeof(x_seed),Val{x_complex}}(f(x_seed), copy(x_seed), [1])
end

# Used for objectives and solvers where the gradient is available/exists
struct OnceDifferentiable <: AbstractObjective
    f
    g!
    fg!
end
# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, g!)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    return OnceDifferentiable(f, g!, fg!)
end

mutable struct OnceDifferentiableCache{T, Tgrad, A<:AbstractArray{T}, Tcplx} <: AbstractObjectiveCache where {T<:Real, Tgrad, Tcplx<:Union{Val{true},Val{false}}}
    f_x::T
    g::Tgrad
    last_x_f::A
    last_x_g::A
    f_calls::Vector{Int}
    g_calls::Vector{Int}
end
iscomplex(obj::OnceDifferentiableCache{T,Tgrad,A,Val{true}}) where {T,Tgrad,A} = true
iscomplex(obj::OnceDifferentiableCache{T,Tgrad,A,Val{false}}) where {T,Tgrad,A} = false
function OnceDifferentiableCache(f, g!, x_seed::AbstractArray)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    OnceDifferentiableCache(f, g!, fg!, x_seed)
end
function OnceDifferentiableCache(f, g!, fg!, x_seed::AbstractArray)
    x_complex = eltype(x_seed) <: Complex
    g = similar(x_seed)
    f_val = fg!(g, x_seed)

    if x_complex
        x_seed = complex_to_real(x_seed)
        g = complex_to_real(g)
    end
    OnceDifferentiableCache{eltype(x_seed),typeof(g),typeof(x_seed),Val{x_complex}}(f_val, g, copy(x_seed), copy(x_seed), [1], [1])
end

# Used for objectives and solvers where the gradient and Hessian is available/exists
struct TwiceDifferentiable <: AbstractObjective
    f
    g!
    fg!
    h!
end
# Automatically create the fg! helper function if only f, g! and h! is provided
function TwiceDifferentiable(f, g!, h!)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    return TwiceDifferentiable(f, g!, fg!, h!)
end

mutable struct TwiceDifferentiableCache{T<:Real,Tgrad,Thess,A<:AbstractArray{T}} <: AbstractObjectiveCache
    f_x::T
    g::Tgrad
    H::Thess
    last_x_f::A
    last_x_g::A
    last_x_h::A
    f_calls::Vector{Int}
    g_calls::Vector{Int}
    h_calls::Vector{Int}
end
iscomplex(obj::TwiceDifferentiableCache) = false
# The user friendly/short form TwiceDifferentiableCache constructor
function TwiceDifferentiableCache(f, g!, h!, x_seed::AbstractArray)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    TwiceDifferentiableCache(f, g!, fg!, h!, x_seed)
end
function TwiceDifferentiableCache(f, g!, fg!, h!, x_seed::AbstractArray{T}) where T
    n_x = length(x_seed)

    g = similar(x_seed)
    H = Array{T}(n_x, n_x)

    f_val = fg!(g, x_seed)
    h!(H, x_seed)

    TwiceDifferentiableCache(f_val, g, H, copy(x_seed), copy(x_seed), copy(x_seed), [1], [1], [1])
end