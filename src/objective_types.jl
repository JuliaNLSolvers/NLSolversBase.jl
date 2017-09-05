abstract type AbstractObjective end
real_to_complex(d::AbstractObjective, x) = iscomplex(d) ? real_to_complex(x) : x
complex_to_real(d::AbstractObjective, x) = iscomplex(d) ? complex_to_real(x) : x

# Used for objectives and solvers where no gradient is available/exists
mutable struct NonDifferentiable{T,A<:AbstractArray{T},Tcplx} <: AbstractObjective where {T<:Real,
                                                            Tcplx<:Union{Val{true},Val{false}}  #if true, must convert back on every f call
                                                            }
    f
    f_x::T
    last_x_f::A
    f_calls::Vector{Int}
end
iscomplex(obj::NonDifferentiable{T,A,Val{true}}) where {T,A} = true
iscomplex(obj::NonDifferentiable{T,A,Val{false}}) where {T,A} = false
NonDifferentiable(f,f_x::T, last_x_f::AbstractArray{T}, f_calls::Vector{Int}) where {T} = NonDifferentiable{T,typeof(last_x_f),Val{false}}(f,f_x,last_x_f,f_calls) #compatibility with old constructor

function NonDifferentiable(f, x_seed::AbstractArray)
    iscomplex = eltype(x_seed) <: Complex
    if iscomplex
        x_seed = complex_to_real(x_seed)
    end
    NonDifferentiable{eltype(x_seed),typeof(x_seed),Val{iscomplex}}(f, f(x_seed), copy(x_seed), [1])
end

# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiable{T, Tgrad, A<:AbstractArray{T}, Tcplx} <: AbstractObjective where {T<:Real, Tgrad, Tcplx<:Union{Val{true},Val{false}}}
    f
    g!
    fg!
    f_x::T
    g::Tgrad
    last_x_f::A
    last_x_g::A
    f_calls::Vector{Int}
    g_calls::Vector{Int}
end
iscomplex(obj::OnceDifferentiable{T,Tgrad,A,Val{true}}) where {T,Tgrad,A} = true
iscomplex(obj::OnceDifferentiable{T,Tgrad,A,Val{false}}) where {T,Tgrad,A} = false
OnceDifferentiable(f,g!,fg!,f_x::T, g::Tgrad, last_x_f::A, last_x_g::A, f_calls::Vector{Int}, g_calls::Vector{Int}) where {T, Tgrad, A<:AbstractArray{T}} = OnceDifferentiable{T,Tgrad,A,Val{false}}(f,g!,fg!,f_x, g, last_x_f, last_x_g, f_calls, g_calls) #compatibility with old constructor

# The user friendly/short form OnceDifferentiable constructor
function OnceDifferentiable(f, g!, fg!, x_seed::AbstractArray)
    iscomplex = eltype(x_seed) <: Complex
    g = similar(x_seed)
    f_val = fg!(g, x_seed)

    if iscomplex
        x_seed = complex_to_real(x_seed)
        g = complex_to_real(g)
    end
    OnceDifferentiable{eltype(x_seed),typeof(g),typeof(x_seed),Val{iscomplex}}(f, g!, fg!, f_val, g, copy(x_seed), copy(x_seed), [1], [1])
end
# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, g!, x_seed::AbstractArray)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    return OnceDifferentiable(f, g!, fg!, x_seed)
end

# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T<:Real,Tgrad,A<:AbstractArray{T}} <: AbstractObjective
    f
    g!
    fg!
    h!
    f_x::T
    g::Tgrad
    H::Matrix{T}
    last_x_f::A
    last_x_g::A
    last_x_h::A
    f_calls::Vector{Int}
    g_calls::Vector{Int}
    h_calls::Vector{Int}
end
iscomplex(obj::TwiceDifferentiable) = false
# The user friendly/short form TwiceDifferentiable constructor
function TwiceDifferentiable(td::TwiceDifferentiable, x::AbstractArray)
    value_gradient!(td, x)
    hessian!(td, x)
    td
end

function TwiceDifferentiable(f, g!, fg!, h!, x_seed::AbstractArray{T}) where T
    n_x = length(x_seed)
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)

    f_val = fg!(g, x_seed)
    h!(H, x_seed)

    TwiceDifferentiable(f, g!, fg!, h!, f_val,
                                g, H, copy(x_seed),
                                copy(x_seed), copy(x_seed), [1], [1], [1])
end
# Automatically create the fg! helper function if only f, g! and h! is provided
function TwiceDifferentiable(f,
                             g!,
                             h!,
                             x_seed::AbstractArray{T}) where T
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    return TwiceDifferentiable(f, g!, fg!, h!, x_seed)
end
