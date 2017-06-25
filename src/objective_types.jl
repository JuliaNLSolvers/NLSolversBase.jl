@compat abstract type AbstractObjective end
real_to_complex(d::AbstractObjective, x) = iscomplex(d) ? real_to_complex(x) : x
complex_to_real(d::AbstractObjective, x) = iscomplex(d) ? complex_to_real(x) : x

function fix_order(storage_input, x_input, fun!, fun!_msg)
    _storage = copy(storage_input)
    _x = copy(x_input)
    fun!(_storage, _x)
    if _storage == storage_input && _x != x_input
        warn("Storage (g) and evaluation point (x) order has changed. The order is now $(fun!_msg)(storage, x) as opposed to the old $(fun!_msg)(x, storage). Changing the order and proceeding, but please change your code to use the new syntax.")
        return (storage, x) -> fun!(x, storage)
    else
        return (storage, x) -> fun!(storage, x)
    end
end
# Used for objectives and solvers where no gradient is available/exists
type NonDifferentiable{T<:Real} <: AbstractObjective
    f
    f_x::T
    last_x_f::Array{T}
    f_calls::Vector{Int}
    iscomplex::Bool #if true, must convert back on every f call
end
iscomplex(obj::NonDifferentiable) = obj.iscomplex
NonDifferentiable(f,f_x, last_x_f::Array, f_calls::Vector{Int}) = NonDifferentiable(f,f_x,last_x_f,f_calls,false) #compatibility with old constructor

function NonDifferentiable(f, x_seed::AbstractArray)
    iscomplex = eltype(x_seed) <: Complex
    if iscomplex
        x_seed = complex_to_real(x_seed)
    end
    NonDifferentiable(f, f(x_seed), copy(x_seed), [1], iscomplex)
end

# Used for objectives and solvers where the gradient is available/exists
type OnceDifferentiable{T<:Real, Tgrad} <: AbstractObjective
    f
    g!
    fg!
    f_x::T
    g::Tgrad
    last_x_f::Array{T}
    last_x_g::Array{T}
    f_calls::Vector{Int}
    g_calls::Vector{Int}
    iscomplex::Bool #if true, must convert back on every f and g call
end
iscomplex(obj::OnceDifferentiable) = obj.iscomplex
OnceDifferentiable(f,g!,fg!,f_x::T, g::Tgrad, last_x_f::Array, last_x_g::Array, f_calls::Vector{Int}, g_calls::Vector{Int}) where {T, Tgrad} = OnceDifferentiable(f,g!,fg!,f_x, g, last_x_f, last_x_g, f_calls, g_calls, false) #compatibility with old constructor

# The user friendly/short form OnceDifferentiable constructor
function OnceDifferentiable(f, g!, fg!, x_seed::AbstractArray)
    iscomplex = eltype(x_seed) <: Complex
    g = similar(x_seed)

     _new_g! = fix_order(g, x_seed, g!,   "g!")
    _new_fg! = fix_order(g, x_seed, fg!, "fg!")
    f_val = _new_fg!(g, x_seed)
    if iscomplex
        x_seed = complex_to_real(x_seed)
        g = complex_to_real(g)
    end
    OnceDifferentiable(f, _new_g!, _new_fg!, f_val, g, copy(x_seed), copy(x_seed), [1], [1], iscomplex)
end
# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, g!, x_seed::AbstractArray)
    g = x_seed+one(eltype(x_seed))

    _new_g! = fix_order(g, x_seed, g!, "g!")

    function fg!(storage, x)
        _new_g!(storage, x)
        return f(x)
    end
    return OnceDifferentiable(f, _new_g!, fg!, x_seed)
end

# Used for objectives and solvers where the gradient and Hessian is available/exists
type TwiceDifferentiable{T<:Real} <: AbstractObjective
    f
    g!
    fg!
    h!
    f_x::T
    g::Vector{T}
    H::Matrix{T}
    last_x_f::Vector{T}
    last_x_g::Vector{T}
    last_x_h::Vector{T}
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

function TwiceDifferentiable{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    g = x_seed+one(T)
    H = Array{T}(n_x, n_x)

    _new_g!  = fix_order(g, x_seed, g!,   "g!")
    _new_fg! = fix_order(g, x_seed, fg!, "fg!")

    local _new_h!
    try
        _H = copy(H)+one(T)
        _x = copy(x_seed)
        h!(_H, _x)
        _new_h! = (storage, x) -> h!(storage, x)
    catch m
        if isa(m, MethodError) || isa(m, BoundsError)
            warn("Storage and evaluation point order has changed. The syntax is now h!(storage, x) as opposed to the old h!(x, storage). Your Hessian appears to have it the wrong way around. Changing the order and proceeding, but please change your code to use the new syntax.")
            _new_h! = (storage, x) -> h!(x, storage)
        else
            rethrow(m)
        end
    end

    f_val = _new_fg!(g, x_seed)
    _new_h!(H, x_seed)

    TwiceDifferentiable(f, _new_g!, _new_fg!, _new_h!, f_val,
                                g, H, copy(x_seed),
                                copy(x_seed), copy(x_seed), [1], [1], [1])
end
# Automatically create the fg! helper function if only f, g! and h! is provided
function TwiceDifferentiable{T}(f,
                                 g!,
                                 h!,
                                 x_seed::Array{T})
    g = x_seed+one(T)
    _new_g! = fix_order(g, x_seed, g!,   "g!")

    function fg!(storage::Vector, x::Vector)
        _new_g!(storage, x)
        return f(x)
    end
    return TwiceDifferentiable(f, _new_g!, fg!, h!, x_seed)
end
