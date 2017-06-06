@compat abstract type AbstractObjective end

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
type NonDifferentiable{T} <: AbstractObjective
    f
    f_x::T
    last_x_f::Array{T}
    f_calls::Vector{Int}
end
type UnitializedNonDifferentiable <: AbstractObjective
    f
end
# The user friendly/short form NonDifferentiable constructor
NonDifferentiable(f) = UnitializedNonDifferentiable(f)
NonDifferentiable{T}(f, x_seed::Array{T}) = NonDifferentiable(f, f(x_seed), copy(x_seed), [1])

# Used for objectives and solvers where the gradient is available/exists
type OnceDifferentiable{T, Tgrad} <: AbstractObjective
    f
    g!
    fg!
    f_x::T
    g::Tgrad
    last_x_f::Array{T}
    last_x_g::Array{T}
    f_calls::Vector{Int}
    g_calls::Vector{Int}
end
type UnitializedOnceDifferentiable <: AbstractObjective
    f
    g!
    fg!
end
# The user friendly/short form OnceDifferentiable constructor
OnceDifferentiable(f, g!, fg!) = UnitializedOnceDifferentiable(f, g!,      fg!)
OnceDifferentiable(f, g!)      = UnitializedOnceDifferentiable(f, g!,      nothing)
OnceDifferentiable(f)          = UnitializedOnceDifferentiable(f, nothing, nothing)

function OnceDifferentiable(f, g!, fg!, x_seed::AbstractArray)
    g = similar(x_seed)

     _new_g! = fix_order(g, x_seed, g!,   "g!")
    _new_fg! = fix_order(g, x_seed, fg!, "fg!")

    f_val = _new_fg!(g, x_seed)
    OnceDifferentiable(f, _new_g!, _new_fg!, f_val, g, copy(x_seed), copy(x_seed), [1], [1])
end

# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, g!, x_seed::AbstractArray)
    g = similar(x_seed)

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
type UnitializedTwiceDifferentiable <: AbstractObjective
    f
    g!
    fg!
    h!
end
TwiceDifferentiable(f, g!, fg!, h!) = UnitializedTwiceDifferentiable(f, g!, fg!, h!)
TwiceDifferentiable(f, g!, h!) = UnitializedTwiceDifferentiable(f, g!,      nothing, h!)
TwiceDifferentiable(f, g!)     = UnitializedTwiceDifferentiable(f, g!,      nothing, nothing)
TwiceDifferentiable(f)         = UnitializedTwiceDifferentiable(f, nothing, nothing, nothing)
# The user friendly/short form TwiceDifferentiable constructor
function TwiceDifferentiable{T}(f, g!, fg!, h!, x_seed::Array{T})
    n_x = length(x_seed)
    g = similar(x_seed)
    H = Array{T}(n_x, n_x)

    _new_g!  = fix_order(g, x_seed, g!,   "g!")
    _new_fg! = fix_order(g, x_seed, fg!, "fg!")

    local _new_h!
    try
        _H = copy(H)
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
    g = similar(x_seed)
    _new_g! = fix_order(g, x_seed, g!,   "g!")

    function fg!(storage::Vector, x::Vector)
        _new_g!(storage, x)
        return f(x)
    end
    return TwiceDifferentiable(f, _new_g!, fg!, h!, x_seed)
end
