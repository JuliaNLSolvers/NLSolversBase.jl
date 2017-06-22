@compat abstract type AbstractObjective end

# Used for objectives and solvers where no gradient is available/exists
type NonDifferentiable{T} <: AbstractObjective
    f
    f_x::T
    last_x_f::Array{T}
    f_calls::Vector{Int}
end

NonDifferentiable(f, x_seed::AbstractArray) = NonDifferentiable(f, f(x_seed), copy(x_seed), [1])

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

# The user friendly/short form OnceDifferentiable constructor
function OnceDifferentiable(f, g!, fg!, x_seed::AbstractArray)
    g = similar(x_seed)
    f_val = fg!(g, x_seed)

    OnceDifferentiable(f, g!, fg!, f_val, g, copy(x_seed), copy(x_seed), [1], [1])
end
# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, g!, x_seed::AbstractArray)
    g = x_seed+one(eltype(x_seed))

    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    return OnceDifferentiable(f, g!, fg!, x_seed)
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

    f_val = fg!(g, x_seed)
    h!(H, x_seed)

    TwiceDifferentiable(f, g!, fg!, h!, f_val,
                                g, H, copy(x_seed),
                                copy(x_seed), copy(x_seed), [1], [1], [1])
end
# Automatically create the fg! helper function if only f, g! and h! is provided
function TwiceDifferentiable{T}(f,
                                 g!,
                                 h!,
                                 x_seed::Array{T})
    g = similar(x_seed)

    function fg!(storage::Vector, x::Vector)
        g!(storage, x)
        return f(x)
    end
    return TwiceDifferentiable(f, g!, fg!, h!, x_seed)
end
