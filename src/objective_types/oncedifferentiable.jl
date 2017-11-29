# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiable{TF, TDF, TX, Tcplx<:Union{Val{true},Val{false}}} <: AbstractObjective
    f # objective
    df # (partial) derivative of objective
    fdf # objective and (partial) derivative of objective
    F::TF # cache for f output
    DF::TDF # cache for df output
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    f_calls::Vector{Int}
    df_calls::Vector{Int}
end
iscomplex(obj::OnceDifferentiable{T,Tgrad,A,Val{true}}) where {T,Tgrad,A} = true
iscomplex(obj::OnceDifferentiable{T,Tgrad,A,Val{false}}) where {T,Tgrad,A} = false

# Compatibility with old constructor that doesn't have the complex field
function OnceDifferentiable(f, df, fdf, F, DF, x)
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
        DF = complex_to_real(DF)
    end
    x_f = similar(vec(x))
    x_df = similar(vec(x))
    Fv = typeof(F) <: Real ? F : similar(vec(F))
    xv = similar(vec(x))
    if !(typeof(F)<:AbstractVector)
        function f!(fx::AbstractVector, x::AbstractVector)
            f!(reshape(fx, size(F)...), reshape(x, size(x)...))
        end
    end
    function jv!(jx::AbstractMatrix, x::AbstractVector)
        j!(jx, reshape(x, size(x)...))
    end
    function fjv!(fx::AbstractVector, gx::AbstractMatrix, x::AbstractVector)
        fj!(reshape(fx, size(F)...), gx, reshape(x, size(x)...))
    end
    
    OnceDifferentiable{typeof(Fv),typeof(DF),typeof(xv),Val{iscomplex}}(f, df, fdf,
                                                Fv,     # copy works for both scalars and arrays
                                                DF, # no need to copy as ∇F is always an abstract array
                                                x_f, x_df,
                                                [0,], [0,])
end
OnceDifferentiable(f, g!, fg!, F::Real, x) = OnceDifferentiable(f, g!, fg!, F, similar(x), x) 
OnceDifferentiable(f, g!, fg!, F::AbstractVector, x) = OnceDifferentiable(f, g!, fg!, F, alloc_J(x), x) 

# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, g!, fx::Real, x::AbstractArray)
    function fg!(g, x)
        g!(g, x)
        return f(x)
    end
    return OnceDifferentiable(f, g!, fg!, fx,  x)
end

# Automatically create the fj! helper function if only f! and j! is provided
OnceDifferentiable(f!, j!, fx::AbstractArray, x::AbstractArray) = OnceDifferentiable(f!, j!, fx, alloc_J(x), x)
function OnceDifferentiable(f!, j!, fx::AbstractArray, J, x::AbstractArray)
    function fj!(F, J, x)
        j!(J, x)
        return f!(F, x)
    end
    return OnceDifferentiable(f!, j!, fj!, fx, J, x)
end
function alloc_J(x::AbstractArray{T}) where T
    # Initialize an n-by-n Array{T, 2}
    Array{eltype(x), 2}(length(x), length(x))
end
# Generic function to allocate the full OnceDifferentiable given all functions,
# an initial point, and possibly a custom Jacobian cache variable (that j!, fj!)
# expect as first positional argument.
#=
function OnceDifferentiable(f!, j!, fj!, F::TF, J, x0::TX) where {TF<:AbstractArray, TX<:AbstractArray}
    xv = similar(vec(x0))
    
    function fv!(fx::AbstractVector, x::AbstractVector)
        f!(reshape(fx, size(F)...), reshape(x, size(x0)...))
    end
    function jv!(jx::AbstractMatrix, x::AbstractVector)
        j!(jx, reshape(x, size(x0)...))
    end
    function fjv!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        fj!(reshape(fx, size(F)...), gx, reshape(x, size(x0)...))
    end
    OnceDifferentiable(fv!, jv!, fjv!, similar(xv), J, similar(xv)) 
end
=#


