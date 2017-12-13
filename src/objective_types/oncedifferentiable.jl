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
function OnceDifferentiable(f, df, fdf, F::Union{Real, AbstractArray}, DF, x::AbstractArray)
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
        DF = complex_to_real(DF)
    end
           
    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    OnceDifferentiable{typeof(F),typeof(DF),typeof(x),Val{iscomplex}}(f, df, fdf,
                                                copy(F), # copy works for both scalars and arrays
                                                DF, # no need to copy as ∇F is always an abstract array
                                                x_f, x_df,
                                                [0,], [0,])
end
#=function OnceDifferentiable(f, df, fdf, F::AbstractArray, J, x::AbstractArray)
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
        J = complex_to_real(J)
    end
    x_f = similar(vec(x))
    x_df = similar(vec(x))
    Fv = similar(vec(F))
    xv = similar(vec(x))
    function fv!(fx::AbstractVector, x::AbstractVector)
        f(reshape(fx, size(F)...), reshape(x, size(x)...))
    end
    function jv!(jx::AbstractMatrix, x::AbstractVector)
        df(reshape(jx, size(J)...), reshape(x, size(x)...))
    end
    function fjv!(fx::AbstractVector, jx::AbstractMatrix, x::AbstractVector)
        fdf(reshape(fx, size(F)...), reshape(jx, size(J)...), reshape(x, size(x)...))
    end
        
    OnceDifferentiable{typeof(Fv),typeof(J),typeof(xv),Val{iscomplex}}(fv!, jv!, fjv!,
                                                Fv,     # copy works for both scalars and arrays
                                                J, # no need to copy as ∇F is always an abstract array
                                                x_f, x_df,
                                                [0,], [0,])
end
=#
#=function OnceDifferentiable(f, df, fdf, F::Real, G, x::AbstractArray)
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
        G = complex_to_real(G)
    end
    x_f = similar(vec(x))
    x_df = similar(vec(x))
    xv = similar(vec(x))
    function fv(x::AbstractVector)
        f(reshape(x, size(x)...))
    end
    function gv!(gx::AbstractVector, x::AbstractVector)
        df(reshape(gx, size(G)...), reshape(x, size(x)...))
    end
    function fgv!(gx::AbstractVector, x::AbstractVector)
        fdf(reshape(gx, size(G)...), reshape(x, size(x)...))
    end
        
    OnceDifferentiable{typeof(Fv),typeof(G),typeof(xv),Val{iscomplex}}(fv!, gv!, fgv!,
                                                F, G, x_f, x_df,
                                                [0,], [0,])
end
=#
# Automatically create the fg! helper function if only f and g! is provided
OnceDifferentiable(f, g!, fg!, fx::Real, x::AbstractArray) =
    OnceDifferentiable(f, g!, fg!, fx, similar(x), x)
OnceDifferentiable(f!, j!, fj!, fx::AbstractArray, x::AbstractArray) =
    OnceDifferentiable(f!, j!, fj!, fx, alloc_J(x), x)

function OnceDifferentiable(f, g!, fx::Real, x)
    function fg!(g, x)
        g!(g, x)
        return f(x)
    end
    return OnceDifferentiable(f, g!, fg!, fx, similar(x), x)
end
OnceDifferentiable(f!, j!, fx::AbstractArray, x::AbstractArray) =
    OnceDifferentiable(f!, j!, fx::AbstractArray, alloc_J(x), x::AbstractArray)

function OnceDifferentiable(f!, j!, fx::AbstractArray, J::AbstractArray, x::AbstractArray)
    function fj!(F, J, x)
        j!(J, x)
        return f!(F, x)
    end
    return OnceDifferentiable(f!, j!, fj!, fx, alloc_J(x), x)
end


function alloc_J(x::AbstractArray{T}) where T
    # Initialize an n-by-n Array{T, 2}
    Array{eltype(x), 2}(length(x), length(x))
end
