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
function OnceDifferentiable(f, df, fdf,
                            x::AbstractArray,
                            F::Real = real(zero(eltype(x))),
                            DF = alloc_DF(x, F);
                            inplace = true)
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
        DF = complex_to_real(DF)
    end

    x_f, x_df = x_of_nans(x), x_of_nans(x)
    OnceDifferentiable{typeof(F),typeof(DF),typeof(x),Val{iscomplex}}(f, df, fdf,
                                                copy(F), copy(DF),
                                                x_f, x_df,
                                                [0,], [0,])
end
function OnceDifferentiable(f, df, fdf,
                            x::AbstractArray,
                            F::AbstractArray,
                            DF = alloc_DF(x, F);
                            inplace = true)
    iscomplex = eltype(x) <: Complex
    if iscomplex
        x = complex_to_real(x)
        DF = complex_to_real(DF)
    end

    x_f, x_df = x_of_nans(x), x_of_nans(x)
    OnceDifferentiable{typeof(F),typeof(DF),typeof(x),Val{iscomplex}}(f, df, fdf,
                                                copy(F), copy(DF),
                                                x_f, x_df,
                                                [0,], [0,])
end

# Automatically create the fg! helper function if only f and g! is provided
function OnceDifferentiable(f, df, x::AbstractArray)
    F = real(zero(eltype(x)))
    fdf = make_fdf(x, F, f, df)
    return OnceDifferentiable(f, df, fdf, x, F)
end
OnceDifferentiable(f, df, x::AbstractArray, F::AbstractArray, DF::AbstractArray = alloc_DF(x, F)) =
    OnceDifferentiable(f, df, make_fdf(x, F, f, df), x, F, DF)
OnceDifferentiable(f, df, x::AbstractArray, F::Real, DF::AbstractArray =  alloc_DF(x, F)) =
    OnceDifferentiable(f, df, make_fdf(x, F, f, df), x, F, DF)
