abstract type OnceDifferentiable{TF, TDF, TX, Tcplx<:Union{Val{true},Val{false}}} <: AbstractObjective end
abstract type OnceDifferentiableDynamic{TF, TDF, TX, Tcplx} <: OnceDifferentiable{TF, TDF, TX, Tcplx} end
# abstract type OnceDifferentiableStatic{TF, TDF, TX, Tcplx} <: OnceDifferentiable{TF, TDF, TX, Tcplx} end


# Func typed is the current default.
# Func untyped is preferred if you're dynamically dispatching on functions, eg
# whether Forward or finite diff derivatives are used is unknown at compile time.
# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiableDynamicFuncTyped{TF, TDF, TX,F,DF,FDF} <: OnceDifferentiableDynamic{TF, TDF, TX, Tcplx}
    f::F # objective
    df::DF # (partial) derivative of objective
    fdf::FDF # objective and (partial) derivative of objective
    F::TF # cache for f output
    DF::TDF # cache for df output
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    f_calls::Vector{Int}
    df_calls::Vector{Int}
end
mutable struct OnceDifferentiableDynamicFuncUntyped{TF, TDF, TX} <: OnceDifferentiableDynamic{TF, TDF, TX}
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

# Compatibility with old constructor that doesn't have the complex field
function OnceDifferentiable(f, df, fdf, x::AbstractArray{T}, F::Union{Real, AbstractArray}, ::Val{S}) where {S,T}#;
    #                            inplace = true) #inplace currently unused.
        
    iscomplex = T <: Complex
    DF = alloc_DF(x, F)

    x_f, x_df = x_of_nans(x), x_of_nans(x)
    if S
        OnceDifferentiableDynamicFuncTyped{typeof(F),typeof(DF),typeof(x),Val{false},typeof(f),typeof(df),typeof(fdf)}(f, df, fdf, copy(F), DF, x_f, x_df, [0,], [0,])
    else
        OnceDifferentiableDynamicFuncUntyped{typeof(F),typeof(DF),typeof(x),Val{false}}(f, df, fdf,
                                                copy(F), copy(DF), x_f, x_df, [0,], [0,])
    end
end
function OnceDifferentiable(f, df, fdf, x::AbstractArray{T}, F::Union{Real, AbstractArray}, ::Val{S}) where {S,T <: Complex}#;
    #                            inplace = true) #inplace currently unused.
        
    iscomplex = T <: Complex
    DF = complex_to_real(alloc_DF(x, F))
    x = complex_to_real(x)

    x_f, x_df = x_of_nans(x), x_of_nans(x)
    if S
        OnceDifferentiableDynamicFuncTyped{typeof(F),typeof(DF),typeof(x),Val{true},typeof(f),typeof(df),typeof(fdf)}(f, df, fdf, copy(F), DF, x_f, x_df, [0,], [0,])
    else
        OnceDifferentiableDynamicFuncUntyped{typeof(F),typeof(DF),typeof(x),Val{true}}(f, df, fdf,
                                                copy(F), copy(DF), x_f, x_df, [0,], [0,])
    end
end
function OnceDifferentiable(f, df, fdf,
                            x::AbstractArray,
                            F::Real = real(zero(eltype(x))),
                            DF = alloc_DF(x, F), ::Val{S} = Val{true}() ) where S
                            #; inplace = true) #inplace currently unused.
    x_f, x_df = x_of_nans(x), x_of_nans(x)
    if S
        OnceDifferentiableDynamicFuncTyped{typeof(F),typeof(DF),typeof(x),typeof(f),typeof(df),typeof(fdf)}(f, df, fdf,
                                                    copy(F), copy(DF),
                                                    x_f, x_df,
                                                    [0,], [0,])
    else
        OnceDifferentiableDynamicFuncUntyped{typeof(F),typeof(DF),typeof(x)}(f, df, fdf,
                                                copy(F), copy(DF),
                                                x_f, x_df,
                                                [0,], [0,])
    end
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
