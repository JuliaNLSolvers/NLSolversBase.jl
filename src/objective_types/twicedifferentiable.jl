abstract type TwiceDifferentiable{T,TDF,TH,TX} <: AbstractObjective end
abstract type TwiceDifferentiableDynamic{T,TDF,TH,TX} <: TwiceDifferentiable{T,TDF,TH,TX} end
# abstract type TwiceDifferentiableStatic{T,TDF,TH,TX} <: TwiceDifferentiable{T,TDF,TH,TX} end


# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiableDynamicFuncTyped{T,TDF,TH,TX,F,DF,FDF,H} <: TwiceDifferentiableDynamic{T,TDF,TH,TX}
    f::F
    df::DF
    fdf::FDF
    h::H
    F::T
    DF::TDF
    H::TH
    x_f::TX
    x_df::TX
    x_h::TX
    f_calls::Vector{Int}
    df_calls::Vector{Int}
    h_calls::Vector{Int}
end
# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiableDynamicFuncUntyped{T,TDF,TH,TX} <: TwiceDifferentiableDynamic{T,TDF,TH,TX}
    f
    df
    fdf
    h
    F::T
    DF::TDF
    H::TH
    x_f::TX
    x_df::TX
    x_h::TX
    f_calls::Vector{Int}
    df_calls::Vector{Int}
    h_calls::Vector{Int}
end
iscomplex(obj::TwiceDifferentiable) = false
# compatibility with old constructor
function TwiceDifferentiable(f, g!, fg!, h!, x::TX, F::T, ::Val{S}) where {T, TG, TH, TX, S}
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)
    H = alloc_H(x)
    if S
        TwiceDifferentiableDynamicFuncTyped{T,TX,typeof(H),TX,typeof(f),typeof(g!),typeof(fg!),typeof(h!)}(
                                        f, g!, fg!, h!,
                                        copy(F), similar(x), H,
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
    else
        TwiceDifferentiableDynamicFuncUntyped{T,TX,typeof(H),TX,typeof(f),typeof(g!),typeof(fg!),typeof(h!)}(
                                        f, g!, fg!, h!,
                                        copy(F), similar(x), H,
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
    end
end
function TwiceDifferentiable(f, g!, fg!, h!, x::TX, F::T = real(zero(eltype(x))), G::TG = similar(x), H::TH = alloc_H(x), ::Val{S} = Val{true}()) where {T, TG, TH, TX, S}
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)
    if S
        TwiceDifferentiableDynamicFuncTyped{T,TG, TH, TX,typeof(f),typeof(g!),typeof(fg!),typeof(h!)}(
                                        f, g!, fg!, h!,
                                        copy(F), similar(G), copy(H),
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
    else
        TwiceDifferentiableDynamicFuncUntyped{T,TG, TH, TX,typeof(f),typeof(g!),typeof(fg!),typeof(h!)}(
                                        f, g!, fg!, h!,
                                        copy(F), similar(G), copy(H),
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
    end
end



function TwiceDifferentiable(f, g!, h!, x::AbstractVector, F = real(zero(eltype(x))), G = similar(x), H = alloc_H(x))
    fg! = make_fdf(x, F, f, g!)
    return TwiceDifferentiable(f, g!, fg!, h!, x, F, G, H)
end