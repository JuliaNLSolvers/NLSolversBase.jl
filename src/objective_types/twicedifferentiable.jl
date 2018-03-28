# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T,TDF,TH,TX,F,DF,FDF,H} <: AbstractObjective
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
iscomplex(obj::TwiceDifferentiable) = false
# compatibility with old constructor
function TwiceDifferentiable(f, g!, fg!, h!, x::TX, F::T = real(zero(eltype(x))), G::TG = similar(x), H::TH = alloc_H(x)) where {T, TG, TH, TX}
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)
    TwiceDifferentiable{T,TG, TH, TX,typeof(f),typeof(g!),typeof(fg!),typeof(h!)}(f, g!, fg!, h!,
                                        copy(F), similar(G), copy(H),
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
end

function TwiceDifferentiable(f, g!, h!, x::AbstractVector, F = real(zero(eltype(x))), G = similar(x), H = alloc_H(x))
    fg! = make_fdf(x, F, f, g!)
    return TwiceDifferentiable(f, g!, fg!, h!, x, F, G, H)
end