# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T<:Real,TDF<:AbstractArray,TH<:AbstractMatrix,TX<:AbstractArray} <: AbstractObjective
    f
    df
    fdf
    dfh
    fdfh
    h
    F::T
    DF::TDF
    H::TH
    x_f::TX
    x_df::TX
    x_h::TX
    f_calls::Int
    df_calls::Int
    h_calls::Int
end
# compatibility with old constructor
function TwiceDifferentiable(f, g, fg, h, x::TX, F::T = real(zero(eltype(x))), G::TG = alloc_DF(x, F), H::TH = alloc_H(x, F); inplace::Bool = true) where {T<:Real, TG<:AbstractArray, TH<:AbstractMatrix, TX<:AbstractArray}
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)

    g! = df!_from_df(g, F, inplace)
    fg! = fdf!_from_fdf(fg, F, inplace)
    h! = h!_from_h(h, F, inplace)

    TwiceDifferentiable{T,TG,TH,TX}(f, g!, fg!, nothing, nothing, h!,
                                        copy(F), copy(G), copy(H),
                                        x_f, x_df, x_h,
                                        0, 0, 0)
end

function TwiceDifferentiable(f, g, h,
                             x::AbstractArray,
                             F::Real = real(zero(eltype(x))),
                             G::AbstractArray = alloc_DF(x, F),
                             H::AbstractMatrix = alloc_H(x, F); inplace = true)
    g! = df!_from_df(g, F, inplace)
    h! = h!_from_h(h, F, inplace)

    fg! = make_fdf(x, F, f, g!)
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)

    return TwiceDifferentiable(f, g!, fg!, nothing, nothing, h!, F, G, H, x_f, x_df, x_h, 0, 0, 0)
end



function TwiceDifferentiable(f, g,
                             x_seed::AbstractArray,
                             F::Real = real(zero(eltype(x_seed)));
                             inplace::Bool = true,
                             autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    g! = df!_from_df(g, F, inplace)
    fg! = make_fdf(x_seed, F, f, g!)

    hess_prep = DI.prepare_hessian(f, autodiff, x_seed)
    h! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return _h
        end
    end
    TwiceDifferentiable(f, g!, fg!, h!, x_seed, F)
end

function TwiceDifferentiable(d::NonDifferentiable,
                             x_seed::AbstractArray = d.x_f,
                             F::Real = real(zero(eltype(x_seed)));
                             autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    TwiceDifferentiable(d.f, x_seed, F; autodiff)
end

function TwiceDifferentiable(d::OnceDifferentiable,
                             x_seed::AbstractArray = d.x_f,
                             F::Real = real(zero(eltype(x_seed)));
                             autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    hess_prep = DI.prepare_hessian(d.f, autodiff, x_seed)
    h! = let f = d.f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return _h
        end
    end
    return TwiceDifferentiable(d.f, d.df, d.fdf, h!, x_seed, F, gradient(d))
end

function TwiceDifferentiable(f, x::AbstractArray, F::Real = real(zero(eltype(x)));
                             inplace::Bool = true,
                             autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    grad_prep = DI.prepare_gradient(f, autodiff, x)
    g! = let f = f, grad_prep = grad_prep, autodiff = autodiff
        function (_g, _x)
            DI.gradient!(f, _g, grad_prep, autodiff, _x)
            return nothing
        end
    end
    fg! = let f = f, grad_prep = grad_prep, autodiff = autodiff
        function (_g, _x)
            y, _ = DI.value_and_gradient!(f, _g, grad_prep, autodiff, _x)
            return y
        end
    end
    hess_prep = DI.prepare_hessian(f, autodiff, x)
    h! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return _h
        end
    end
    TwiceDifferentiable(f, g!, fg!, h!, x, F)
end

function hv_product!(obj::TwiceDifferentiable, x, v)
    H = hessian!(obj, x)
    return H*v
end
