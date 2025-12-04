# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T<:Real,TDF<:AbstractArray,TJVP<:Number,TH<:AbstractMatrix,THv<:AbstractArray,TX<:AbstractArray} <: AbstractObjective
    const f
    const df
    const fdf
    const jvp # Jacobian-vector product of objective
    const fjvp # objective and Jacobian-vector product of objective
    const dfh
    const fdfh
    const h
    const hv
    F::T
    const DF::TDF
    JVP::TJVP
    const H::TH
    const Hv::THv
    const x_f::TX
    const x_df::TX
    const x_jvp::TX
    const v_jvp::TX
    const x_h::TX
    const x_hv::TX
    const v_hv::TX
    f_calls::Int
    df_calls::Int
    jvp_calls::Int
    h_calls::Int
    hv_calls::Int
end

# compatibility with old constructor
function TwiceDifferentiable(f, g, fg, h, x::TX, F::T = real(zero(eltype(x))), G::TG = alloc_DF(x, F), H::TH = alloc_H(x, F); inplace::Bool = true) where {T<:Real, TG<:AbstractArray, TH<:AbstractMatrix, TX<:AbstractArray}
    g! = df!_from_df(g, F, inplace)
    fg! = fdf!_from_fdf(fg, F, inplace)
    h! = h!_from_h(h, F, inplace)
    dfh! = make_dfh(x, F, g!, h!)
    fdfh! = make_fdfh(x, F, fg!, h!)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hv = x_of_nans(x)
    v_hv = x_of_nans(x)

    JVP = alloc_JVP(x, F)

    TwiceDifferentiable{T,TG,typeof(JVP),TH,TG,TX}(f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, nothing,
                                        copy(F), copy(G), JVP, copy(H), copy(G),
                                        x_f, x_df, x_jvp, v_jvp, x_h, x_hv, v_hv,
                                        0, 0, 0, 0, 0)
end

function TwiceDifferentiable(f, g, h,
                             x::AbstractArray,
                             F::Real = real(zero(eltype(x))),
                             G::AbstractArray = alloc_DF(x, F),
                             H::AbstractMatrix = alloc_H(x, F); inplace = true)
    g! = df!_from_df(g, F, inplace)
    fg! = make_fdf(x, F, f, g!)
    h! = h!_from_h(h, F, inplace)
    dfh! = make_dfh(x, F, g!, h!)
    fdfh! = make_fdfh(x, F, fg!, h!)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hv = x_of_nans(x)
    v_hv = x_of_nans(x)

    return TwiceDifferentiable(f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, nothing, F, G, alloc_JVP(x, F), H, copy(G), x_f, x_df, x_jvp, v_jvp, x_h, x_hv, v_hv, 0, 0, 0, 0, 0)
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
    dfh! = make_dfh(x_seed, F, g!, h!)
    fdfh! = make_fdfh(x_seed, F, fg!, h!)

    hvp_prep = DI.prepare_hvp(f, autodiff, x_seed, (x_seed,))
    hv! = let f = f, hv_prep = hvp_prep, autodiff = autodiff
        function (_hv, _x, _v)
            DI.hvp!(f, (_hv,), hv_prep, autodiff, _x, (_v,))
            return nothing
        end
    end

    # TODO: Define dedicated AD-based functions for JVPs as well
    # Currently, this is disabled as it can lead to inconsistencies with gradient calculations with finite differencing (default)
    # We probably need a more fine-grained way for choosing AD backends, as JVP is a prime candidate for forward-mode AD

    x_f = x_of_nans(x_seed)
    x_df = x_of_nans(x_seed)
    x_jvp = x_of_nans(x_seed)
    v_jvp = x_of_nans(x_seed)
    x_h = x_of_nans(x_seed)
    x_hv = x_of_nans(x_seed)
    v_hv = x_of_nans(x_seed)

    return TwiceDifferentiable(f, g!, fg!, nothing, nothing, dfh!, fdfh!, h!, hv!, F, alloc_DF(x_seed, F), F, alloc_H(x_seed, F), alloc_DF(x_seed, F), x_f, x_df, x_jvp, v_jvp, x_h, x_hv, v_hv, 0, 0, 0, 0, 0)
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
    dfh! = make_dfh(x_seed, F, d.df, h!)
    fdfh! = make_fdfh(x_seed, F, d.fdf, h!)

    hvp_prep = DI.prepare_hvp(d.f, autodiff, x_seed, (x_seed,))
    hv! = let f = d.f, hv_prep = hvp_prep, autodiff = autodiff
        function (_hv, _x, _v)
            DI.hvp!(f, (_hv,), hv_prep, autodiff, _x, (_v,))
            return nothing
        end
    end

    # TODO: Define dedicated AD-based functions for JVPs as well
    # Currently, this is disabled as it can lead to inconsistencies with gradient calculations with finite differencing (default)
    # We probably need a more fine-grained way for choosing AD backends, as JVP is a prime candidate for forward-mode AD

    x_f = x_of_nans(x_seed)
    x_df = x_of_nans(x_seed)
    x_jvp = x_of_nans(x_seed)
    v_jvp = x_of_nans(x_seed)
    x_h = x_of_nans(x_seed)
    x_hv = x_of_nans(x_seed)
    v_hv = x_of_nans(x_seed)

    return TwiceDifferentiable(d.f, d.df, d.fdf, nothing, nothing, dfh!, fdfh!, h!, hv!, F, alloc_DF(x_seed, F), F, alloc_H(x_seed, F), alloc_DF(x_seed, F), x_f, x_df, x_jvp, v_jvp, x_h, x_hv, v_hv, 0, 0, 0, 0, 0)
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
    gh! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_g, _h, _x)
            DI.value_gradient_and_hessian!(f, _g, _h, hess_prep, autodiff, _x)
            return nothing
        end
    end
    fgh! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_g, _h, _x)
            y, _, _ = DI.value_gradient_and_hessian!(f, _g, _h, hess_prep, autodiff, _x)
            return y
        end
    end
    h! = let f = f, hess_prep = hess_prep, autodiff = autodiff
        function (_h, _x)
            DI.hessian!(f, _h, hess_prep, autodiff, _x)
            return nothing
        end
    end
    hvp_prep = DI.prepare_hvp(f, autodiff, x, (x,))
    hv! = let f = f, hv_prep = hvp_prep, autodiff = autodiff
        function (_hv, _x, _v)
            DI.hvp!(f, (_hv,), hv_prep, autodiff, _x, (_v,))
            return nothing
        end
    end

    # TODO: Define dedicated AD-based functions for JVPs as well
    # Currently, this is disabled as it can lead to inconsistencies with gradient calculations with finite differencing (default)
    # We probably need a more fine-grained way for choosing AD backends, as JVP is a prime candidate for forward-mode AD

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hv = x_of_nans(x)
    v_hv = x_of_nans(x)

    TwiceDifferentiable(f, g!, fg!, nothing, nothing, gh!, fgh!, h!, hv!, F, alloc_DF(x, F), F, alloc_H(x, F), alloc_DF(x, F), x_f, x_df, x_jvp, v_jvp, x_h, x_hv, v_hv, 0, 0, 0, 0, 0)
end
