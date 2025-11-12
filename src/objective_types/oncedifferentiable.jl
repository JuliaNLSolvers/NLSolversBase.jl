# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiable{TF<:Union{AbstractArray,Real}, TDF<:AbstractArray, TX<:AbstractArray} <: AbstractObjective
    f # objective
    df # (partial) derivative of objective
    fdf # objective and (partial) derivative of objective
    jvp # Jacobian-vector product of objective
    fjvp # objective and Jacobian-vector product of objective
    F::TF # cache for f output
    DF::TDF # cache for df output
    JVP::TF # cache for jvp output
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    x_jvp::TX # x used to evaluate jvp (stored in JVP)
    v_jvp::TX # v used to evaluate jvp (stored in JVP) 
    f_calls::Int
    df_calls::Int
    jvp_calls::Int
end

### Only the objective
# Ambiguity
function OnceDifferentiable(f, x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F),
                   JVP::Real = real(zero(promote_type(eltype(x), typeof(F))));
                   inplace::Bool = true,
                   autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    OnceDifferentiable(f, x, F, DF, JVP, autodiff)
end
function OnceDifferentiable(f, x::AbstractArray,
                   F::AbstractArray, DF::AbstractArray = alloc_DF(x, F), JVP::AbstractArray = F;
                   inplace::Bool = true,
                   autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    f! = f!_from_f(f, F, inplace)
    OnceDifferentiable(f!, x, F, DF, JVP, autodiff)
end

function OnceDifferentiable(f, x_seed::AbstractArray,
                            F::Real,
                            DF::AbstractArray,
                            JVP::Real,
                            autodiff::AbstractADType)
    # When here, at the constructor with positional autodiff, it should already
    # be the case, that f is inplace.
    if f isa Union{InplaceObjective, NotInplaceObjective}

        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)
        jvpF = make_jvp(f, x_seed, F)
        fjvpF = make_fjvp(f, x_seed, F)

        return OnceDifferentiable(fF, dfF, fdfF, jvpF, fjvpF, x_seed, F, DF, JVP)
    else
        grad_prep = DI.prepare_gradient(f, autodiff, x_seed)
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
        pushforward_prep = DI.prepare_pushforward(f, backend, x_seed, (x_seed,))
        function jvp(_x, _v)
            ty = DI.pushforward(f, pushforward_prep, backend, _x, (_v,))
            return only(ty)
        end
        function fjvp(_x, _v)
            y, ty = DI.value_and_pushforward(f, pushforward_prep, backend, _x, (_v,))
            return y, only(ty)
        end
        return OnceDifferentiable(f, g!, fg!, jvp, fjvp, x_seed, F, DF, JVP)
    end
end

OnceDifferentiable(f, x::AbstractArray, F::AbstractArray, autodiff::AbstractADType) =
    OnceDifferentiable(f, x, F, alloc_DF(x, F), autodiff)
OnceDifferentiable(f, x::AbstractArray, F::AbstractArray, DF::AbstractArray, autodiff::AbstractADType) =
    OnceDifferentiable(f, x, F, DF, F, autodiff)
function OnceDifferentiable(f, x_seed::AbstractArray,
                            F::AbstractArray,
                            DF::AbstractArray,
                            JVP::AbstractArray,
                            autodiff::AbstractADType)
    if  f isa Union{InplaceObjective, NotInplaceObjective}
        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)
        jvpF = make_jvp(f, x_seed, F)
        fjvpF = make_fjvp(f, x_seed, F)
        return OnceDifferentiable(fF, dfF, fdfF, jvpF, fjvpF, x_seed, F, DF, JVP)
    else
        F2 = similar(F)
        jac_prep = DI.prepare_jacobian(f, F2, autodiff, x_seed)
        j! = let f = f, F2 = F2, jac_prep = jac_prep, autodiff = autodiff
            function (_j, _x)
                DI.jacobian!(f, F2, _j, jac_prep, autodiff, _x)
                return _j
            end
        end
        fj! = let f = f, jac_prep = jac_prep, autodiff = autodiff
            function (_y, _j, _x)
                y, _ = DI.value_and_jacobian!(f, _y, _j, jac_prep, autodiff, _x)
                return y
            end
        end
        pushforward_prep = DI.prepare_pushforward!(f, F2, backend, x_seed, (x_seed,))
        function jvp!(_jvp, _x, _v)
            DI.pushforward!(f, F2, _jvp, pushforward_prep, backend, _x, (_v,))
            return _jvp
        end
        function fjvp!(_y, _jvp, _x, _v)
            DI.value_and_pushforward!(f, _y, _jvp, pushforward_prep, backend, _x, (_v,))
            return _y, _jvp
        end
        return OnceDifferentiable(f, j!, fj!, jvp!, fjvp!, x_seed, F, DF, JVP)
    end
end

### Objective and derivative
function OnceDifferentiable(f, df,
                   x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F),
                   JVP::Real = F;
                   inplace::Bool = true)


    df! = df!_from_df(df, F, inplace)
    fdf! = make_fdf(x, F, f, df!)
    jvp = make_jvp(x, F, f, df!)
    fjvp = make_fjvp(x, F, f, fdf!)

    OnceDifferentiable(f, df!, fdf!, jvp, fjvp, x, F, DF, JVP)
end

function OnceDifferentiable(f, j,
                   x::AbstractArray,
                   F::AbstractArray,
                   J::AbstractArray = alloc_DF(x, F),
                   JVP::AbstractArray = F;
                   inplace::Bool = true)

    f! = f!_from_f(f, F, inplace)
    j! = df!_from_df(j, F, inplace)
    fj! = make_fdf(x, F, f!, j!)
    jvp! = make_jvp(x, F, f!, j!)
    fjvp! = make_fjvp(x, F, f!, j!)

    OnceDifferentiable(f!, j!, fj!, jvp, fjvp, x, F, J, JVP)
end


### Objective, derivative and combination
function OnceDifferentiable(f, df, fdf, jvp, fjvp,
    x::AbstractArray,
    F::Real = real(zero(eltype(x))),
    DF::AbstractArray = alloc_DF(x, F),
    JVP::Real = real(zero(promote_type(eltype(x), typeof(F))));
    inplace::Bool = true)

    # f, jvp and fjvp are never "inplace" since F is scalar
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)

    OnceDifferentiable{typeof(F),typeof(DF),typeof(x)}(f, df!, fdf!, jvp, fjvp,
    copy(F), copy(DF), copy(JVP),
    x_f, x_df, x_jvp, v_jvp,
    0, 0, 0)
end

function OnceDifferentiable(f, df, fdf, jvp, fjvp,
                            x::AbstractArray,
                            F::AbstractArray,
                            DF::AbstractArray = alloc_DF(x, F),
                            JVP::AbstractArray = F;
                            inplace::Bool = true)

    f = f!_from_f(f, F, inplace)
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)
    jvp! = jvp!_from_jvp(jvp, F, inplace)
    fjvp! = fjvp!_from_fjvp(fjvp, F, inplace)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)

    OnceDifferentiable(f, df!, fdf!, jvp!, fjvp!, similar(F), similar(DF), similar(JVP), x_f, x_df, x_jvp, v_jvp, 0, 0, 0)
end
