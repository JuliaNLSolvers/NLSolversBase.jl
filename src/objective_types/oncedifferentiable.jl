# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiable{TF<:Union{AbstractArray,Real}, TDF<:AbstractArray, TX<:AbstractArray} <: AbstractObjective
    f # objective
    df # (partial) derivative of objective
    fdf # objective and (partial) derivative of objective
    F::TF # cache for f output
    DF::TDF # cache for df output
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    f_calls::Int
    df_calls::Int
end

### Only the objective
# Ambiguity
function OnceDifferentiable(f, x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F);
                   inplace::Bool = true,
                   autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    OnceDifferentiable(f, x, F, DF, autodiff)
end
function OnceDifferentiable(f, x::AbstractArray,
                   F::AbstractArray, DF::AbstractArray = alloc_DF(x, F);
                   inplace::Bool = true,
                   autodiff::AbstractADType = AutoFiniteDiff(; fdtype = Val(:central)))
    f! = f!_from_f(f, F, inplace)
    OnceDifferentiable(f!, x, F, DF, autodiff)
end


function OnceDifferentiable(f, x_seed::AbstractArray,
                            F::Real,
                            DF::AbstractArray,
                            autodiff::AbstractADType)
    # When here, at the constructor with positional autodiff, it should already
    # be the case, that f is inplace.
    if f isa Union{InplaceObjective, NotInplaceObjective}

        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)

        return OnceDifferentiable(fF, dfF, fdfF, x_seed, F, DF)
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
        return OnceDifferentiable(f, g!, fg!, x_seed, F, DF)
    end
end

OnceDifferentiable(f, x::AbstractArray, F::AbstractArray, autodiff::AbstractADType) =
    OnceDifferentiable(f, x, F, alloc_DF(x, F), autodiff)
function OnceDifferentiable(f, x_seed::AbstractArray,
                            F::AbstractArray,
                            DF::AbstractArray,
                            autodiff::AbstractADType)
    if  f isa Union{InplaceObjective, NotInplaceObjective}
        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)
        return OnceDifferentiable(fF, dfF, fdfF, x_seed, F, DF)
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
        return OnceDifferentiable(f, j!, fj!, x_seed, F, DF)
    end
end

### Objective and derivative
function OnceDifferentiable(f, df,
                   x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F);
                   inplace::Bool = true)


    df! = df!_from_df(df, F, inplace)

    fdf! = make_fdf(x, F, f, df!)

    OnceDifferentiable(f, df!, fdf!, x, F, DF)
end

function OnceDifferentiable(f, j,
                   x::AbstractArray,
                   F::AbstractArray,
                   J::AbstractArray = alloc_DF(x, F);
                   inplace::Bool = true)

    f! = f!_from_f(f, F, inplace)
    j! = df!_from_df(j, F, inplace)
    fj! = make_fdf(x, F, f!, j!)

    OnceDifferentiable(f!, j!, fj!, x, F, J)
end


### Objective, derivative and combination
function OnceDifferentiable(f, df, fdf,
    x::AbstractArray,
    F::Real = real(zero(eltype(x))),
    DF::AbstractArray = alloc_DF(x, F);
    inplace::Bool = true)

    # f is never "inplace" since F is scalar
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f, x_df = x_of_nans(x), x_of_nans(x)

    OnceDifferentiable{typeof(F),typeof(DF),typeof(x)}(f, df!, fdf!,
    copy(F), copy(DF),
    x_f, x_df,
    0, 0)
end

function OnceDifferentiable(f, df, fdf,
                            x::AbstractArray,
                            F::AbstractArray,
                            DF::AbstractArray = alloc_DF(x, F);
                            inplace::Bool = true)

    f = f!_from_f(f, F, inplace)
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f, x_df = x_of_nans(x), x_of_nans(x)

    OnceDifferentiable(f, df!, fdf!, copy(F), copy(DF), x_f, x_df, 0, 0)
end
