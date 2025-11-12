# To pass incomplete specifications, we provide the "only_"
# family of functions. It allows the users to simply provide
# fg! for optimization or fj! for solving systems of equations
# and have NLSolversBase wrap them up properly so "F" can be
# calculated on its own and the save for partial derivatives.
# Note, that for TwiceDifferentiable we cannot provide con-
# structors if h === nothing, as that requires automatic dif-
# fferentiation of some sort.
@kwdef struct InplaceObjective{FDF, FGH, Hv, FGHv, JVP, FJVP}
    fdf::FDF = nothing
    fgh::FGH = nothing
    hv::Hv = nothing
    fghv::FGHv = nothing
    jvp::JVP = nothing
    fjvp::FJVP = nothing
end

@kwdef struct NotInplaceObjective{DF, FDF, FGH, JVP, FJVP}
    df::DF = nothing
    fdf::FDF = nothing
    fgh::FGH = nothing
    jvp::JVP = nothing
    fjvp::FJVP = nothing
end

# Mutating version
only_fg!(fg)     = InplaceObjective(; fdf=fg)
only_fgh!(fgh)   = InplaceObjective(; fgh=fgh)

only_fg_and_gv!(fg, jvp) = InplaceObjective(; fdf=fg, jvp)
only_fg_and_fgv!(fg, fjvp) = InplaceObjective(; fdf=fg, fjvp)

only_fg_and_hv!(fg, hv) = InplaceObjective(; fdf=fg, hv=hv)
only_fghv!(fghv)        = InplaceObjective(; fghv=fghv)

# Non-mutating version
only_fg(fg) = NotInplaceObjective(; fdf=fg)
only_fgh(fgh) = NotInplaceObjective(; fgh) 

only_fg_and_gv(fg, jvp) = NotInplaceObjective(; fdf = fg, jvp)
only_fg_and_fgv(fg, fjvp) = NotInplaceObjective(; fdf = fg, fjvp)

only_g_and_fg(g, fg)    = NotInplaceObjective(; df=g, fdf=fg)

# Aliases with `j` (Jacobian)
# Do not apply to functions that involve Hessians!
# Maybe we should clearly separate functions that take gradients (=> scalar-valued functions) and Jacobians (=> array-valued functions)?
const only_fj! = only_fg!
const only_fj_and_jv! = only_fg_and_gv!
const only_fj_and_fjv! = only_fg_and_fgv!

const only_fj = only_fg
const only_fj_and_jv = only_fg_and_gv
const only_fj_and_fjv = only_fg_and_fgv
const only_j_and_fj = only_g_and_fg

# Mutating version
function make_f(t::InplaceObjective, _, F::Real)
    (; fdf, fgh, fghv) = t
    if fdf !== nothing
        return let fdf = fdf, F = F
            x -> fdf(F, nothing, x)
        end
    elseif fgh !== nothing
        return let fgh = fgh, F = F
            x -> fgh(F, nothing, nothing, x)
        end
    elseif fghv !== nothing
        return let fghv = fghv, F = F
            x -> fghv(F, nothing, nothing, x, nothing)
        end
    else
        return (x -> throw(ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.")))
    end
end
function make_f(t::InplaceObjective, _, ::AbstractArray)
    (; fdf) = t
    if fdf !== nothing
        return let fdf = fdf
            (F,x) -> fdf(F, nothing, x)
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` and `fghv` are not appropriate for functions with `::AbstractArray` output
        return ((F, x) -> throw(ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.")))
    end
end

function make_df(t::InplaceObjective, _, ::Real)
    (; fdf, fgh, fghv) = t
    if fdf !== nothing
        return let fdf = fdf
            (DF, x) -> fdf(nothing, DF, x)
        end
    elseif fgh !== nothing
        return let fgh = fgh
            (DF, x) -> fgh(nothing, DF, nothing, x)
        end
    elseif fghv !== nothing
        return let fghv = fghv
            (DF, x) -> fghv(nothing, DF, nothing, x, nothing)
        end
    else
        return ((DF, x) -> throw(ArgumentError("Cannot evaluate the gradient of the objective function: No suitable Julia function available.")))
    end
end
function make_df(t::InplaceObjective, _, ::AbstractArray)
    (; fdf) = t
    if fdf !== nothing
        return let fdf = fdf
            (DF, x) -> fdf(nothing, DF, x)
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` and `fghv` are not appropriate for functions with `::AbstractArray` output
        return ((DF, x) -> throw(ArgumentError("Cannot evaluate the Jacobian of the objective function: No suitable Julia function available.")))
    end
end

function make_fdf(t::InplaceObjective, _, F::Real)
    (; fdf, fgh, fghv) = t
    if fdf !== nothing
        return let fdf = fdf, F = F
            (G, x) -> fdf(F, G, x)
        end
    elseif fgh !== nothing
        return let fgh = fgh, F = F
            (G, x) -> fgh(F, G, nothing, x)
        end
    elseif fghv !== nothing
        return let fghv = fghv, F = F
            (G, x) -> fghv(F, G, nothing, x, nothing)
        end
    else
        return ((DF, x) -> throw(ArgumentError("Cannot evaluate the objective function and its gradient: No suitable Julia function available.")))
    end
end
function make_fdf(t::InplaceObjective, _, ::AbstractArray)
    (; fdf) = t
    if fdf !== nothing
        return fdf
    else
        # Note: Functions involving the Hessian matrix such as `fgh` and `fghv` are not appropriate for functions with `::AbstractArray` output
        return ((F, G, x) -> throw(ArgumentError("Cannot evaluate the objective function and its Jacobian: No suitable Julia function available.")))
    end
end

# Hessian matrix calculations require that the objective function returns a scalar
function make_dfh(t::InplaceObjective, _, ::Real)
    (; fgh) = t
    if fgh !== nothing
        return let fgh = fgh
            (DF, H, x) -> fgh(nothing, DF, H, x)
        end
    else
        return ((DF, H, x) -> throw(ArgumentError("Cannot evaluate the gradient and Hessian of the objective function: No suitable Julia function available.")))
    end
end

function make_fdfh(t::InplaceObjective, _, F::Real)
    (; fgh) = t
    if fgh !== nothing
        return let fgh = fgh, F = F
            (G, H, x) -> fgh(F, G, H, x)
        end
    else
        return ((G, H, x) -> throw(ArgumentError("Cannot evaluate the objective function, its gradient and its Hessian: No suitable Julia function available.")))
    end
end
function make_h(t::InplaceObjective, _, ::Real)
    (; fgh) = t
    if fgh !== nothing
        return let fgh = fgh
            (H, x) -> fgh(nothing, nothing, H, x)
        end
    else
        return ((H, x) -> throw(ArgumentError("Cannot evaluate the Hessian of the objective function: No suitable Julia function available.")))
    end
end
function make_hv(t::InplaceObjective, _, ::Real)
    (; hv, fghv) = t
    if hv !== nothing
        return hv
    elseif fghv !== nothing
        return let fghv = fghv
            (Hv, x, v) -> fghv(nothing, nothing, Hv, x, v)
        end
    else
        return ((Hv, x, v) -> throw(ArgumentError("Cannot evaluate the Hessian-vector product of the objective function: No suitable Julia function available.")))
    end
end

function make_jvp(t::InplaceObjective, x::AbstractArray, F::Real)
    (; jvp, fjvp, fdf, fgh, fghv) = t
    if jvp !== nothing
        return jvp
    elseif fjvp !== nothing
        return let fjvp = fjvp, F = F
            (x, v) -> last ∘ fjvp(nothing, F, x, v)
        end
    elseif fdf !== nothing
        return let fdf = fdf, df = alloc_DF(x, F)
            (x, v) -> begin
                fdf(nothing, df, x)
                return LinearAlgebra.dot(df, v)
            end
        end
    elseif fgh !== nothing
        return let fgh = fgh, df = alloc_DF(x, F)
            (x, v) -> begin
                fgh(nothing, df, nothing, x)
                return LinearAlgebra.dot(df, v)
            end
        end
    elseif fghv !== nothing
        return let fghv = fghv, df = alloc_DF(x, F)
            (x, v) -> begin
                fghv(nothing, df, nothing, x, v)
                return LinearAlgebra.dot(df, v)
            end
        end
    else
        return ((x, v) -> throw(ArgumentError("Cannot evaluate the Jacobian-vector product of the objective function: No suitable Julia function available.")))
    end
end
function make_jvp(t::InplaceObjective, x::AbstractArray, F::AbstractArray)
    (; jvp, fjvp, fdf) = t
    if jvp !== nothing
        return jvp
    elseif fjvp !== nothing
        return let fjvp = fjvp
            (JVP, x, v) -> last ∘ fjvp(nothing, JVP, x, v)
        end
    elseif fdf !== nothing
        return let fdf = fdf, jx = alloc_DF(x, F)
            (JVP, x, v) -> begin
                fdf(nothing, jx, x)
                LinearAlgebra.mul!(JVP, jx, v)
                return JVP
            end
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` and `fghv` are not appropriate for functions with `::AbstractArray` output
        return ((JVP, x, v) -> throw(ArgumentError("Cannot evaluate the Jacobian-vector product of the objective function: No suitable Julia function available.")))
    end
end

function make_fjvp(t::InplaceObjective, x::AbstractArray, F::Real)
    (; fjvp, fdf, fgh, fghv) = t
    if fjvp !== nothing
        return fjvp
    elseif fdf !== nothing
        return let fdf = fdf, F = F, df = alloc_DF(x, F)
            (x, v) -> begin
                fx = fdf(F, df, x)
                return fx, LinearAlgebra.dot(df, v)
            end
        end
    elseif fgh !== nothing
        return let fgh = fgh, F = F, df = alloc_DF(x, F)
            (x, v) -> begin
                fx = fgh(F, df, nothing, x)
                return fx, LinearAlgebra.dot(df, v)
            end
        end
    elseif fghv !== nothing
        return let fghv = fghv, F = F, df = alloc_DF(x, F)
            (x, v) -> begin
                fx = fghv(F, df, nothing, x, v)
                return fx, LinearAlgebra.dot(df, v)
            end
        end
    else
        return ((x, v) -> throw(ArgumentError("Cannot evaluate the objective function and its Jacobian-vector product: No suitable Julia function available.")))
    end
end
function make_fjvp(t::InplaceObjective, x::AbstractArray, F::AbstractArray)
    (; fjvp, fdf) = t
    if fjvp !== nothing
        return fjvp
    elseif fdf !== nothing
        return let fdf = fdf, jx = alloc_DF(x, F)
            (F, JVP, x, v) -> begin
                fdf(F, jx, x)
                LinearAlgebra.mul!(JVP, jx, v)
                return F, JVP
            end
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` and `fghv` are not appropriate for functions with `::AbstractArray` output
        return ((F, JVP, x, v) -> throw(ArgumentError("Cannot evaluate the objective function and its Jacobian-vector product: No suitable Julia function available.")))
    end
end

# Non-mutating version
# The contract with the user is that fdf returns (F, DF)
# and then we simply need to pick out the appropriate element
# of whatever fdf returns.
function make_f(t::NotInplaceObjective, _, _::Real)
    (; fdf, fgh) = t
    if fdf !== nothing
        return first ∘ fdf
    elseif fgh !== nothing
        return first ∘ fgh
    else
        return (x -> throw(ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.")))
    end
end
function make_f(t::NotInplaceObjective, _, ::AbstractArray)
    (; fdf) = t
    if fdf !== nothing
        return let fdf = fdf
            (F, x) -> copyto!(F, first(fdf(x)))
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` are not appropriate for functions with `::AbstractArray` output
        return ((F, x) -> throw(ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.")))
    end
end

function make_df(t::NotInplaceObjective, _, ::Real)
    (; df, fdf, fgh) = t
    if df !== nothing
        return let df = df
            (DF, x) -> copyto!(DF, df(x))
        end
    elseif fdf !== nothing
        return let fdf = fdf
            (DF, x) -> copyto!(DF, fdf(x)[2])
        end
    elseif fgh !== nothing
        return let fgh = fgh
            (DF, x) -> copyto!(DF, fgh(x)[2])
        end
    else
        return ((DF, x) -> throw(ArgumentError("Cannot evaluate the gradient of the objective function: No suitable Julia function available.")))
    end
end
function make_df(t::NotInplaceObjective, _, ::AbstractArray)
    (; df, fdf) = t
    if df !== nothing
        return let df = df
            (DF, x) -> copyto!(DF, df(x))
        end
    elseif fdf !== nothing
        return let fdf = fdf
            (DF, x) -> copyto!(DF, fdf(x)[2])
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` are not appropriate for functions with `::AbstractArray` output
        return ((DF, x) -> throw(ArgumentError("Cannot evaluate the Jacobian of the objective function: No suitable Julia function available.")))
    end
end

function make_fdf(t::NotInplaceObjective, _, F::Real)
    (; fdf, fgh) = t
    if fdf !== nothing
        return let fdf = fdf
            (DF, x) -> begin
                f, g = fdf(x)
                copyto!(DF, g)
                return f
            end
        end
    elseif fgh !== nothing
        return let fgh = fgh
            (DF, x) -> begin
                f, g, _ = fgh(x)
                copyto!(DF, g)
                return f
            end
        end
    else
        return ((DF, x) -> throw(ArgumentError("Cannot evaluate the objective function and its gradient: No suitable Julia function available.")))
    end
end
function make_fdf(t::NotInplaceObjective, _, ::AbstractArray)
    (; fdf) = t
    if fdf !== nothing
        return let fdf = fdf
            (F, DF, x) -> begin
                f, j = fdf(x)
                copyto!(DF, j)
                copyto!(F, f)
            end
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` are not appropriate for functions with `::AbstractArray` output
        return ((F, DF, x) -> throw(ArgumentError("Cannot evaluate the objective function and its Jacobian: No suitable Julia function available.")))
    end
end

function make_jvp(t::NotInplaceObjective, ::AbstractArray, ::Real)
    (; jvp, fjvp, df, fdf, fgh) = t
    if jvp !== nothing
        return jvp
    elseif fjvp !== nothing
        return last ∘ fjvp
    elseif df !== nothing
        return let df = df
            (x, v) -> LinearAlgebra.dot(df(x), v)
        end
    elseif fdf !== nothing
        return let fdf = fdf
            (x, v) -> LinearAlgebra.dot(last(fdf(x)), v)
        end
    elseif fgh !== nothing
        return let fgh = fgh
            (x, v) -> LinearAlgebra.dot(fgh(x)[2], v)
        end
    else
        return ((x, v) -> throw(ArgumentError("Cannot evaluate the Jacobian-vector product of the objective function: No suitable Julia function available.")))
    end
end
function make_jvp(t::NotInplaceObjective, ::AbstractArray, ::AbstractArray)
    (; jvp, fjvp, df, fdf) = t
    if jvp !== nothing
        return let jvp = jvp
            (JVP, x, v) -> copyto!(JVP, jvp(x, v))
        end
    elseif fjvp !== nothing
        return let fjvp = fjvp
            (JVP, x, v) -> copyto!(JVP, fjvp(x, v)[2])
        end
    elseif df !== nothing
        return let df = df
            (JVP, x, v) -> LinearAlgebra.mul!(JVP, df(x), v)
        end
    elseif fdf !== nothing
        return let fdf = fdf
            (JVP, x, v) -> LinearAlgebra.mul!(JVP, last(fdf(x)), v)
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` are not appropriate for functions with `::AbstractArray` output
        return ((JVP, x, v) -> throw(ArgumentError("Cannot evaluate the Jacobian-vector product of the objective function: No suitable Julia function available.")))
    end
end

function make_fjvp(t::NotInplaceObjective, ::AbstractArray, ::Real)
    (; fjvp, fdf, fgh) = t
    if fjvp !== nothing
        return fjvp
    elseif fdf !== nothing
        return let fdf = fdf
            (x, v) -> begin
                fx, gx = fdf(x)
                return fx, LinearAlgebra.dot(gx, v)
            end
        end
    elseif fgh !== nothing
        return let fgh = fgh
            (x, v) -> begin
                fx, gx, _ = fgh(x)
                return fx, LinearAlgebra.dot(gx, v)
            end
        end
    else
        return ((x, v) -> throw(ArgumentError("Cannot evaluate the objective function and its Jacobian-vector product: No suitable Julia function available.")))
    end
end
function make_fjvp(t::NotInplaceObjective, ::AbstractArray, ::AbstractArray)
    (; fjvp, fdf) = t
    if fjvp !== nothing
        return let fjvp = fjvp
            (F, JVP, x, v) -> begin
                fx, jvp = fjvp(x, v)
                copyto!(F, fx)
                copyto!(JVP, jvp)
                return F, JVP
            end
        end
    elseif fdf !== nothing
        return let fdf = fdf
            (F, JVP, x, v) -> begin
                fx, jx = fdf(x)
                copyto!(F, fx)
                LinearAlgebra.mul!(JVP, jx, v)
                return F, JVP
            end
        end
    else
        # Note: Functions involving the Hessian matrix such as `fgh` and `fghv` are not appropriate for functions with `::AbstractArray` output
        return ((F, JVP, x, v) -> throw(ArgumentError("Cannot evaluate the objective function and its Jacobian-vector product: No suitable Julia function available.")))
    end
end

# Constructors
function NonDifferentiable(t::Union{InplaceObjective, NotInplaceObjective}, x::AbstractArray, F::Real = real(zero(eltype(x))))
    f = make_f(t, x, F)
    NonDifferentiable(f, x, F)
end
# this would not be possible if we could mark f, g, ... as non-AbstractArrays
function NonDifferentiable(t::Union{InplaceObjective, NotInplaceObjective}, x::AbstractArray, F::AbstractArray)
    f = make_f(t, x, F)
    NonDifferentiable(f, x, F)
end

function TwiceDifferentiable(t::InplaceObjective, x::AbstractArray, F::Real = real(zero(eltype(x))), G::AbstractArray = alloc_DF(x, F), H::AbstractMatrix = alloc_H(x, F))
    f   = make_f(t, x, F)
    df  = make_df(t, x, F)
    fdf = make_fdf(t, x, F)
    fdfh = make_fdfh(t, x, F)
    dfh = make_dfh(t, x, F)
    h   = make_h(t, x, F)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_h = x_of_nans(x)

    TwiceDifferentiable(f, df, fdf, dfh, fdfh, h,
                                        copy(F), copy(G), copy(H),
                                        x_f, x_df, x_h,
                                        0, 0, 0)
end

function value_gradient_hessian!!(obj, x)
    obj.f_calls += 1
    obj.df_calls += 1
    obj.h_calls += 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    copyto!(obj.x_h, x)
    if obj.fdfh === nothing
        obj.F = obj.fdf(obj.DF, x)
        obj.h(obj.H, x)
    else
        obj.F = obj.fdfh(obj.DF, obj.H, x)
    end
    obj.F, obj.DF, obj.H
end

function gradient_hessian!!(obj, x)
    if obj.dfh === nothing
        gradient!!(obj, x)
        hessian!!(obj, x)
    else
        obj.df_calls += 1
        obj.h_calls += 1
        copyto!(obj.x_df, x)
        copyto!(obj.x_h, x)
        obj.dfh(obj.DF, obj.H, x)
    end
    obj.DF, obj.H
end

function TwiceDifferentiableHV(t::InplaceObjective, x::AbstractVector, F::Real = real(zero(eltype(x))))
    f = make_f(t, x, F)
    fg = make_fdf(t, x, F)
    hv = make_hv(t, x, F)
    return TwiceDifferentiableHV(f, fg, hv, x)
end
