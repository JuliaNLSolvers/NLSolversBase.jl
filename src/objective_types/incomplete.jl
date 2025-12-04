# To pass incomplete specifications, we provide the "only_"
# family of functions. It allows the users to simply provide
# fg! for optimization or fj! for solving systems of equations
# and have NLSolversBase wrap them up properly so "F" can be
# calculated on its own and the save for partial derivatives.
# Note, that for TwiceDifferentiable we cannot provide con-
# structors if h === nothing, as that requires automatic dif-
# fferentiation of some sort.
@kwdef struct InplaceObjective{FDF, FGH, Hv, FGHv, FJVP}
    fdf::FDF = nothing
    fgh::FGH = nothing
    hv::Hv = nothing
    fghv::FGHv = nothing
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

only_fg_and_hv!(fg, hv) = InplaceObjective(; fdf=fg, hv=hv)
only_fghv!(fghv)        = InplaceObjective(; fghv=fghv)

# Non-mutating version
only_fg(fg) = NotInplaceObjective(; fdf=fg)
only_fgh(fgh) = NotInplaceObjective(; fgh) 

only_g_and_fg(g, fg)    = NotInplaceObjective(; df=g, fdf=fg)

# Aliases with `j` (Jacobian)
# Do not apply to functions that involve Hessians!
# Maybe we should clearly separate functions that take gradients (=> scalar-valued functions) and Jacobians (=> array-valued functions)?
const only_fj! = only_fg!
const only_fj = only_fg
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
        # By returning `nothing`, `hv_product!` will reuse the cache for the Hessian of `TwiceDifferentiable`
        return nothing
    end
end

function make_jvp(t::InplaceObjective, ::AbstractArray, F::Real)
    (; fjvp) = t
    if fjvp !== nothing
        return let fjvp = fjvp, F = F
            (x, v) -> last ∘ fjvp(nothing, F, x, v)
        end
    else
        # By returning `nothing`, `jvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
    end
end
function make_jvp(t::InplaceObjective, ::AbstractArray, F::AbstractArray)
    (; fjvp) = t
    if fjvp !== nothing
        return let fjvp = fjvp
            (JVP, x, v) -> last ∘ fjvp(nothing, JVP, x, v)
        end
    else
        # By returning `nothing`, `jvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
    end
end

function make_fjvp(t::InplaceObjective, ::AbstractArray, F::Real)
    (; fjvp) = t
    if fjvp !== nothing
        return let fjvp = fjvp, F = F
            (x, v) -> fjvp(F, F, x, v)
        end
    else
        # By returning `nothing`, `fjvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
    end
end
function make_fjvp(t::InplaceObjective, x::AbstractArray, F::AbstractArray)
    (; fjvp) = t
    if fjvp !== nothing
        return fjvp
    else
        # By returning `nothing`, `fjvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
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
    (; jvp, fjvp) = t
    if jvp !== nothing
        return jvp
    elseif fjvp !== nothing
        return last ∘ fjvp
    else
        # By returning `nothing`, `jvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
    end
end
function make_jvp(t::NotInplaceObjective, ::AbstractArray, ::AbstractArray)
    (; jvp, fjvp) = t
    if jvp !== nothing
        return let jvp = jvp
            (JVP, x, v) -> copyto!(JVP, jvp(x, v))
        end
    elseif fjvp !== nothing
        return let fjvp = fjvp
            (JVP, x, v) -> copyto!(JVP, fjvp(x, v)[2])
        end
    else
        # By returning `nothing`, `jvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
    end
end

function make_fjvp(t::NotInplaceObjective, ::AbstractArray, ::Real)
    (; fjvp) = t
    if fjvp !== nothing
        return fjvp
    else
        # By returning `nothing`, `fjvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
    end
end
function make_fjvp(t::NotInplaceObjective, ::AbstractArray, ::AbstractArray)
    (; fjvp) = t
    if fjvp !== nothing
        return let fjvp = fjvp
            (F, JVP, x, v) -> begin
                fx, jvp = fjvp(x, v)
                copyto!(F, fx)
                copyto!(JVP, jvp)
                return F, JVP
            end
        end
    else
        # By returning `nothing`, `fjvp!` will reuse the cache for the gradient/Jacobian of `OnceDifferentiable` and `TwiceDifferentiable`
        return nothing
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
    jvp = make_jvp(t, x, F)
    fjvp = make_fjvp(t, x, F)
    fdfh = make_fdfh(t, x, F)
    dfh = make_dfh(t, x, F)
    h   = make_h(t, x, F)
    hv = make_hv(t, x, F)

    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_jvp = x_of_nans(x)
    v_jvp = x_of_nans(x)
    x_h = x_of_nans(x)
    x_hv = x_of_nans(x)
    v_hv = x_of_nans(x)

    TwiceDifferentiable(f, df, fdf, jvp, fjvp, dfh, fdfh, h, hv,
                                        copy(F), copy(G), alloc_JVP(x, F), copy(H), copy(G),
                                        x_f, x_df, x_jvp, v_jvp, x_h, x_hv, v_hv,
                                        0, 0, 0, 0, 0)
end
