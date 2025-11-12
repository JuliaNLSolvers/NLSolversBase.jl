# To pass incomplete specifications, we provide the "only_"
# family of functions. It allows the users to simply provide
# fg! for optimization or fj! for solving systems of equations
# and have NLSolversBase wrap them up properly so "F" can be
# calculated on its own and the save for partial derivatives.
# Note, that for TwiceDifferentiable we cannot provide con-
# structors if h === nothing, as that requires automatic dif-
# fferentiation of some sort.
struct InplaceObjective{DF, FDF, FGH, Hv, FGHv}
    df::DF
    fdf::FDF
    fgh::FGH
    hv::Hv
    fghv::FGHv
end
InplaceObjective(; df = nothing, fdf = nothing, fgh = nothing, hv = nothing, fghv = nothing) = InplaceObjective(df, fdf, fgh, hv, fghv)
const InPlaceObjectiveFGH = InplaceObjective{<:Nothing, <:Nothing, <:Any, <:Nothing, <:Nothing}
const InPlaceObjectiveFG_Hv = InplaceObjective{<:Nothing, <:Any, <:Nothing, <:Any, <:Nothing}
const InPlaceObjectiveFGHv = InplaceObjective{<:Nothing, <:Nothing, <:Nothing, <:Nothing, <:Any}
struct NotInplaceObjective{DF, FDF, FGH}
    df::DF
    fdf::FDF
    fgh::FGH
end
# Mutating version
only_fg!(fg) = InplaceObjective(fdf = fg)
only_fgh!(fgh) = InplaceObjective(fgh = fgh)
only_fj!(fj) = InplaceObjective(fdf = fj)

only_fg_and_hv!(fg, hv) = InplaceObjective(fdf = fg, hv = hv)
only_fghv!(fghv) = InplaceObjective(fghv = fghv)

# Non-mutating version
only_fg(fg) = NotInplaceObjective(nothing, fg, nothing)
only_fj(fj) = NotInplaceObjective(nothing, fj, nothing)

only_g_and_fg(g, fg) = NotInplaceObjective(g, fg, nothing)
only_j_and_fj(j, fj) = NotInplaceObjective(j, fj, nothing)

df(t::Union{InplaceObjective, NotInplaceObjective}) = t.df
fdf(t::Union{InplaceObjective, NotInplaceObjective}) = t.fdf

# Mutating version
function make_f(t::InplaceObjective, x, F::Real)
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
        throw(ArgumentError("Cannot construct function for evaluating the objective function: No suitable function was provided."))
    end
end
make_f(t::InplaceObjective, x, F) = let fdf = t.fdf
    (F, x) -> fdf(F, nothing, x)
end

function make_df(t::InplaceObjective, x, F)
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
        throw(ArgumentError("Cannot construct function for evaluating the gradient of the objective function: No suitable function was provided.."))
    end
end

function make_fdf(t::InplaceObjective, x, F::Real)
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
        throw(ArgumentError("Cannot construct function that evaluates both the objective function and its gradient: No suitable function was provided."))
    end
end
make_fdf(t::InplaceObjective, x, F) = fdf(t)

# Non-mutating version
# The contract with the user is that fdf returns (F, DF)
# and then we simply need to pick out the appropriate element
# of whatever fdf returns.
make_f(t::NotInplaceObjective, x, F::Real) = x -> fdf(t)(x)[1]
make_f(t::NotInplaceObjective, x, F) = (F, x) -> copyto!(F, fdf(t)(x)[1])
make_df(t::NotInplaceObjective{DF, TDF}, x, F) where {DF <: Nothing, TDF} = (DF, x) -> copyto!(DF, fdf(t)(x)[2])
make_df(t::NotInplaceObjective, x, F) = t.df
function make_fdf(t::NotInplaceObjective, x, F::Real)
    return function ffgg!(G, x)
        f, g = fdf(t)(x)
        copyto!(G, g)
        return f
    end
end
function make_fdf(t::NotInplaceObjective, x, F)
    return function ffjj!(F, J, x)
        f, j = fdf(t)(x)
        copyto!(J, j)
        return copyto!(F, f)
    end
end

# Constructors
function NonDifferentiable(t::Union{InplaceObjective, NotInplaceObjective}, x::AbstractArray, F::Real = real(zero(eltype(x))))
    f = make_f(t, x, F)
    return NonDifferentiable(f, x, F)
end
# this would not be possible if we could mark f, g, ... as non-AbstractArrays
function NonDifferentiable(t::Union{InplaceObjective, NotInplaceObjective}, x::AbstractArray, F::AbstractArray)
    f = make_f(t, x, F)
    return NonDifferentiable(f, x, F)
end

const InPlaceFGH = InplaceObjective{<:Nothing, <:Nothing, TH, <:Nothing, <:Nothing} where {TH}
const InPlaceFG_HV = InplaceObjective{<:Nothing, TFG, <:Nothing, THv, <:Nothing} where {TFG, THv}
const InPlaceFGHV = InplaceObjective{<:Nothing, <:Nothing, <:Nothing, <:Nothing, TFGHv} where {TFGHv}
function TwiceDifferentiable(t::InPlaceFGH, x::AbstractArray, F::Real = real(zero(eltype(x))), G::AbstractArray = alloc_DF(x, F), H::AbstractMatrix = alloc_H(x, F))
    f = x -> t.fgh(F, nothing, nothing, x)
    df = (G, x) -> t.fgh(nothing, G, nothing, x)
    fdf = (G, x) -> t.fgh(F, G, nothing, x)
    fdfh = (G, H, x) -> t.fgh(F, G, H, x)
    dfh = (G, H, x) -> t.fgh(nothing, G, H, x)
    h = (H, x) -> t.fgh(F, nothing, H, x)

    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)

    return TwiceDifferentiable(
        f, df, fdf, dfh, fdfh, h,
        copy(F), copy(G), copy(H),
        x_f, x_df, x_h,
        0, 0, 0
    )
end
function TwiceDifferentiable(t::InPlaceFGH, x::AbstractVector{T}, F::Real = real(zero(eltype(x))), G::AbstractVector{Tx} = alloc_DF(x, F)) where {T, Tx}
    f = x -> t.fgh(F, nothing, nothing, x)
    df = (G, x) -> t.fgh(nothing, G, nothing, x)
    fdf = (G, x) -> t.fgh(F, G, nothing, x)
    fdfh = (G, H, x) -> t.fgh(F, G, H, x)
    dfh = (G, H, x) -> t.fgh(nothing, G, H, x)
    h = (H, x) -> t.fgh(F, nothing, H, x)

    H = alloc_H(x, F)
    x_f, x_df, x_h = x_of_nans(x), x_of_nans(x), x_of_nans(x)

    return TwiceDifferentiable(
        f, df, fdf, dfh, fdfh, h,
        copy(F), copy(G), copy(H),
        x_f, x_df, x_h,
        0, 0, 0
    )
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
    return obj.F, obj.DF, obj.H
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
    return obj.DF, obj.H
end

function TwiceDifferentiableHV(t::InPlaceFG_HV, x::AbstractVector)
    return TwiceDifferentiableHV(nothing, t.fdf, t.hv, x)
end

function TwiceDifferentiableHV(t::InPlaceFGHV, x::AbstractVector, F::Real = real(zero(eltype(x))))
    fg = (F, G, x) -> t.fghv(F, G, nothing, x, nothing)
    Hv = (Hv, x, v) -> t.fghv(nothing, nothing, Hv, x, v)
    return TwiceDifferentiableHV(nothing, fg, Hv, x)
end
