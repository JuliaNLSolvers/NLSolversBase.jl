# To pass incomplete specifications, we provide the "only_"
# family of functions. It allows the users to simply provide
# fg! for optimization or fj! for solving systems of equations
# and have NLSolversBase wrap them up properly so "F" can be
# calculated on its own and the save for partial derivatives.
# Note, that for TwiceDifferentiable we cannot provide con-
# structors if h == nothing, as that requires automatic dif-
# fferentiation of some sort.
struct InplaceObjective{DF, FDF, FGH, Hv, FGHv}
    df::DF
    fdf::FDF
    fgh::FGH
    hv::Hv
    fghv::FGHv
end
InplaceObjective(;df=nothing, fdf=nothing, fgh=nothing, hv=nothing, fghv=nothing) = InplaceObjective(df, fdf, fgh, hv, fghv)

struct NotInplaceObjective{DF, FDF, FGH}
    df::DF
    fdf::FDF
    fgh::FGH
end
# Mutating version
only_fg!(fg)     = InplaceObjective(fdf=fg)
only_fgh!(fgh)   = InplaceObjective(fgh=fgh)
only_fj!(fj)     = InplaceObjective(fdf=fj)

only_fg_and_hv!(fg, hv) = InplaceObjective(fdf=fg, hv=hv)
only_fghv!(fghv)        = InplaceObjective(fghv=fghv)

# Non-mutating version
only_fg(fg)     = NotInplaceObjective(nothing, fg, nothing)
only_fj(fj)     = NotInplaceObjective(nothing, fj, nothing)

only_g_and_fg(g, fg)    = NotInplaceObjective(g, fg, nothing)
only_j_and_fj(j, fj)    = NotInplaceObjective(j, fj, nothing)

df(t::Union{InplaceObjective, NotInplaceObjective}) = t.df
fdf(t::Union{InplaceObjective, NotInplaceObjective}) = t.fdf

# Mutating version
make_f(t::InplaceObjective, x, F::Real) = x -> fdf(t)(F, nothing, x)
make_f(t::InplaceObjective, x, F) =  (F, x) -> fdf(t)(F, nothing, x)
make_df(t::InplaceObjective, x, F) = (DF, x) -> fdf(t)(nothing, DF, x)
make_fdf(t::InplaceObjective, x, F::Real) = (G, x) -> fdf(t)(F, G, x)
make_fdf(t::InplaceObjective, x, F) = fdf(t)

# Non-mutating version
# The contract with the user is that fdf returns (F, DF)
# and then we simply need to pick out the appropriate element
# of whatever fdf returns.
make_f(t::NotInplaceObjective, x, F::Real) = x -> fdf(t)(x)[1]
make_f(t::NotInplaceObjective, x, F) = (F, x) -> copyto!(F, fdf(t)(x)[1])
make_df(t::NotInplaceObjective{DF, TDF}, x, F) where {DF<:Nothing, TDF} = (DF, x) -> copyto!(DF, fdf(t)(x)[2])
make_df(t::NotInplaceObjective, x, F) = t.df
function make_fdf(t::NotInplaceObjective, x, F::Real)
    return function ffgg!(G, x)
        f, g = fdf(t)(x)
        copyto!(G, g)
        f
    end
end
function make_fdf(t::NotInplaceObjective, x, F)
    return function ffjj!(F, J, x)
        f, j = fdf(t)(x)
        copyto!(J, j)
        copyto!(F, f)
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

const InPlaceFGH = InplaceObjective{<:Nothing,<:Nothing,TH,<:Nothing,<:Nothing} where {TH}
const InPlaceFG_HV = InplaceObjective{<:Nothing,TFG,<:Nothing,THv,<:Nothing} where {TFG,THv}
const InPlaceFGHV = InplaceObjective{<:Nothing,<:Nothing,<:Nothing,<:Nothing,TFGHv} where {TFGHv}

function TwiceDifferentiable(t::InPlaceFGH, x::AbstractArray, F::Real = real(zero(eltype(x))), G::AbstractArray = similar(x), H = alloc_H(x)) where {TH}
    f   =     x  -> t.fgh(F, nothing, nothing, x)
    df  = (G, x) -> t.fgh(nothing, G, nothing, x)
    fdf = (G, x) -> t.fgh(F, G, nothing, x)
    h   = (H, x) -> t.fgh(F, nothing, H, x)
    TwiceDifferentiable(f, df, fdf, h, x, F, G, H)
end

function TwiceDifferentiable(t::InPlaceFGH, x::AbstractVector, F::Real = real(zero(eltype(x))), G::AbstractVector = similar(x)) where {TH}

    H = alloc_H(x)
    f   =     x  -> t.fgh(F, nothing, nothing, x)
    df  = (G, x) -> t.fgh(nothing, G, nothing, x)
    fdf = (G, x) -> t.fgh(F, G, nothing, x)
    h   = (H, x) -> t.fgh(F, nothing, H, x)
    TwiceDifferentiable(f, df, fdf, h, x, F, G, H)
end

function TwiceDifferentiableHV(t::InPlaceFG_HV, x::AbstractVector)
    TwiceDifferentiableHV(nothing, t.fdf, t.hv, x)
end

function TwiceDifferentiableHV(t::InPlaceFGHV, x::AbstractVector, F::Real = real(zero(eltype(x))))
    fg  =  (F, G, x) -> t.fghv(F, G, nothing, x, nothing)
    Hv  = (Hv, x, v) -> t.fghv(nothing, nothing, Hv, x, v)
    TwiceDifferentiableHV(nothing, fg, Hv, x)
end
