# To pass incomplete specifications, we provide the "only_"
# family of functions. It allows the users to simply provide
# fg! for optimization or fj! for solving systems of equations
# and have NLSolversBase wrap them up properly so "F" can be 
# calculated on its own and the save for partial derivatives.
# Note, that for TwiceDifferentiable we cannot provide con-
# structors if h == nothing, as that requires automatic dif-
# fferentiation of some sort.

# Mutating version
only_fg!(fg, h = nothing)   = (nothing, nothing, fg, h, true)
only_fj!(fj, h = nothing)   = (nothing, nothing, fj, h, true)

only_g!_and_fg!(g, fg, h = nothing)     = (nothing, g, fg, h, true)
only_j!_and_fj!(j, fj, h = nothing)     = (nothing, j, fj, h, true)

only_f_and_fg!(f, fg, h = nothing)      = (f, nothing, fg, h, true)
only_f!_and_fj!(f, fj, h = nothing)     = (f, nothing, fj, h, true)

# Non-mutating version
only_fg(fg, h = nothing)   = (nothing, nothing, fg, h, false)
only_fj(fj, h = nothing)   = (nothing, nothing, fj, h, false)

only_g_and_fg(g, fg, h = nothing)     = (nothing, g, fg, h, false)
only_j_and_fj(j, fj, h = nothing)     = (nothing, j, fj, h, false)

only_f_and_fg(f, fg, h = nothing)     = (f, nothing, fg, h, false)
only_f_and_fj(f, fj, h = nothing)     = (f, nothing, fj, h, false)

function make_f_from_t(t, x, F::TF) where TF
    fdf = t[3]
    if t[end]
        # Mutating version
        if TF <: Real
            return x -> fdf(alloc_DF(x, F), x)
        else
            return (F, x) -> fdf(F, alloc_DF(x, F), x)
        end
    else
        # Non-mutating version
        # The contract with the user is that fdf returns (F, DF)
        # and then we simply need to pick out the first element
        # of whatever fdf returns.
        if TF <: Real
            return x -> fdf(x)[1]
        else
            return (F, x) -> copy!(F, fdf(x)[1])
        end
    end
end
function make_f_and_df_from_t(t, x, F::TF) where TF
    fdf = t[3]
    if t[end]
        # Mutating version
        if TF <: Real
            f(x) = fdf(alloc_DF(x, F), x)[1]
            g!(G, x) = fdf(G, x)
            return f, g!
        else
            f!(F, x) = fdf(F, alloc_DF(x, F), x)
            j!(J, x) = fdf(similar(F), J, x)
            return f!, j!
        end
    else
        # Non-mutating version
        # The contract with the user is that fdf returns (F, DF)
        # and then we simply need to pick out the first element
        # of whatever fdf returns. We need weird names to avoid
        # overwritten method warning.
        if TF <: Real
            ff(x) = fdf(x)[1]
            gg!(G, x) = copy!(G, fdf(x)[2])
            function ffgg!(G, x)
                f, g = fdf(x)
                copy!(G, g)
                f
            end
            return ff, gg!, ffgg!
        else
            ff!(F, x) = copy!(F, fdf(x)[1])
            jj!(J, x) = copy!(J, fdf(x)[2])
            function ffjj!(F, J, x)
                f, j = fdf(x)
                copy!(J, j)
                copy!(F, f)
            end
            return ff!, jj!, ffjj!
        end
    end
end
function make_df_from_t(t, x, F::TF) where TF
    fdf = t[3]
    if t[end]
        # Mutating version
        if TF <: Real
            return fdf
        else
            return (J, x) -> fdf(similar(F), J, x)
        end
    else
        # Non-mutating version
        # The contract with the user is that fdf returns (F, DF)
        # and then we simply need to pick out the first element
        # of whatever fdf returns.
        if TF <: Real
            return (G, x) -> copy!(G, fdf(x)[2])
        else
            return (J, x) -> copy!(J, fdf(x)[2])
        end
    end
end

# Constructors
NonDifferentiable(t::Tuple, x::AbstractArray, F::Real = real(zero(eltype(x)))) =
    NonDifferentiable(make_f_from_t(t, x, F), x, F)
NonDifferentiable(t::Tuple, x::AbstractArray, F::AbstractArray) =
    NonDifferentiable(make_f_from_t(t, x, F), x, F)

function OnceDifferentiable(t::Tuple, x::AbstractArray, F::Real = real(zero(eltype(x))))
    f, g!, fg! = make_f_and_df_from_t(t, x, F)
    OnceDifferentiable(f, g!, fg!, x, F)
end
function OnceDifferentiable(t::Tuple, x::AbstractArray, F::AbstractArray)
    f!, j!, fj! = make_f_and_df_from_t(t, x, F)
    OnceDifferentiable(f!, j!, fj!, x, F)
end

function TwiceDifferentiable(t::Tuple, x::AbstractArray, F::Real = real(zero(eltype(x))))
    f, g! = make_f_and_df_from_t(t, x, F)
    TwiceDifferentiable(f, g!, t[3], t[4], x, F)
end
function TwiceDifferentiable(t::Tuple{TF, TG, TFG, TH, TB}, x::AbstractArray, F::Real = real(zero(eltype(x)))) where {TF, TG, TFG, TH <: Void, TB}
    throw(MethodError())
end