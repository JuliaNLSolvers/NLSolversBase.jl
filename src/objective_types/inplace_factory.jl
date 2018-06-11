"""
    f!_from_f(f, F::Abstractarray)

Return an inplace version of f
"""
function f!_from_f(f, F::AbstractArray, inplace)
    if inplace
        return f
    else
        return function ff!(F, x)
            copyto!(F, f(x))
            F
        end
    end
end
function df!_from_df(g, F::Real, inplace)
    if inplace
        return g
    else
        return function gg!(G, x)
            gx = g(x)
            copyto!(G, gx)
            G
        end
    end
end
function df!_from_df(j, F::AbstractArray, inplace)
    if inplace
        return j
    else
        return function jj!(J, x)
            jx = j(x)
            copyto!(J, jx)
            J
        end
    end
end
function fdf!_from_fdf(fg, F::Real, inplace)
    if inplace
        return fg
    else
        return function ffgg!(G, x)
            f, g = fg(x)
            copyto!(G, g)
            f
        end
    end
end
function fdf!_from_fdf(fj, F::AbstractArray, inplace)
    if inplace
        return fj
    else
        return function ffjj!(F, J, x)
            f, j = fj(x)
            copyto!(J, j)
            copyto!(F, f)
        end
    end
end
function h!_from_h(h, F::Real, inplace)
    if inplace
        return h
    else
        return function hh!(H, x)
            h = h(x)
            copyto!(H, h)
            H
        end
    end
end
