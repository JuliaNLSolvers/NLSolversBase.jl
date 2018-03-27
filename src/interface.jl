"""
Force (re-)evaluation of the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function value!!(obj::AbstractObjective, x)
    obj.f_calls .+= 1
    copyto!(obj.x_f, x)
    obj.F = obj.f(real_to_complex(obj, x))
end
"""
Evaluates the objective value at `x`.

Returns `f(x)`, but does *not* store the value in `obj.F`
"""
function value(obj::AbstractObjective, x)
    if x != obj.x_f
        obj.f_calls .+= 1
        return obj.f(real_to_complex(obj,x))
    end
    obj.F
end
"""
Evaluates the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function value!(obj::AbstractObjective, x)
    if x != obj.x_f
        value!!(obj, x)
    end
    obj.F
end

"""
Evaluates the gradient value at `x`

This does *not* update `obj.DF`.
"""
function gradient(obj::AbstractObjective, x)
    if x != obj.x_df
        tmp = copy(obj.DF)
        gradient!!(obj, x)
        newdf = copy(obj.DF)
        copyto!(obj.DF, tmp)
        return newdf
    end
    obj.DF
end
"""
Evaluates the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function gradient!(obj::AbstractObjective, x)
    if x != obj.x_df
        gradient!!(obj, x)
    end
end
"""
Force (re-)evaluation of the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function gradient!!(obj::AbstractObjective, x)
    obj.df_calls .+= 1
    copyto!(obj.x_df, x)
    obj.df(real_to_complex(obj, obj.DF), real_to_complex(obj, x))
end

function value_gradient!(obj::AbstractObjective, x)
    if x != obj.x_f && x != obj.x_df
        value_gradient!!(obj, x)
    elseif x != obj.x_f
        value!!(obj, x)
    elseif x != obj.x_df
        gradient!!(obj, x)
    end
    obj.F
end
function value_gradient!!(obj::AbstractObjective, x)
    obj.f_calls .+= 1
    obj.df_calls .+= 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    obj.F = obj.fdf(real_to_complex(obj, obj.DF), real_to_complex(obj, x))
end

function hessian!(obj::AbstractObjective, x)
    if x != obj.x_h
        hessian!!(obj, x)
    end
end
function hessian!!(obj::AbstractObjective, x)
    obj.h_calls .+= 1
    copyto!(obj.x_h, x)
    obj.h(obj.H, x)
end

# Getters are without ! and accept only an objective and index or just an objective
"Get the most recently evaluated objective value of `obj`."
value(obj::AbstractObjective) = obj.F
"Get the most recently evaluated gradient of `obj`."
gradient(obj::AbstractObjective) = obj.DF
"Get the most recently evaluated Jacobian of `obj`."
jacobian(obj::AbstractObjective) = obj.DF
"Get the `i`th element of the most recently evaluated gradient of `obj`."
gradient(obj::AbstractObjective, i::Integer) = obj.DF[i]
"Get the most recently evaluated Hessian of `obj`"
hessian(obj::AbstractObjective) = obj.H

value_jacobian!(obj, x) = value_jacobian!(obj, obj.F, obj.DF, x)
function value_jacobian!(obj, F, DF, x)
    if x != obj.x_f && x != obj.x_df
        value_jacobian!!(obj, F, DF, x)
    elseif x != obj.x_f
        value!!(obj, x)
    elseif x != obj.x_df
        jacobian!!(obj, x)
    end
end
value_jacobian!!(obj, x) = value_jacobian!!(obj, obj.F, obj.DF, x)
function value_jacobian!!(obj, F, J, x)
    obj.fdf(F, J, x)
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    obj.f_calls .+= 1
    (obj.df_calls .+= 1; nothing)
    obj.df_calls
end

function jacobian!(obj, x)
    if x != obj.x_df
        jacobian!!(obj, x)
    end
end
function jacobian!!(obj, x)
    obj.df(obj.DF, x)
    copyto!(obj.x_df, x)
    (obj.df_calls .+= 1; nothing)
    obj.df_calls
end
function jacobian(obj::AbstractObjective, x)
    if x != obj.x_df
        tmp = copy(obj.DF)
        jacobian!!(obj, x)
        newdf = copy(obj.DF)
        copyto!(obj.DF, tmp)
        return newdf
    end
    obj.DF
end

value!!(obj::NonDifferentiable{TF, TX, Tcplx}, x) where {TF<:AbstractArray, TX, Tcplx} = value!!(obj, obj.F, x)
value!!(obj::OnceDifferentiable{TF, TDF, TX, Tcplx}, x) where {TF<:AbstractArray, TDF, TX, Tcplx} = value!!(obj, obj.F, x)
function value!!(obj, F, x)
    obj.f(F, x)
    copyto!(obj.x_f, x)
    (obj.f_calls .+= 1; nothing)
    obj.f_calls
end

function _clear_f!(d::NLSolversBase.AbstractObjective)
    d.f_calls .= 0
    if typeof(d.F) <: AbstractArray
        d.F .= eltype(d.F)(NaN)
    else
        d.F = typeof(d.F)(NaN)
    end
    (d.x_f .= eltype(d.x_f)(NaN); nothing)
    d.x_f
end

function _clear_df!(d::NLSolversBase.AbstractObjective)
    d.df_calls .= 0
    d.DF .= eltype(d.DF)(NaN)
    (d.x_df .= eltype(d.x_df)(NaN); nothing)
    d.x_df
end

function _clear_h!(d::NLSolversBase.AbstractObjective)
    d.h_calls .= 0
    d.H .= eltype(d.H)(NaN)
    (d.x_h .= eltype(d.x_h)(NaN); nothing)
    d.x_h
end

function _clear_hv!(d::NLSolversBase.AbstractObjective)
    d.hv_calls .= 0
    d.Hv .= eltype(d.Hv)(NaN)
    d.x_hv .= eltype(d.x_hv)(NaN)
    (d.v_hv .= eltype(d.v_h)(NaN); nothing)
    d.v_hv
end

clear!(d::NonDifferentiable)  = _clear_f!(d)

function clear!(d::OnceDifferentiable)
    _clear_f!(d)
    _clear_df!(d)
end

function clear!(d::TwiceDifferentiable)
    _clear_f!(d)
    _clear_df!(d)
    _clear_h!(d)
end

function clear!(d::TwiceDifferentiableHV)
    _clear_f!(d)
    _clear_df!(d)
    _clear_hv!(d)
end
