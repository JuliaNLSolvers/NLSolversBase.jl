function value!!(obj::AbstractObjective, x)
    obj.f_calls .+= 1
    copy!(obj.x_f, x)
    obj.F = obj.f(real_to_complex(obj, x))
end
function value(obj::AbstractObjective, x)
    if x != obj.x_f
        obj.f_calls .+= 1
        return obj.f(real_to_complex(obj,x))
    end
    obj.F
end
function value!(obj::AbstractObjective, x)
    if x != obj.x_f
        value!!(obj, x)
    end
    obj.F
end

function gradient(obj::AbstractObjective, x)
    if x != obj.x_df
        tmp = copy(obj.DF)
        gradient!!(obj, x)
        newdf = copy(obj.DF)
        copy!(obj.DF, tmp)
        return newdf
    end
    obj.DF
end
function gradient!(obj::AbstractObjective, x)
    if x != obj.x_df
        gradient!!(obj, x)
    end
end
function gradient!!(obj::AbstractObjective, x)
    obj.df_calls .+= 1
    copy!(obj.x_df, x)
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
    copy!(obj.x_f, x)
    copy!(obj.x_df, x)
    obj.F = obj.fdf(real_to_complex(obj, obj.DF), real_to_complex(obj, x))
end

function hessian!(obj::AbstractObjective, x)
    if x != obj.x_h
        hessian!!(obj, x)
    end
end
function hessian!!(obj::AbstractObjective, x)
    obj.h_calls .+= 1
    copy!(obj.x_h, x)
    obj.h(obj.H, x)
end

# Getters are without ! and accept only an objective and index or just an objective
value(obj::AbstractObjective) = obj.F
gradient(obj::AbstractObjective) = obj.DF
jacobian(obj::AbstractObjective) = gradient(obj)
gradient(obj::AbstractObjective, i::Integer) = obj.DF[i]
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
    copy!(obj.x_f, x)
    copy!(obj.x_df, x)
    obj.f_calls .+= 1
    obj.df_calls .+= 1
end

function jacobian!(obj, x)
    if x != obj.x_df
        jacobian!!(obj, x)
    end
end
function jacobian!!(obj, x)
    obj.df(obj.DF, x)
    copy!(obj.x_df, x)
    obj.df_calls .+= 1
end

value!!(obj::NonDifferentiable{TF, TX, Tcplx}, x) where {TF<:AbstractArray, TX, Tcplx} = value!!(obj, obj.F, x)
value!!(obj::OnceDifferentiable{TF, TDF, TX, Tcplx}, x) where {TF<:AbstractArray, TDF, TX, Tcplx} = value!!(obj, obj.F, x)
function value!!(obj, F, x)
    obj.f(F, x)
    copy!(obj.x_f, x)
    obj.f_calls .+= 1
end

function _clear_f!(d::NLSolversBase.AbstractObjective)
    d.f_calls .= 0
    if typeof(d.F) <: AbstractArray
        d.F .= eltype(d.F)(NaN)
    else
        d.F = typeof(d.F)(NaN)
    end
    d.x_f .= eltype(d.x_f)(NaN)
end

function _clear_df!(d::NLSolversBase.AbstractObjective)
    d.df_calls .= 0
    d.DF .= eltype(d.DF)(NaN)
    d.x_df .= eltype(d.x_df)(NaN)
end

function _clear_h!(d::NLSolversBase.AbstractObjective)
    d.h_calls .= 0
    d.H .= eltype(d.H)(NaN)
    d.x_h .= eltype(d.x_h)(NaN)
end

function _clear_hv!(d::NLSolversBase.AbstractObjective)
    d.hv_calls .= 0
    d.Hv .= eltype(d.Hv)(NaN)
    d.x_hv .= eltype(d.x_hv)(NaN)
    d.v_hv .= eltype(d.v_h)(NaN)
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
