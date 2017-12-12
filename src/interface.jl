function _unchecked_value!(obj::AbstractObjective, x)
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
        _unchecked_value!(obj, x)
    end
    obj.F
end

function _unchecked_gradient!(obj::AbstractObjective, x)
    obj.df_calls .+= 1
    copy!(obj.x_df, x)
    obj.df(real_to_complex(obj, obj.DF), real_to_complex(obj, x))
end
function gradient(obj::AbstractObjective, x)
    if x != obj.x_df
        tmp = copy(obj.x_df)
        _unchecked_gradient!(obj, x)
        newdf = copy(obj.x_df)
        copy!(obj.x_df, tmp)
        return newdf
    end
    obj.DF
end
function gradient!(obj::AbstractObjective, x)
    if x != obj.x_df
        _unchecked_gradient!(obj, x)
    end
end

function _unchecked_value_gradient!(obj::AbstractObjective, x)
    obj.f_calls .+= 1
    obj.df_calls .+= 1
    copy!(obj.x_f, x)
    copy!(obj.x_df, x)
    obj.F = obj.fdf(real_to_complex(obj, obj.DF), real_to_complex(obj, x))    
end
function value_gradient!(obj::AbstractObjective, x)
    if x != obj.x_f && x != obj.x_df
        obj.f_calls .+= 1
        obj.df_calls .+= 1
        copy!(obj.x_f, x)
        copy!(obj.x_df, x)
        obj.F = obj.fdf(real_to_complex(obj, obj.DF), real_to_complex(obj, x))
    elseif x != obj.x_f
        _unchecked_value!(obj, x)
    elseif x != obj.x_df
        _unchecked_gradient!(obj, x)
    end
    obj.F
end

function _unchecked_hessian!(obj::AbstractObjective, x)
    obj.h_calls .+= 1
    copy!(obj.x_h, x)
    obj.h(obj.H, x)
end
function hessian!(obj::AbstractObjective, x)
    if x != obj.x_h
        _unchecked_hessian!(obj, x)
    end
end

# Getters are without ! and accept only an objective and index or just an objective
value(obj::AbstractObjective) = obj.F
gradient(obj::AbstractObjective) = obj.DF
jacobian(obj::AbstractObjective) = gradient(obj)
gradient(obj::AbstractObjective, i::Integer) = obj.DF[i]
hessian(obj::AbstractObjective) = obj.H

value_jacobian!(df, x) = value_jacobian!(df, df.F, df.DF, x)
function value_jacobian!(df, fvec, fjac, x)
    df.fdf(fvec, fjac, x)
    df.f_calls .+= 1
    df.df_calls .+= 1
end

jacobian!(df, x) = jacobian!(df, df.DF, x)
function jacobian!(df, fjac, x)
    df.df(fjac, x)
    df.df_calls .+= 1
end

function value!(df::OnceDifferentiable{TF, TDF, TX, Tcplx}, x) where {TF<:AbstractArray, TDF, TX, Tcplx}
    value!(df, df.F, x)
end
function value!(df, fvec, x)
    df.f(fvec, x)
    df.f_calls .+= 1
end
