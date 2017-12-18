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
        gradient!!(obj, x)
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

jacobian!(obj, x) = jacobian!(obj, obj.DF, x)
function jacobian!(obj, DF, x)
    if x != obj.x_df
        jacobian!!(obj, DF, x)
    end
end
function jacobian!!(obj, fjac, x)
    obj.df(fjac, x)
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
