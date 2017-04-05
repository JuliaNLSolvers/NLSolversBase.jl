function _unchecked_value!(obj, x)
    obj.f_calls .+= 1
    copy!(obj.last_x_f, x)
    obj.f_x = obj.f(x)
end
function value(obj, x)
    if x != obj.last_x_f
        obj.f_calls += 1
        return obj.f(x)
    end
    obj.f_x
end
function value!(obj, x)
    if x != obj.last_x_f
        _unchecked_value!(obj, x)
    end
    obj.f_x
end


function _unchecked_gradient!(obj, x)
    obj.g_calls .+= 1
    copy!(obj.last_x_g, x)
    obj.g!(obj.g, x)
end
function gradient!(obj::AbstractObjective, x)
    if x != obj.last_x_g
        _unchecked_gradient!(obj, x)
    end
end

function value_gradient!(obj::AbstractObjective, x)
    if x != obj.last_x_f && x != obj.last_x_g
        obj.f_calls .+= 1
        obj.g_calls .+= 1
        obj.last_x_f[:], obj.last_x_g[:] = copy(x), copy(x)
        obj.f_x = obj.fg!(obj.g, x)
    elseif x != obj.last_x_f
        _unchecked_value!(obj, x)
    elseif x != obj.last_x_g
        _unchecked_gradient!(obj, x)
    end
    obj.f_x
end

function _unchecked_hessian!(obj::AbstractObjective, x)
    obj.h_calls .+= 1
    copy!(obj.last_x_h, x)
    obj.h!(obj.H, x)
end
function hessian!(obj::AbstractObjective, x)
    if x != obj.last_x_h
        _unchecked_hessian!(obj, x)
    end
end

# Getters are without ! and accept only an objective and index or just an objective
value(obj::AbstractObjective) = obj.f_x
gradient(obj::AbstractObjective) = obj.g
gradient(obj::AbstractObjective, i::Integer) = obj.g[i]
hessian(obj::AbstractObjective) = obj.H
