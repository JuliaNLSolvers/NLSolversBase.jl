function value(cache, obj::AbstractObjective, x)
    if x != cache.last_x_f
        cache.f_calls .+= 1
        return obj.f(real_to_complex(cache, x))
    end
    cache.f_x
end
function value!(cache, obj::AbstractObjective, x)
    if x != cache.last_x_f
        cache.f_calls .+= 1
        copy!(cache.last_x_f, x)
        cache.f_x = obj.f(real_to_complex(cache, x))
    end
    cache.f_x
end

function gradient!(cache, obj::AbstractObjective, x)
    if x != cache.last_x_g
        cache.g_calls .+= 1
        copy!(cache.last_x_g, x)
        obj.g!(real_to_complex(cache, cache.g), real_to_complex(cache, x))
    end
end

function value_gradient!(cache, obj::AbstractObjective, x)
    if x != cache.last_x_f && x != cache.last_x_g
        cache.f_calls .+= 1
        cache.g_calls .+= 1
        copy!(cache.last_x_f, x)
        copy!(cache.last_x_g, x)
        cache.f_x = obj.fg!(real_to_complex(cache, cache.g), real_to_complex(cache, x))
    elseif x != cache.last_x_f
        cache.f_calls .+= 1
        copy!(cache.last_x_f, x)
        cache.f_x = obj.f(real_to_complex(cache, x))
    elseif x != cache.last_x_g
        cache.g_calls .+= 1
        copy!(cache.last_x_g, x)
        obj.g!(real_to_complex(cache, cache.g), real_to_complex(cache, x))
    end
    cache.f_x
end

function hessian!(cache, obj::AbstractObjective, x)
    if x != cache.last_x_h
        cache.h_calls .+= 1
        copy!(cache.last_x_h, x)
        obj.h!(cache.H, x)
    end
end

# Getters are without ! and accept only an objective cache and index or just an objective cache
value(cache::AbstractObjectiveCache) = cache.f_x
gradient(cache::AbstractObjectiveCache) = cache.g
gradient(cache::AbstractObjectiveCache, i::Integer) = cache.g[i]
hessian(cache::AbstractObjectiveCache) = cache.H
