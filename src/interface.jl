"""
Force (re-)evaluation of the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function value!!(obj::Union{NonDifferentiable, OnceDifferentiable, TwiceDifferentiable}, x)
    if obj.F isa Real
        obj.F = obj.f(x)
    else
        obj.f(obj.F, x)
    end
    obj.f_calls += 1
    copyto!(obj.x_f, x)
    return obj.F
end

"""
Evaluates the objective value at `x`.

Returns `f(x)`, but does *not* store the value in `obj.F`
"""
function value(obj::Union{NonDifferentiable, OnceDifferentiable, TwiceDifferentiable}, x)
    if obj.F isa Real
        F = obj.f(x)
    else
        F = copy(obj.F)
        obj.f(F, x)
    end
    obj.f_calls += 1
    return F
end

"""
Evaluates the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function value!(obj::Union{NonDifferentiable, OnceDifferentiable, TwiceDifferentiable}, x)
    if x != obj.x_f
        value!!(obj, x)
    end
    return obj.F
end

"""
Evaluates the gradient value at `x`

This does *not* update `obj.DF` or `obj.x_df`.
"""
function gradient(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    DF = copy(obj.DF)
    obj.df(DF, x)
    obj.df_calls += 1
    return DF
end
"""
Evaluates the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function gradient!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    if x != obj.x_df
        gradient!!(obj, x)
    end
    return obj.DF
end
"""
Force (re-)evaluation of the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function gradient!!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    obj.df(obj.DF, x)
    obj.df_calls += 1
    copyto!(obj.x_df, x)
    return obj.DF
end

function value_gradient!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    if x != obj.x_f
        if x != obj.x_df
            value_gradient!!(obj, x)
        else
            value!!(obj, x)
        end
    elseif x != obj.x_df
        gradient!!(obj, x)
    end
    return obj.F, obj.DF
end
function value_gradient!!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    obj.F = obj.fdf(obj.DF, x)
    obj.f_calls += 1
    obj.df_calls += 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    return obj.F, obj.DF
end

function hessian!(obj::TwiceDifferentiable, x)
    if x != obj.x_h
        hessian!!(obj, x)
    end
    return obj.H
end
function hessian!!(obj::TwiceDifferentiable, x)
    obj.h(obj.H, x)
    obj.h_calls += 1
    copyto!(obj.x_h, x)
    return obj.H
end

# Getters are without ! and accept only an objective and index or just an objective
"Get the most recently evaluated objective value of `obj`."
value(obj::Union{NonDifferentiable, OnceDifferentiable, TwiceDifferentiable}) = obj.F
"Get the most recently evaluated gradient of `obj`."
gradient(obj::Union{OnceDifferentiable, TwiceDifferentiable}) = obj.DF
"Get the most recently evaluated Jacobian of `obj`."
jacobian(obj::Union{OnceDifferentiable, TwiceDifferentiable}) = obj.DF
"Get the `i`th element of the most recently evaluated gradient of `obj`."
gradient(obj::Union{OnceDifferentiable, TwiceDifferentiable}, i::Integer) = obj.DF[i]
"Get the most recently evaluated Hessian of `obj`"
hessian(obj::TwiceDifferentiable) = obj.H

function value_jacobian!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    if x != obj.x_f
        if x != obj.x_df
            value_jacobian!!(obj, x)
        else
        value!!(obj, x)
        end
    elseif x != obj.x_df
        jacobian!!(obj, x)
    end
    return obj.F, obj.DF
end
function value_jacobian!!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    if obj.F isa Real
        obj.F = obj.fdf(obj.DF, x)
    else
        obj.fdf(obj.F, obj.DF, x)
    end
    obj.f_calls += 1
    obj.df_calls += 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    return obj.F, obj.DF
end

function jacobian!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    if x != obj.x_df
        jacobian!!(obj, x)
    end
    return obj.DF
end
function jacobian!!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    obj.df(obj.DF, x)
    obj.df_calls += 1
    copyto!(obj.x_df, x)
    return obj.DF
end
function jacobian(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x)
    DF = copy(obj.DF)
    obj.df(DF, x)
    obj.df_calls += 1
    return DF
end

"""
    jvp!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x::AbstractArray, v::AbstractArray)

Return the Jacobian-vector product of the objective function `obj` at point `x` with tangents `v`,
and cache the results in `obj`.

!!! note
    This function does use cached results if available.
"""
function jvp!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x::AbstractArray, v::AbstractArray)
    if x != obj.x_jvp || v != obj.v_jvp
        jvp!!(obj, x, v)
    end
    return obj.JVP
end

# Internal: Unconditionally evaluate the JVP
function jvp!!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x::AbstractArray, v::AbstractArray)
    if obj.jvp !== nothing
        if obj.F isa Real
            obj.JVP = obj.jvp(x, v)
        else
            obj.jvp(obj.JVP, x, v)
        end
        obj.jvp_calls += 1
    else
        jacobian!(obj, x)
        if obj.F isa Real
            obj.JVP = LinearAlgebra.dot(obj.DF, v)
        else
            LinearAlgebra.mul!(obj.JVP, obj.DF, v)
        end
    end
    copyto!(obj.x_jvp, x)
    copyto!(obj.v_jvp, v)
    return obj.JVP
end        

"""
    value_jvp!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x::AbstractArray, v::AbstractArray)

Return the value and the Jacobian-vector product of the objective function `obj` at point `x` with tangents `v`,
and cache the results in `obj`.

!!! note
    This function does use cached results if available.
"""
function value_jvp!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x::AbstractArray, v::AbstractArray)
    if x != obj.x_f
        if x != obj.x_jvp || v != obj.v_jvp
            # Both value and Jacobian-vector product have to be evaluated
            value_jvp!!(obj, x, v)
        else
            # Only value has to be evaluated
            value!!(obj, x)
        end
    elseif x != obj.x_jvp || v != obj.v_jvp
        # Only JVP has to be evaluated
        jvp!!(obj, x, v)
    end
    return obj.F, obj.JVP
end

# Internal: Unconditionally evaluate the function + JVP
function value_jvp!!(obj::Union{OnceDifferentiable, TwiceDifferentiable}, x, v)
    if obj.fjvp !== nothing
        if obj.F isa Real
            y, ty = obj.fjvp(x, v)
            obj.F = y
            obj.JVP = ty
        else
            obj.fjvp(obj.F, obj.JVP, x, v)
        end
        obj.f_calls += 1
        obj.jvp_calls += 1
        copyto!(obj.x_f, x)
    else
        value_jacobian!(obj, x)
        if obj.F isa Real
            obj.JVP = LinearAlgebra.dot(obj.DF, v)
        else
            LinearAlgebra.mul!(obj.JVP, obj.DF, v)
        end
    end
    copyto!(obj.x_jvp, x)
    copyto!(obj.v_jvp, v)
    return obj.F, obj.JVP
end

value(obj::NonDifferentiable{TF, TX}, x) where {TF<:AbstractArray, TX} = value(obj, copy(obj.F), x)
value(obj::OnceDifferentiable{TF, TDF, TX}, x) where {TF<:AbstractArray, TDF, TX} = value(obj, copy(obj.F), x)
function value(obj::AbstractObjective, F, x)
    obj.f_calls += 1
    return obj.f(F, x)
end

value!!(obj::NonDifferentiable{TF, TX}, x) where {TF<:AbstractArray, TX} = value!!(obj, obj.F, x)
value!!(obj::OnceDifferentiable{TF, TDF, TX}, x) where {TF<:AbstractArray, TDF, TX} = value!!(obj, obj.F, x)
function value!!(obj::AbstractObjective, F, x)
    obj.f(F, x)
    copyto!(obj.x_f, x)
    obj.f_calls += 1
    obj.f_calls
    F
end

"""
    hv_product!(obj::TwiceDifferentiable, x::AbstractArray, v::AbstractArray)

Return the Hessian-vector product of the objective function `obj` at point `x` with tangents `v`,
and cache the results in `obj`.

!!! note
    This function does use cached results if available.
"""
function hv_product!(obj::TwiceDifferentiable, x, v)
    if x != obj.x_hv || v != obj.v_hv
        hv_product!!(obj, x, v)
    end
    return obj.Hv
end

# Internal function: Unconditionally evaluate the Hessian-vector product
function hv_product!!(obj::TwiceDifferentiable, x, v)
    if obj.hv !== nothing
        obj.hv(obj.Hv, x, v)
        obj.hv_calls += 1
    else
        hessian!(obj, x)
        LinearAlgebra.mul!(obj.Hv, obj.H, v)
    end
    copyto!(obj.x_hv, x)
    copyto!(obj.v_hv, v)
    return obj.Hv
end

"""
    value_gradient_hessian!(obj::TwiceDifferentiable)

Return the value, gradient, and Hessian of the objective function `obj` at point `x`,
and cache the results in `obj`.

!!! note
    This function does use cached results if available.
"""
function value_gradient_hessian!(obj::TwiceDifferentiable, x)
    if x != obj.x_f
        if x != obj.x_h
            # There's no function for evaluating only the function value + Hessian,
            # so we evaluate objective function value, gradient, and Hessian,
            # regardless of whether the gradient is cached or not.
            value_gradient_hessian!!(obj, x)
        elseif x != obj.x_df
            # Only function value and gradient have to be evaluated
            value_gradient!!(obj, x)
        end
    elseif x != obj.x_df
        if x != obj.x_h
            # Gradient and Hessian have to be evaluated
            gradient_hessian!!(obj, x)
        else
            # Only gradient has to be evaluated
            gradient!!(obj, x)
        end
    elseif x != obj.x_h
        # Only Hessian has to be evaluated
        hessian!!(obj, x)
    end
    return obj.F, obj.DF, obj.H
end

# Internal function: Unconditionally evaluate the value, gradient, and Hessian of the objective function
function value_gradient_hessian!!(obj::TwiceDifferentiable, x)
    obj.F = obj.fdfh(obj.DF, obj.H, x)
    obj.f_calls += 1
    obj.df_calls += 1
    obj.h_calls += 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    copyto!(obj.x_h, x)
    return obj.F, obj.DF, obj.H
end

"""
    gradient_hessian!(obj::TwiceDifferentiable)

Return the gradient and Hessian of the objective function `obj` at point `x`,
and cache the results in `obj`.

!!! note
    This function does use cached results if available.
"""
function gradient_hessian!(obj::TwiceDifferentiable, x)
    if x != obj.x_df
        if x != obj.x_h
            # Gradient and Hessian have to be evaluated
            gradient_hessian!!(obj, x)
        else
            # Only gradient has to be evaluated
            gradient!!(obj, x)
        end
    elseif x != obj.x_h
        # Only Hessian has to be evaluated
        hessian!!(obj, x)
    end
    obj.DF, obj.H
end

# Internal function: Unconditionally evaluate the gradient and the Hessian of the objective function
function gradient_hessian!!(obj::TwiceDifferentiable, x)
    obj.dfh(obj.DF, obj.H, x)
    obj.df_calls += 1
    obj.h_calls += 1
    copyto!(obj.x_df, x)
    copyto!(obj.x_h, x)
    return obj.DF, obj.H
end

function _clear_f!(d::AbstractObjective)
    d.f_calls = 0
    if d.F isa AbstractArray
        fill!(d.F, NaN)
    else
        d.F = NaN
    end
    fill!(d.x_f, NaN)
    nothing
end

function _clear_df!(d::AbstractObjective)
    d.df_calls = 0
    fill!(d.DF, NaN)
    fill!(d.x_df, NaN)
    nothing
end

function _clear_jvp!(d::AbstractObjective)
    d.jvp_calls = 0
    if d.JVP isa AbstractArray
        fill!(d.JVP, NaN)
    else
        d.JVP = NaN
    end
    fill!(d.x_jvp, NaN)
    fill!(d.v_jvp, NaN)
    nothing
end

function _clear_h!(d::AbstractObjective)
    d.h_calls = 0
    fill!(d.H, NaN)
    fill!(d.x_h, NaN)
    nothing
end

function _clear_hv!(d::AbstractObjective)
    d.hv_calls = 0
    fill!(d.Hv, NaN)
    fill!(d.x_hv, NaN)
    fill!(d.v_hv, NaN)
    nothing
end

clear!(d::NonDifferentiable)  = _clear_f!(d)

function clear!(d::OnceDifferentiable)
    _clear_f!(d)
    _clear_df!(d)
    _clear_jvp!(d)
    nothing
end

function clear!(d::TwiceDifferentiable)
    _clear_f!(d)
    _clear_df!(d)
    _clear_jvp!(d)
    _clear_h!(d)
    _clear_hv!(d)
    nothing
end

f_calls(d::Union{NonDifferentiable, OnceDifferentiable, TwiceDifferentiable}) = d.f_calls
g_calls(d::NonDifferentiable) = 0
g_calls(d::Union{OnceDifferentiable, TwiceDifferentiable}) = d.df_calls
jvp_calls(d::NonDifferentiable) = 0
jvp_calls(d::Union{OnceDifferentiable, TwiceDifferentiable}) = d.jvp_calls
h_calls(d::Union{NonDifferentiable, OnceDifferentiable}) = 0
h_calls(d::TwiceDifferentiable) = d.h_calls
hv_calls(d::Union{NonDifferentiable, OnceDifferentiable}) = 0
hv_calls(d::TwiceDifferentiable) = d.hv_calls
