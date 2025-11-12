abstract type AbstractObjective end

# Given callables to calculate objectives and partial first derivatives
# create a function that calculates both.
function make_fdf(x, F, f!, j!)
    function fj!(fx, jx, x)
        j!(jx, x)
        return f!(fx, x)
    end
end
function make_fdf(x, F::Number, f, g!)
    function fg!(gx, x)
        g!(gx, x)
        return f(x)
    end
end

# Given Julia functions of the gradient/Jacobian and the Hessian, create a function that calculates both.
function make_dfh(::AbstractArray, ::Number, g!, h!)
    let g! = g!, h! = h!
        function gh!(gx, hx, x)
            g!(gx, x)
            h!(hx, x)
            return nothing
        end
    end
end

# Given Julia functions of the objective function + gradient/Jacobian and the Hessian, create a function that calculates all three of them.
function make_fdfh(::AbstractArray, ::Number, fg!, h!)
    let fg! = fg!, h! = h!
        function fgh!(gx, hx, x)
            fx = fg!(gx, x)
            h!(hx, x)
            return fx
        end
    end
end

# Initialize an n-by-n Jacobian
alloc_DF(x, F) = eltype(x)(NaN) .* vec(F) .* vec(x)'

# Initialize a gradient shaped like x
alloc_DF(x, F::T) where T<:Number = x_of_nans(x, promote_type(eltype(x), T))
# Initialize an n-by-n Hessian
function alloc_H(x, F::T) where T<:Number
  eltype(x)(NaN).*x*x'
end

# Initialize the result of a Jacobian vector product
alloc_JVP(x, F::Number) = float(promote_type(eltype(x), typeof(F)))(NaN)
alloc_JVP(x, F::AbstractArray) = x_of_nans(vec(F), promote_type(eltype(x), eltype(F)))
