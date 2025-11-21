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

# Given Julia functions of the gradient/Jacobian, create a function that calculates the Jacobian-vector product.
function make_jvp(x::AbstractArray, F::AbstractArray, j!)
    let j! = j!, jx = alloc_DF(x, F)
        function jvp!(jvp, x, v)
            j!(jx, x)
            LinearAlgebra.mul!(jvp, jx, v)
            return jvp
        end
    end
end
function make_jvp(x::AbstractArray, F::Number, g!)
    let g! = g!, gx = alloc_DF(x, F)
        function jvp(x, v)
            g!(gx, x)
            return LinearAlgebra.dot(gx, v)
        end
    end
end

# Given a Julia function of the objective function and its gradient/Jacobian, create a function that calculates the function value and the Jacobian-vector product.
function make_fjvp(x::AbstractArray, F::AbstractArray, fj!)
    let fj! = fj!, jx = alloc_DF(x, F)
        function fjvp!(fx, jvp, x, v)
            fj!(fx, jx, x)
            LinearAlgebra.mul!(jvp, jx, v)
            return fx, jvp
        end
    end
end
function make_fjvp(x::AbstractArray, F::Number, fg!)
    let fg! = fg!, gx = alloc_DF(x, F)
        function jvp(x, v)
            fx = fg!(gx, x)
            return fx, LinearAlgebra.dot(gx, v)
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
