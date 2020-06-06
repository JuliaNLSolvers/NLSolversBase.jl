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

# Initialize an n-by-n Jacobian
function alloc_DF(x, F)
  a = (Base.OneTo(length(F)), Base.OneTo(length(x)))
  df = similar(F, a)
  fill!(df, NaN)
  return df
end
# Initialize a gradient shaped like x
alloc_DF(x, F::T) where T<:Number = x_of_nans(x, promote_type(eltype(x), T))
# Initialize an n-by-n Hessian
alloc_H(x, F::T) where T<:Number = alloc_DF(x, promote_type(eltype(x), T).(x))
