abstract type AbstractObjective end
real_to_complex(d::AbstractObjective, x) = iscomplex(d) ? real_to_complex(x) : x
complex_to_real(d::AbstractObjective, x) = iscomplex(d) ? complex_to_real(x) : x

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
alloc_DF(x, F) = zeros(eltype(x), length(x), length(x)) #similar(x, length(x), length(x))
# Initialize a gradient shaped like x
alloc_DF(x, F::Number) = similar(x)
# Initialize an n-by-n Hessian
alloc_H(x) = zeros(eltype(x), length(x), length(x)) # similar(x, length(x), length(x))
