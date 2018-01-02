NLSolversBase.jl
========

Base functionality for optimization and solving systems of equations in Julia.

NLSolversBase.jl is the core, common dependency of several packages in the [JuliaNLSolvers](https://julianlsolvers.github.io) family.


| **PackageEvaluator**            |**Build Status**                                   |
|:-------------------------------:|:-------------------------------------------------:|
| [![][pkg-0.4-img]][pkg-0.4-url] | [![Build Status][build-img]][build-url]           |
| [![][pkg-0.5-img]][pkg-0.5-url] | [![Codecov branch][cov-img]][cov-url]             |
| [![][pkg-0.6-img]][pkg-0.6-url] | [![Coverage Status][coveralls-img]][coveralls-url]|


# Purpose

The package aims at establishing common ground for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl), and [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl). The common ground is mainly the types used to hold objective related callables, information about the objectives, and an interface to interact with these types.

## NDifferentiable
There are currently three main types: `NonDifferentiable`, `OnceDifferentiable`, and `TwiceDifferentiable`. There's also a more experimental `TwiceDifferentiableHV` for optimization algorithms that use Hessian-vector products. An `NDifferentiable` instance can be used to hold relevant functions for

 - Optimization: ![Objective for optimization](https://user-images.githubusercontent.com/8431156/33996090-6224581c-e0e0-11e7-8737-5dd659745dcb.gif)
 - Solving systems of equations: ![Objective for systems of equations](https://user-images.githubusercontent.com/8431156/33996088-60760c4a-e0e0-11e7-96ca-470f2731f1c7.gif)

The words in front of `Differentiable` in the type names (`Non`, `Once`, `Twice`) are not meant to indicate and specific classification of the function as such, but more the requirement of the algorithms used.

## Examples
#### Optimization
Say we want to minimize the Hosaki test function

![Himmelblau test function](https://user-images.githubusercontent.com/8431156/33995927-c5b9f950-e0df-11e7-8760-9ba792c2b331.gif)

The relevant functions are coded in Julia as
```julia
function f(x)
    a = (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4)
    return a * x[2]^2 * exp(-x[2])
end

function g!(G, x)
    G[1] = (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8)* x[2]^2 * exp(-x[2])
    G[2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) - (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
end

function fg!(G, x)
    g!(G, x)
    f(x)
end

function h!(H, x)
    H[1, 1] = (3.0 * x[1]^2 - 14.0 * x[1] + 14.0) * x[2]^2 * exp(-x[2])
    H[1, 2] = 2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    H[2, 1] =  2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    H[2, 2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * exp(-x[2]) - 4.0 * ( 1.0 - 8.0 * x[1] + 7.0 *  x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) + (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
end
```
The `NDifferentiable` interface can be used as shown below to create various objectives:
```julia
x = zeros(4)
nd   = NonDifferentiable(f, x)
od   = OnceDifferentiable(f, g!, x)
odfg = OnceDifferentiable(f, g!, fg! x)
td1  = Twicedifferentiable(f, g!, h! x)
tdfg = Twicedifferentiable(f, g!, fg!, h! x)
```
#### Multivalued objective
If we consider the gradient of the Himmelblau function above, we can try to solve ![FOCs](https://user-images.githubusercontent.com/8431156/34005673-f7bc5b52-e0fb-11e7-8bd9-86efad17cb95.gif) without caring about the objective value. Then we can still create `NDifferentiable`s, but we need to specify the cache to hold the value of ![Multivalued objective](https://user-images.githubusercontent.com/8431156/34006586-2de39a3a-e0ff-11e7-8453-48aad94c6b5e.gif). Currently, the only relevant ones are `NonDifferentiable` and `OnceDifferentiable`. `TwiceDifferentiable` could be used for higher order (tensor) methods, though they are rarely worth the cost. The relevant functions coded in Julia are:

```julia
function f!(F, x)
    F[1] = (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8)* x[2]^2 * exp(-x[2])
    F[2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) - (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
end

function j!(J, x)
    J[1, 1] = (3.0 * x[1]^2 - 14.0 * x[1] + 14.0) * x[2]^2 * exp(-x[2])
    J[1, 2] = 2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    J[2, 1] =  2.0 * (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2] * exp(-x[2])  - (x[1]^3 - 7.0 * x[1]^2 + 14.0 * x[1] - 8.0) * x[2]^2 * exp(-x[2])
    J[2, 2] = 2.0 * (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * exp(-x[2]) - 4.0 * ( 1.0 - 8.0 * x[1] + 7.0 *  x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2] * exp(-x[2]) + (1.0 - 8.0 * x[1] + 7.0 * x[1]^2 - (7.0 / 3.0) * x[1]^3 + (1.0 / 4.0) * x[1]^4) * x[2]^2 * exp(-x[2])
end

function fj!(F, G, x)
    g!(G, x)
    f!(F, x)
end
```
The `NDifferentiable` interface can be used as shown below to create various objectives:
```julia
x = zeros(4)
F = zeros(4)
nd   = NonDifferentiable(f!, x, F)
od   = OnceDifferentiable(f!, j!, x, F)
odfj = OnceDifferentiable(f!, j!, fj! x, F)
```

## Interface

To extract information about the objective, and to update given some input, we provide a function based interface. For all purposes it should be possible to use a function to extract/update information, and no field access should be necessary. Actually, we proactively discourage it, as it makes it much more difficult to make changes in the future.

### Single-valued objectives 
To retrieve relevant information about single-valued functions, the following functions are available where applicable:
```julia
# obj is the objective function defined as shown above
value(df)       # return the objective evaluated at df.x_f
gradient(df)    # return the gradient evaluated at df.x_df
gradient(df, i) # return the gradient evaluated at df.x_df
hessian(df)     # return the hessian evaluated at df.x_h
```
To update the various quantities, use:
```julia
# obj is the objective function defined as shown above
value!(df, x)     # update the objective if !(df.x_f==x) and set df.x_f to x
value!!(df, x)    # update the objective and set df.x_f to x
gradient!(df, x)  # update the gradient if !(df.x_df==x) and set df.x_df to x
gradient!!(df, x) # update the gradient and set df.x_df to x
hessian!(df,x)    # update the hessian if !(df.x_df==x) and set df.x_h to x
hessian!!(df,x)   # update the hessian and set df.x_h to x
```

### Multivalued 
To retrieve relevant information about multivalued functions, the following functions are available where applicable:
```julia
# obj is the objective function defined as shown above
value(df)    # return the objective evaluated at df.x_f
jacobian(df) # return the jacobian evaluated at df.x_df
jacobian(df) # return the jacobian evaluated at df.x_df
```
To update the various quantities, use:
```julia
# obj is the objective function defined as shown above
value!(df, x)     # update the objective if !(df.x_f==x) and set df.x_f to x
value!!(df, x)    # update the objective and set df.x_f to x
jacobian!(df, x)  # update the jacobian if !(df.x_df==x) and set df.x_df to x
jacobian!!(df, x) # update the jacobian and set df.x_df to x
```

## Special single-function interface
In some cases the objective and partial derivaties share
common terms that are expensive to calculate. One such case is if
the underlying problem requires solution of a model or simulation
of a some system. In that case the `only_fg!`/`only_fj!` and `only_fgh!`
interfaces can be used.

### Example
Say we have some common functionality in `common_calc(...)` that is used
in both the objective and partial derivative. Then we might construct a
OnceDifferentiable instance as
```julia
function f(x)
    common_calc(...)
    # calculations specific to f
    return f
end
function g!(G, x)
    common_calc(...)
    # mutating calculations specific to g!
end
OnceDifferentiable(f, g!, x0)
```
However, in many algorithms `f` and `g!` are evaluated together, so the
common calculations are done twice instead of once. We can use the special
interface as shown below.
```julia
function fg!(F, G, x)
    common_calc(...)
    if !(G == nothing)
        # mutating calculations specific to g!
    end
    if !(F == nothing)
        # calculations specific to f
        return f
    end
end
OnceDifferentiable(only_fg!(fg!), x0)
```
Notice the important check in the `if` statements. This makes sure that `G` is only
updated when we want to, and, if only `G` is to be updated, that we don't calculate 
the objective.

[build-img]: https://travis-ci.org/JuliaNLSolvers/NLSolversBase.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaNLSolvers/NLSolversBase.jl

[pkg-0.4-img]: http://pkg.julialang.org/badges/NLSolversBase_0.4.svg
[pkg-0.4-url]: http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.4
[pkg-0.5-img]: http://pkg.julialang.org/badges/NLSolversBase_0.5.svg
[pkg-0.5-url]: http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.5
[pkg-0.6-img]: http://pkg.julialang.org/badges/NLSolversBase_0.6.svg
[pkg-0.6-url]: http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.6

[cov-img]: http://codecov.io/github/JuliaNLSolvers/NLSolversBase.jl/coverage.svg?branch=master
[cov-url]: http://codecov.io/github/pkofod/NLSolversBase.jl?branch=master

[coveralls-img]: https://coveralls.io/repos/JuliaNLSolvers/NLSolversBase.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/JuliaNLSolvers/NLSolversBase.jl?branch=master