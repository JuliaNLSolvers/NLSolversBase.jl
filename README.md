NLSolversBase.jl
========

Base functionality for optimization and solving systems of equations in Julia.

NLSolversBase.jl is the core, common dependency of several packages in the [JuliaNLSolvers](https://julianlsolvers.github.io) family.

The package aims at establishing common ground for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl), but [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) will eventually also depend on this package. The common ground is mainly the types used to hold objectives and information about the objectives, and an interface to interact with these types.

Travis-CI

[![Build Status](https://travis-ci.org/JuliaNLSolvers/NLSolversBase.jl.svg?branch=master)](https://travis-ci.org/JuliaNLSolvers/NLSolversBase.jl)

Package evaluator

[![pkg-0.4-img](http://pkg.julialang.org/badges/NLSolversBase_0.5.svg)](http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.5)
[![pkg-0.4-img](http://pkg.julialang.org/badges/NLSolversBase_0.6.svg)](http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.6)

Code coverage

[![Coverage Status](https://coveralls.io/repos/JuliaNLSolvers/NLSolversBase.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaNLSolvers/NLSolversBase.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaNLSolvers/NLSolversBase.jl/coverage.svg?branch=master)](http://codecov.io/github/pkofod/NLSolversBase.jl?branch=master)

# What
This package holds types for use in the JuliaNLSolvers family of packages. The types act as collections of callables relevant for optimization or solution of systems of equations (objective, (partial) derivatives of different orders), and cache variables needed to evaluate their values.

## NDifferentiable
There are currently three main types: `NonDifferentiable`, `OnceDifferentiable`, and `TwiceDifferentiable`. There's also a more experimental `TwiceDifferentiableHV` for optimization algorithms that use Hessian-vector products. An `NDifferentiable` instance can be used to hold relevant functions for

 - Optimization: ![Objective for optimization](https://user-images.githubusercontent.com/8431156/33996090-6224581c-e0e0-11e7-8737-5dd659745dcb.gif)
 - Solving systems of equations: ![Objective for systems of equations](https://user-images.githubusercontent.com/8431156/33996088-60760c4a-e0e0-11e7-96ca-470f2731f1c7.gif)

The words in front of `Differentiable` in the type names (`Non`, `Once`, `Twice`) are not meant to indicate and specific classification of the function as such, but more the requirement of the algorithms used.

### Examples
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
