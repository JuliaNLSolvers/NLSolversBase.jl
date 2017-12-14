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
There are currently three main types: NonDifferentiable, OnceDifferentiable, and TwiceDifferentiable. There's also a more experimental TwiceDifferentiableHV for optimization algorithms that use Hessian-vector products. An NDifferentiable instance can be used to hold relevant functions for

 - Optimization: ![Objective for optimization](https://user-images.githubusercontent.com/8431156/33996090-6224581c-e0e0-11e7-8737-5dd659745dcb.gif)
 - Solving systems of equations: ![Objective for systems of equations](https://user-images.githubusercontent.com/8431156/33996088-60760c4a-e0e0-11e7-96ca-470f2731f1c7.gif)

### Examples
Say we want to minimize the Hosaki test function

![Hosaki test function](https://user-images.githubusercontent.com/8431156/33995927-c5b9f950-e0df-11e7-8760-9ba792c2b331.gif)

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
f(x) = 
g!(G, x) = 
fg!(G, x) = 
h!(H, x) = 
```
We can then evaluate the function at various places using the `value`/`value!` functions.