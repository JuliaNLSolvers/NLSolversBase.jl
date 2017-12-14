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

 - Optimization................... F:R^N -> R
 - Solving systems of equations... F:R^N -> R^N

### Examples
Say we want to minimize the Hosaki test function

![Hosaki test function][https://user-images.githubusercontent.com/8431156/33995927-c5b9f950-e0df-11e7-8760-9ba792c2b331.gif]

The relevant functions are coded in Julia as
```julia
f(x) = 
g!(G, x) = 
fg!(G, x) = 
h!(H, x) = 
```
We can then evaluate the function at various places using the `value`/`value!` functions.