# NLSolversBase

NLSolversBase is the core, common dependency of several [JuliaNLSolvers](https://github.com/JuliaNLSolvers) packages. Currently, it aims at establishing common ground for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) and [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl), but [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) will eventually also depend on this package. The common ground is mainly the types used to hold objectives and information about the objectives, and an interface to interact with these types.

Travis-CI

[![Build Status](https://travis-ci.org/JuliaNLSolvers/NLSolversBase.jl.svg?branch=master)](https://travis-ci.org/pkofod/NLSolversBase.jl)

Package evaluator

[![pkg-0.4-img](http://pkg.julialang.org/badges/NLSolversBase_0.5.svg)](http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.5)
[![pkg-0.4-img](http://pkg.julialang.org/badges/NLSolversBase_0.6.svg)](http://pkg.julialang.org/?pkg=NLSolversBase&ver=0.6)

Code coverage

[![Coverage Status](https://coveralls.io/repos/JuliaNLSolvers/NLSolversBase.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/pkofod/NLSolversBase.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaNLSolvers/NLSolversBase.jl/coverage.svg?branch=master)](http://codecov.io/github/pkofod/NLSolversBase.jl?branch=master)
