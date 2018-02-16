@testset "Constraints" begin
    cb = ConstraintBounds([], [], [0.0], [0.0])
    @test NLSolversBase.nconstraints(cb) == 1
    @test NLSolversBase.nconstraints_x(cb) == 0
    @test eltype(cb) == Float64

    cb = ConstraintBounds([0], [0], [], [])
    @test NLSolversBase.nconstraints(cb) == 0
    @test NLSolversBase.nconstraints_x(cb) == 1
    @test eltype(cb) == Int

    cb = ConstraintBounds([], [3.0], [0.0], [])
    @test NLSolversBase.nconstraints(cb) == 1
    @test NLSolversBase.nconstraints_x(cb) == 1
    @test cb.bx[1] == 3.0
    @test cb.σx[1] == -1
    @test cb.bc[1] == 0.0
    @test cb.σc[1] == 1
    @test eltype(cb) == Float64

    cb =  ConstraintBounds([1,2], [3,4.0], [], [10.,20.,30.])
    io = IOBuffer()
    show(io, cb)
    s = String(take!(io))

    @test s == "ConstraintBounds:\n  Variables:\n    x[1]≥1.0, x[1]≤3.0, x[2]≥2.0, x[2]≤4.0\n  Linear/nonlinear constraints:\n    c_1≤10.0, c_2≤20.0, c_3≤30.0"
end
