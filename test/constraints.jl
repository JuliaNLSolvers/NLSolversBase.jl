@testset "Constraints" begin
    @testset "ConstraintBounds" begin
        cb = ConstraintBounds([], [], [0.0], [0.0])
        @test NLSolversBase.nconstraints(cb) == 1
        @test NLSolversBase.nconstraints_x(cb) == 0
        @test cb.valc == [0.0]
        @test eltype(cb) == Float64

        cb = ConstraintBounds([0], [0], [], [])
        @test NLSolversBase.nconstraints(cb) == 0
        @test NLSolversBase.nconstraints_x(cb) == 1
        @test cb.valx == [0]
        @test eltype(cb) == Int
        io = IOBuffer()
        show(io, cb)
        s = String(take!(io))
        @static if VERSION > v"0.7.0-DEV.393" #picked arbitrarily, I do not know the correct one.
            @test s == "ConstraintBounds:\n  Variables:\n    NLSolversBase.UnquotedStringx[1]=0\n  Linear/nonlinear constraints:"
        else
            @test s == "ConstraintBounds:\n  Variables:\n    x[1]=0\n  Linear/nonlinear constraints:"
        end
        cb = ConstraintBounds([], [3.0], [0.0], [])
        @test NLSolversBase.nconstraints(cb) == 1
        @test NLSolversBase.nconstraints_x(cb) == 1
        @test cb.bx[1] == 3.0
        @test cb.σx[1] == -1
        @test cb.bc[1] == 0.0
        @test cb.σc[1] == 1
        @test eltype(cb) == Float64

        cb = ConstraintBounds([],[],[],[])
        @test eltype(cb) == Union{}
        @test eltype(convert(ConstraintBounds{Int}, cb)) == Int

        cb =  ConstraintBounds([1,2], [3,4.0], [], [10.,20.,30.])
        io = IOBuffer()
        show(io, cb)
        s = String(take!(io))
        @static if VERSION > v"0.7.0-DEV.393" #picked arbitrarily, I do not know the correct one.
            @test s == "ConstraintBounds:\n  Variables:\n    NLSolversBase.UnquotedStringx[1]≥1.0, x[1]≤3.0, x[2]≥2.0, x[2]≤4.0\n  Linear/nonlinear constraints:\n    NLSolversBase.UnquotedStringc_1≤10.0, c_2≤20.0, c_3≤30.0"
        else
            @test s == "ConstraintBounds:\n  Variables:\n    x[1]≥1.0, x[1]≤3.0, x[2]≥2.0, x[2]≤4.0\n  Linear/nonlinear constraints:\n    c_1≤10.0, c_2≤20.0, c_3≤30.0"
        end
    end

    @testset "Once differentiable constraints" begin
        lx, ux = (1.0,2.0)
        odc = OnceDifferentiableConstraints([lx], [ux])
        @test odc.bounds.bx == [lx, ux]
        @test isempty(odc.bounds.bc)

        prob = MVP.ConstrainedProblems.examples["HS9"]
        cbd = prob.constraintdata
        cb = ConstraintBounds(cbd.lx, cbd.ux, cbd.lc, cbd.uc)

        odc = OnceDifferentiableConstraints(cb)
        @test odc.bounds == cb

        odc = OnceDifferentiableConstraints(cbd.c!, cbd.jacobian!,
                                            cbd.lx, cbd.ux, cbd.lc, cbd.uc)
        @test isempty(odc.bounds.bx)
        @test isempty(odc.bounds.bc)
        @test odc.bounds.valc == [0.0]

        #TODO: add tests calling c! and jacobian!
    end

    @testset "Twice differentiable constraints" begin
        lx, ux = (1.0,2.0)
        odc = TwiceDifferentiableConstraints([lx], [ux])
        @test odc.bounds.bx == [lx, ux]
        @test isempty(odc.bounds.bc)

        prob = MVP.ConstrainedProblems.examples["HS9"]
        cbd = prob.constraintdata
        cb = ConstraintBounds(cbd.lx, cbd.ux, cbd.lc, cbd.uc)

        odc = TwiceDifferentiableConstraints(cb)
        @test odc.bounds == cb

        odc = TwiceDifferentiableConstraints(cbd.c!, cbd.jacobian!, cbd.h!,
                                            cbd.lx, cbd.ux, cbd.lc, cbd.uc)
        @test isempty(odc.bounds.bx)
        @test isempty(odc.bounds.bc)
        @test odc.bounds.valc == [0.0]

    end
end
