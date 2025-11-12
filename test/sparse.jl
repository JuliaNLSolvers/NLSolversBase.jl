@testset "sparse" begin
    @testset "𝐑ⁿ → 𝐑" begin
        f(x) = sum(x -> x^2, x)
        g(G, x) = G .= 2 .* x
        h(H, _) = coptyo!(H, 2 * I)

        obj_dense = TwiceDifferentiable(f, g, h, rand(40))
        @test !issparse(obj_dense.H)

        obj_sparse = TwiceDifferentiable(f, g, h, rand(40), 0.0, rand(40), sparse(1.0I, 40, 40))
        @test obj_sparse.H isa SparseMatrixCSC

        function fgh!(F, G, H, x)
            if G !== nothing
                G .= 2 .* x
            end
            if H !== nothing
                copyto!(H, 2 * I)
            end
            if F === nothing
                return nothing
            else
                return sum(x -> x^2, x)
            end
        end

        obj_fgh = TwiceDifferentiable(NLSolversBase.only_fgh!(fgh!), rand(40), 0.0, rand(40), sparse(1.0I, 40, 40))
        @test obj_fgh.H isa SparseMatrixCSC
    end
    @testset "𝐑ⁿ → 𝐑ⁿ" begin
        f(F, x) = F .= 2 .* x
        j(J, _) = copyto!(J, 2.0 * I)

        # Test that with no spec on the Jacobian cache it is dense
        obj_dense = OnceDifferentiable(f, j, rand(40), rand(40))
        @test !issparse(obj_dense.DF)

        obj_dense = OnceDifferentiable(f, j, rand(40), rand(40), rand(40))
        @test !issparse(obj_dense.DF)


        obj_sparse = OnceDifferentiable(f, j, rand(40), rand(40), sparse(1.0I, 40, 40))
        @test obj_sparse.DF isa SparseMatrixCSC

        function fj!(F, J, x)
            if F !== nothing
                F .= 2 .* x
            end
            if J !== nothing
                copyto!(J, 2 * I)
            end
            return F
        end

        obj_fj = OnceDifferentiable(NLSolversBase.only_fj!(fj!), rand(40), rand(40), sparse(1.0I, 40, 40))
        @test obj_fj.DF isa SparseMatrixCSC
    end
end
