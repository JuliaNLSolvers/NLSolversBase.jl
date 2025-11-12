@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(NLSolversBase)
    end

    @testset "ExplicitImports" begin
        # No implicit imports (`using XY`)
        @test ExplicitImports.check_no_implicit_imports(NLSolversBase) === nothing

        # All explicit imports (`using XY: Z`) are loaded via their owners
        @test ExplicitImports.check_all_explicit_imports_via_owners(NLSolversBase) === nothing

        # No explicit imports (`using XY: Z`) that are not used
        @test ExplicitImports.check_no_stale_explicit_imports(NLSolversBase) === nothing

        # Nothing is accessed via modules other than its owner
        @test ExplicitImports.check_all_qualified_accesses_via_owners(NLSolversBase) === nothing

        # NLSolversBase accesses almost no non-public names
        # The only exception is `ForwardDiff.Chunk`
        @test ExplicitImports.check_all_qualified_accesses_are_public(NLSolversBase; ignore = (:Chunk,)) === nothing

        # No self-qualified accesses
        @test ExplicitImports.check_no_self_qualified_accesses(NLSolversBase) === nothing
    end

    @testset "JET" begin
        # Check that there are no undefined global references and undefined field accesses
        JET.test_package(NLSolversBase; target_defined_modules = true, mode = :typo, toplevel_logger = nothing)

        # Analyze methods based on their declared signature
        JET.test_package(NLSolversBase; target_defined_modules = true, toplevel_logger = nothing)
    end
end
