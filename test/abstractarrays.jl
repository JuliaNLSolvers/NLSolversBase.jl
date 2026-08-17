@testset "ComponentArrays" begin
    x_seed_1 = [0.0]
    x_seed_2 = [0.0]
    x_seed = ComponentArray(x_seed_1=x_seed_1, x_seed_2=x_seed_2)
    g_seed_1 = [0.0]
    g_seed_2 = [0.0]
    g_seed = ComponentArray(g_seed_1=g_seed_1, g_seed_2=g_seed_2)
    f_x_seed = 8157.682077608529

    nd = NonDifferentiable(exponential, x_seed)
    @test nd.f == exponential
    @test isnan(value(nd))
    @test iszero(nd.f_calls)
    od = OnceDifferentiable(exponential, exponential_gradient!, nothing, x_seed, 0.0, g_seed)
    @test od.f == exponential
    @test od.df == exponential_gradient!
    @test value(od) == 0.0
    @test iszero(od.f_calls)
    @test iszero(od.df_calls)
    @test od.DF isa ComponentArray
    @test od.x_f isa ComponentArray
    @test od.x_df isa ComponentArray
end
@testset "Matrix OnceDifferentiable" begin
    x_seed = fill(0.0, 1, 2)
    g_seed = fill(0.0, 1, 2)
    f_x_seed = 8157.682077608529

    nd = NonDifferentiable(exponential, x_seed)
    @test nd.f == exponential
    @test isnan(value(nd))
    @test iszero(nd.f_calls)
    od = OnceDifferentiable(exponential, exponential_gradient!, nothing, x_seed, 0.0, g_seed)
    @test od.f == exponential
    @test od.df == exponential_gradient!
    @test value(od) == 0.0
    @test iszero(od.f_calls)
    @test iszero(od.df_calls)
    @test od.DF isa Matrix
    @test od.x_f isa Matrix
    @test od.x_df isa Matrix
end
@testset "RecursiveArrays" begin
    x_seed_1 = [0.0]
    x_seed_2 = [0.0]
    x_seed = ArrayPartition(x_seed_1, x_seed_2)
    g_seed_1 = [0.0]
    g_seed_2 = [0.0]
    g_seed = ArrayPartition(g_seed_1, g_seed_2)
    f_x_seed = 8157.682077608529

    nd = NonDifferentiable(exponential, x_seed)
    @test nd.f == exponential
    @test isnan(value(nd))
    @test iszero(nd.f_calls)
    od = OnceDifferentiable(exponential, exponential_gradient!, nothing, x_seed, 0.0, g_seed)
    @test od.f == exponential
    @test od.df == exponential_gradient!
    @test value(od) == 0.0
    @test iszero(od.f_calls)
    @test iszero(od.df_calls)
    @test od.DF isa ArrayPartition
    @test od.x_f isa ArrayPartition
    @test od.x_df isa ArrayPartition
end

@testset "https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/172" begin
    # The x caches come from `similar`, which turns a ReinterpretArray into an Array. Differentiation is
    # prepared once at construction and reused, so it has to be prepared for the type the evaluations use.
    @testset for autodiff in (AutoFiniteDiff(; fdtype = Val(:central)), AutoForwardDiff())
        x_seed = reinterpret(Float64, [2.0 + 3.0im])
        @test !(x_seed isa Array)
        @test NLSolversBase.x_of_nans(x_seed) isa Array

        f!(F, x) = (F[1] = x[1]^2 - 2; F[2] = x[2]^2 - 3)
        od = OnceDifferentiable(f!, x_seed, copy(x_seed), autodiff)
        @test od.x_f isa Array
        value_jacobian!!(od, [2.0, 3.0])
        @test value(od) ≈ [2.0, 6.0]
        @test jacobian(od) ≈ [4.0 0.0; 0.0 6.0]

        g(x) = sum(abs2, x)
        od2 = OnceDifferentiable(g, x_seed, 0.0; autodiff = autodiff)
        value_gradient!!(od2, [2.0, 3.0])
        @test value(od2) ≈ 13.0
        @test gradient(od2) ≈ [4.0, 6.0]

        td = TwiceDifferentiable(g, x_seed, 0.0; autodiff = autodiff)
        value_gradient!!(td, [2.0, 3.0])
        hessian!!(td, [2.0, 3.0])
        @test gradient(td) ≈ [4.0, 6.0]
        @test hessian(td) ≈ [2.0 0.0; 0.0 2.0]
    end
end
