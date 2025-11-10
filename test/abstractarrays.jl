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
    @test value(nd) == 0.0
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
    @test value(nd) == 0.0
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
    @test value(nd) == 0.0
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
