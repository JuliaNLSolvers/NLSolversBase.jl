@testset "objective types" begin
    # TODO: Use OptimTestProblems
    # TODO: MultivariateProblems.UnconstrainedProblems.exampples["Exponential"]

    # Test example
    function exponential(x::Vector)
        return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
    end

    function exponential_gradient!(storage::Vector, x::Vector)
        storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    end

    function exponential_hessian!(storage::Matrix, x::Vector)
        storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = 2.0 * exp((3.0 - x[1])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
    end

    x_seed = [0.0, 0.0]
    g_seed = [0.0, 0.0]
    h_seed = [0.0 0.0; 0.0 0.0]
    f_x_seed = 8157.682077608529

    nd = NonDifferentiable(exponential, x_seed)
    @test nd.f == exponential
    @test value(nd) == 0.0
    @test nd.f_calls == [0]
    od = OnceDifferentiable(exponential, exponential_gradient!, nothing, x_seed, 0.0, g_seed)
    @test od.f == exponential
    @test od.df == exponential_gradient!
    @test value(od) == 0.0
    @test od.f_calls == [0]
    @test od.df_calls == [0]

    td = TwiceDifferentiable(exponential, exponential_gradient!, nothing, exponential_hessian!, x_seed, 0.0, g_seed, h_seed)
    @test td.f == exponential
    @test td.df == exponential_gradient!
    @test value(td) == 0.0
    @test td.f_calls == [0]
    @test td.df_calls == [0]
    @test td.h_calls == [0]

    @testset "no fg!" begin
        srand(324)
        od = OnceDifferentiable(exponential, exponential_gradient!, x_seed, 0.0, g_seed)
        xrand = rand(2)
        value_gradient!(od, xrand)
        fcache = value(od)
        gcache = copy(gradient(od))
        value_gradient!(od, zeros(2))
        gradient!(od, xrand)
        @test value(od, zeros(2)) == od.F
        @test value(od, zeros(2)) == value(od)
        @test gradient(od) == gcache

        od = OnceDifferentiable(exponential, exponential_gradient!, x_seed)
        xrand = rand(2)
        value_gradient!(od, xrand)
        fcache = value(od)
        gcache = copy(gradient(od))
        value_gradient!(od, zeros(2))
        gradient!(od, xrand)
        @test value(od, zeros(2)) == od.F
        @test value(od, zeros(2)) == value(od)
        @test gradient(od) == gcache

        td = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!, x_seed, 0.0, g_seed)
        xrand = rand(2)
        value_gradient!(td, xrand)
        fcache = value(td)
        gcache = copy(gradient(td))
        value_gradient!(td, zeros(2))
        gradient!(td, xrand)
        @test value(td, zeros(2)) == td.F
        @test value(td, zeros(2)) == value(td)
        @test gradient(td) == gcache
    end
end
