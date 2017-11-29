@testset "objective types" begin

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

    nd = NonDifferentiable(exponential, 0.0, x_seed)
    @test nd.f == exponential
    @test value(nd) == 0.0
    @test nd.f_calls == [0]
    od = OnceDifferentiable(exponential, exponential_gradient!, nothing, 0.0, g_seed, x_seed)
    @test od.f == exponential
    @test od.df == exponential_gradient!
    @test value(od) == 0.0
    @test od.f_calls == [0]
    @test od.df_calls == [0]

    td = TwiceDifferentiable(exponential, exponential_gradient!, nothing, exponential_hessian!, 0.0, h_seed, g_seed, x_seed)
    @test td.f == exponential
    @test td.df == exponential_gradient!
    @test value(td) == 0.0
    @test td.f_calls == [0]
    @test td.df_calls == [0]
    @test td.h_calls == [0]
end
