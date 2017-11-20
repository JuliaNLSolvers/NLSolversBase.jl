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
    f_x_seed = 8157.682077608529

    nd = NonDifferentiable(exponential)
    ndc = NonDifferentiableCache(exponential, x_seed)
    @test nd.f == exponential
    @test value(ndc) == f_x_seed
    @test ndc.last_x_f == [0.0, 0.0]
    @test ndc.f_calls == [1]
    od = OnceDifferentiable(exponential, exponential_gradient!)
    odc = OnceDifferentiableCache(exponential, exponential_gradient!, x_seed)
    @test od.f == exponential
    @test od.g! == exponential_gradient!
    @test value(odc) == f_x_seed
    @test odc.last_x_f == [0.0, 0.0]
    @test odc.f_calls == [1]
    @test odc.g_calls == [1]

    td = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!)
    tdc = TwiceDifferentiableCache(exponential, exponential_gradient!, exponential_hessian!, x_seed)
    @test td.f == exponential
    @test td.g! == exponential_gradient!
    @test value(tdc) == f_x_seed
    @test tdc.last_x_f == [0.0, 0.0]
    @test tdc.f_calls == [1]
    @test tdc.g_calls == [1]
    @test tdc.h_calls == [1]
end
