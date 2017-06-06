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

    nd = NonDifferentiable(exponential, x_seed)
    @test nd.f == exponential
    @test value(nd) == f_x_seed
    @test nd.last_x_f == [0.0, 0.0]
    @test nd.f_calls == [1]
    od = OnceDifferentiable(exponential, exponential_gradient!, x_seed)
    @test od.f == exponential
    #@test od.g! == exponential_gradient!
    @test value(od) == f_x_seed
    @test od.last_x_f == [0.0, 0.0]
    @test od.f_calls == [1]
    @test od.g_calls == [1]

    td = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!, x_seed)
    @test td.f == exponential
    #@test td.g! == exponential_gradient!
    @test value(td) == f_x_seed
    @test td.last_x_f == [0.0, 0.0]
    @test td.f_calls == [1]
    @test td.g_calls == [1]
    @test td.h_calls == [1]
end

@testset "uninitialized objectives" begin
    # Test example
    function exponential(x::Vector)
        return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
    end

    function exponential_gradient!(storage::Vector, x::Vector)
        storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    end
    function exponential_fg!(storage, x)
        exponential_gradient!(storage, x)
        exponential(x)
    end
    function exponential_hessian!(storage::Matrix, x::Vector)
        storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = 2.0 * exp((3.0 - x[1])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
    end

    x_seed = [0.0, 0.0]
    f_x_seed = 8157.682077608529

    und = NonDifferentiable(exponential)
    uod1 = OnceDifferentiable(exponential, exponential_gradient!)
    uod2 = OnceDifferentiable(exponential, exponential_gradient!, exponential_fg!)
    utd1 = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!)
    utd2 = TwiceDifferentiable(exponential, exponential_gradient!, exponential_fg!, exponential_hessian!)
    nd = NonDifferentiable(und, x_seed)
    od1 = OnceDifferentiable(uod1, x_seed)
    od2 = OnceDifferentiable(uod2, x_seed)
    # TwiceDifferentiable(utd1, x_seed)
    td1 = TwiceDifferentiable(utd2, x_seed)
    td2 = TwiceDifferentiable(utd1, x_seed)
end
