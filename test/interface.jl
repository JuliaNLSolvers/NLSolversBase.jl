@testset "interface" begin

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
    g_x_seed = [-218.39260013257694, -48618.50356545231]
    h_x_seed = [982.7667005965963 0.0; 0.0 307917.1892478646]

    x_alt = [1.0, 1.0]
    f_x_alt = 57.316431861603284
    g_x_alt = [-5.43656365691809, -218.39260013257694]
    h_x_alt = [16.30969097075427 0.; 0. 982.7667005965963]

    nd = NonDifferentiable(exponential, x_seed)
    od = OnceDifferentiable(exponential, exponential_gradient!, x_seed)
    td = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!, x_seed)

    @test value(nd) == value(od) == value(td) == f_x_seed
    @test gradient(od) == gradient(td) == g_x_seed
    @test hessian(td) == h_x_seed
    @test nd.f_calls == od.f_calls == td.f_calls == [1]
    @test od.g_calls == td.g_calls == [1]
    @test td.h_calls == [1]

    @test_throws ErrorException gradient!(nd, x_alt)
    gradient!(od, x_alt)
    gradient!(td, x_alt)

    @test value(nd) == value(od) == value(td) == f_x_seed
    @test gradient(td) == g_x_alt
    @test hessian(td) == h_x_seed
    @test nd.f_calls == od.f_calls == td.f_calls == [1]
    @test od.g_calls == td.g_calls == [2]
    @test td.h_calls == [1]

    @test_throws ErrorException hessian!(nd, x_alt)
    @test_throws ErrorException hessian!(od, x_alt)
    hessian!(td, x_alt)

    @test value(nd) == value(od) == value(td) == f_x_seed
    @test gradient(td) == g_x_alt
    @test hessian(td) == h_x_alt
    @test nd.f_calls == od.f_calls == td.f_calls == [1]
    @test od.g_calls == td.g_calls == [2]
    @test td.h_calls == [2]

    value!(nd, x_alt)
    value!(od, x_alt)
    value!(td, x_alt)
    @test value(nd) == value(od) == value(td) == f_x_alt
    @test gradient(td) == g_x_alt
    @test hessian(td) == h_x_alt
    @test nd.f_calls == od.f_calls == td.f_calls == [2]
    @test od.g_calls == td.g_calls == [2]
    @test td.h_calls == [2]

    @test_throws ErrorException value_gradient!(nd, x_seed)
    value_gradient!(od, x_seed)
    value_gradient!(td, x_seed)
    @test value(od) == value(td) == f_x_seed
    @test gradient(td) == g_x_seed
    @test hessian(td) == h_x_alt
    @test od.f_calls == td.f_calls == [3]
    @test od.g_calls == td.g_calls == [3]
    @test td.h_calls == [2]
end
