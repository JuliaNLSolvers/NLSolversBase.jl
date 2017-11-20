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

    nd = NonDifferentiable(exponential)
    ndc = NonDifferentiableCache(exponential, x_seed)
    od = OnceDifferentiable(exponential, exponential_gradient!)
    odc = OnceDifferentiableCache(exponential, exponential_gradient!, x_seed)
    td = TwiceDifferentiable(exponential, exponential_gradient!, exponential_hessian!)
    tdc = TwiceDifferentiableCache(exponential, exponential_gradient!, exponential_hessian!, x_seed)

    @test value(ndc) == value(odc) == value(tdc) == f_x_seed
    @test value(ndc, nd, x_seed) == value(odc, od, x_seed) == value(tdc, td, x_seed)
    # change last_x_f manually to test branch
    ndc.last_x_f .*= 0
    odc.last_x_f .*= 0
    tdc.last_x_f .*= 0
    @test value(ndc, nd, x_seed) == value(odc, od, x_seed) == value(tdc, td, x_seed)
    @test gradient(odc) == gradient(tdc) == g_x_seed
    @test hessian(tdc) == h_x_seed
    @test ndc.f_calls == odc.f_calls == tdc.f_calls == [1]
    @test odc.g_calls == tdc.g_calls == [1]
    @test tdc.h_calls == [1]

    # can't update gradient for NonDifferentiable
    @test_throws ErrorException gradient!(ndc, nd, x_alt)
    gradient!(odc, od, x_alt)
    gradient!(tdc, td, x_alt)

    @test value(ndc) == value(odc) == value(tdc) == f_x_seed
    @test gradient(tdc) == g_x_alt
    @test gradient(tdc) == [gradient(tdc, i) for i = 1:length(x_seed)]
    @test hessian(tdc) == h_x_seed
    @test ndc.f_calls == odc.f_calls == tdc.f_calls == [1]
    @test odc.g_calls == tdc.g_calls == [2]
    @test tdc.h_calls == [1]

    @test_throws ErrorException hessian!(ndc, nd, x_alt)
    @test_throws ErrorException hessian!(odc, od, x_alt)
    hessian!(tdc, td, x_alt)

    @test value(ndc) == value(odc) == value(tdc) == f_x_seed
    @test gradient(tdc) == g_x_alt
    @test hessian(tdc) == h_x_alt
    @test ndc.f_calls == odc.f_calls == tdc.f_calls == [1]
    @test odc.g_calls == tdc.g_calls == [2]
    @test tdc.h_calls == [2]

    value!(ndc, nd, x_alt)
    value!(odc, od, x_alt)
    value!(tdc, td, x_alt)
    @test value(ndc) == value(odc) == value(tdc) == f_x_alt
    @test gradient(tdc) == g_x_alt
    @test hessian(tdc) == h_x_alt
    @test ndc.f_calls == odc.f_calls == tdc.f_calls == [2]
    @test odc.g_calls == tdc.g_calls == [2]
    @test tdc.h_calls == [2]

    @test_throws ErrorException value_gradient!(ndc, nd, x_seed)
    value_gradient!(odc, od, x_seed)
    value_gradient!(tdc, td, x_seed)
    @test value(odc) == value(tdc) == f_x_seed
    # change last_x_f manually to test branch
    odc.last_x_f .*= 0
    tdc.last_x_f .*= 0
    value_gradient!(odc, od, x_seed)
    value_gradient!(tdc, td, x_seed)
    @test value(odc) == value(tdc) == f_x_seed
    # change last_x_g manually to test branch
    odc.last_x_g .*= 0
    tdc.last_x_g .*= 0
    value_gradient!(odc, od, x_seed)
    value_gradient!(tdc, td, x_seed)
    @test value(odc) == value(tdc) == f_x_seed
    @test gradient(tdc) == g_x_seed
    @test hessian(tdc) == h_x_alt
    @test odc.f_calls == tdc.f_calls == [3]
    @test odc.g_calls == tdc.g_calls == [3]
    @test tdc.h_calls == [2]
end
