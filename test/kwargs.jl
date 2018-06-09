@testset "inplace" begin
    # R^N → R
    f1 = OnceDifferentiable(exponential, rand(2), 0.0)
    fia1 = OnceDifferentiable(exponential, rand(2); inplace = false)
    fi1 = OnceDifferentiable(exponential, exponential_gradient, rand(2); inplace = false)
    fi2 = OnceDifferentiable(exponential, exponential_gradient, rand(2); inplace = false)
    fi3 = OnceDifferentiable(exponential, exponential_gradient, exponential_value_gradient,
                       rand(2); inplace = false)
    xr = rand(2)
    @test value!(f1, xr) ≈ value!(fia1, xr) ≈ value!(fi1, xr) ≈ value!(fi2, xr) ≈ value!(fi3, xr)
    @test gradient!(f1, xr) ≈ gradient!(fia1, xr)
    @test gradient!(fia1, xr) ≈ gradient!(fi1, xr) ≈ gradient!(fi2, xr) ≈ gradient!(fi3, xr)
    vg1 = value_gradient!(fia1, xr)
    vg2 = value_gradient!(fi1, xr)
    vg3 = value_gradient!(fi2, xr)
    vg4 = value_gradient!(fi3, xr)
    @test vg1[1] ≈ vg2[1] ≈ vg3[1] ≈ vg4[1]
    @test vg1[2] ≈ vg2[2] ≈ vg3[2] ≈ vg4[2]
    # R^N → R^N
    f1 = OnceDifferentiable(exponential_gradient!, rand(2), rand(2))
    fia1 = OnceDifferentiable(exponential_gradient, rand(2), rand(2); inplace = false)
    fia2 = OnceDifferentiable(exponential_gradient, rand(2), rand(2); inplace = false, autodiff = :forward)
    fi1 = OnceDifferentiable(exponential_gradient, exponential_hessian, rand(2), rand(2); inplace = false)
    fi2 = OnceDifferentiable(exponential_gradient, exponential_hessian, rand(2), rand(2); inplace = false)
    fi3 = OnceDifferentiable(exponential_gradient, exponential_hessian, exponential_gradient_hessian,
    rand(2), rand(2); inplace = false)
    xr = fill(2.0, 2)
    @test value!(f1, xr) ≈ value!(fia1, xr) ≈ value!(fia2, xr) ≈ value!(fi1, xr) ≈ value!(fi2, xr) ≈ value!(fi3, xr)
    @test jacobian!(f1, xr) ≈ jacobian!(fia1, xr) ≈ jacobian!(fia2, xr) ≈ jacobian!(fi1, xr) ≈ jacobian!(fi2, xr) ≈ jacobian!(fi3, xr)
    # Finite doesn't cut it here, so had to split it up
    xr = [1.0, 3.4]
    @test value!(f1, xr) ≈ value!(fia1, xr) ≈ value!(fi1, xr) ≈ value!(fi2, xr) ≈ value!(fi3, xr)
    # need to split this up as the example is hard to approximate with finite differences
    @test jacobian!(f1, xr) ≈ jacobian!(fia1, xr) ≈ jacobian!(fia2, xr)
    @test jacobian!(fi1, xr) ≈ jacobian!(fi2, xr) ≈ jacobian!(fi3, xr)
end

@testset "autodiff" begin
    # base line api
    f1 = OnceDifferentiable(exponential, exponential_gradient!, exponential_value_gradient!,
                            rand(2), 0.0, rand(2))
    # default autodiff R^N → R
    fa1 = OnceDifferentiable(exponential, rand(2))
    # specific autodiff R^N → R
    fa2 = OnceDifferentiable(exponential, rand(2); autodiff = :finite)
    fa3 = OnceDifferentiable(exponential, rand(2); autodiff = :forward)
    # random input
    xr = rand(2)
    # test that values are the same approximately
    @test value!(f1, xr) ≈ value!(fa1, xr) ≈ value!(fa2, xr) ≈ value!(fa3, xr)
    @test gradient!(f1, xr) ≈ gradient!(fa1, xr) ≈ gradient!(fa2, xr) ≈ gradient!(fa3, xr)
end

@testset "inplace/autodiff" begin
    # R^N → R^N
    fa1 = OnceDifferentiable(exponential_gradient!, rand(2), rand(2); inplace = true)
    fia1 = OnceDifferentiable(exponential_gradient, rand(2), rand(2); inplace = false)
    fia2 = OnceDifferentiable(exponential_gradient, rand(2), rand(2); inplace = false, autodiff = :finite)
    fia3 = OnceDifferentiable(exponential_gradient, rand(2), rand(2); inplace = false, autodiff = :forward)
    xr = rand(2)
    @test value!(fa1, xr) ≈ value!(fia1, xr) ≈ value!(fia2, xr) ≈ value!(fia3, xr)
    @test jacobian!(fa1, xr) ≈ jacobian!(fia1, xr) ≈ jacobian!(fia2, xr) ≈ jacobian!(fia3, xr)
end
