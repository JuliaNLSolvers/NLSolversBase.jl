@testset "autodiff" begin
    # Should throw, as :wah is not a proper autodiff choice
    @test_throws ErrorException OnceDifferentiable(x->x, rand(10); autodiff=:wah)

    for T in (OnceDifferentiable, TwiceDifferentiable)
        odad1 = T(x->5.0, rand(1); autodiff = :finite)
        odad2 = T(x->5.0, rand(1); autodiff = :forward)
        gradient!(odad1, rand(1))
        gradient!(odad2, rand(1))
        #    odad3 = T(x->5., rand(1); autodiff = :reverse)
        @test gradient(odad1) == [0.0]
        @test gradient(odad2) == [0.0]
        #    @test odad3.g == [0.0]
    end

    for a in (1.0, 5.0)
        xa = rand(1)
        odad1 = OnceDifferentiable(x->a*x[1], xa; autodiff = :finite)
        odad2 = OnceDifferentiable(x->a*x[1], xa; autodiff = :forward)
    #    odad3 = OnceDifferentiable(x->a*x[1], xa; autodiff = :reverse)
        gradient!(odad1, xa)
        gradient!(odad2, xa)
        @test gradient(odad1) ≈ [a]
        @test gradient(odad2) == [a]
    #    @test odad3.g == [a]
    end
    for a in (1.0, 5.0)
        xa = rand(1)
        odad1 = OnceDifferentiable(x->a*x[1]^2, xa; autodiff = :finite)
        odad2 = OnceDifferentiable(x->a*x[1]^2, xa; autodiff = :forward)
    #    odad3 = OnceDifferentiable(x->a*x[1]^2, xa; autodiff = :reverse)
        gradient!(odad1, xa)
        gradient!(odad2, xa)
     @test gradient(odad1) ≈ 2.0*a*xa
        @test gradient(odad2) == 2.0*a*xa
    #    @test odad3.g == 2.0*a*xa
    end
    for dtype in (OnceDifferentiable, TwiceDifferentiable)
        for autodiff in (:finite, :forward)
            differentiable = dtype(x->sum(x), rand(2); autodiff = autodiff)
            value(differentiable)
            value!(differentiable, rand(2))
            value_gradient!(differentiable, rand(2))
            gradient!(differentiable, rand(2))
            dtype == TwiceDifferentiable && hessian!(differentiable, rand(2))
        end
    end
end
@testset "value/grad" begin
    a = 3.0
    x_seed = rand(1)
    odad1 = OnceDifferentiable(x->a*x[1]^2, x_seed)
    value_gradient!(odad1, x_seed)
    @test gradient(odad1) ≈ 2 .* a .* (x_seed)
    @testset "call counters" begin
        @test f_calls(odad1) == 1
        @test g_calls(odad1) == 1
        @test h_calls(odad1) == 0
        value_gradient!(odad1, x_seed .+ 1.0)
        @test f_calls(odad1) == 2
        @test g_calls(odad1) == 2
        @test h_calls(odad1) == 0
    end
    @test gradient(odad1) ≈ 2 .* a .* (x_seed .+ 1.0)
end
