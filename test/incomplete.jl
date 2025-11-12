@testset "incomplete objectives" begin
    import NLSolversBase: make_f, make_df, make_fdf
    function f(x)
        sum(x->x^2,x)
    end

    g!(G, x) = G .= 2 .* x
    g(x) = 2 .* x

    h!(H, _) = copyto!(H, 2*I)
    h(x) = Diagonal(fill(2, length(x)))

    function fg!(G, x)
        G .= 2 .* x
        sum(x->x^2,x)
    end
    function just_fg!(F, G, x)
        if G !== nothing
            G .= 2 .* x
        end
        if F === nothing
            return nothing
        else
            return sum(x->x^2,x)
        end
    end
    fg(x) = f(x), g(x)
    function just_fgh!(F, G, H, x)
        if H !== nothing
            copyto!(H, 2*I)
        end
        if G !== nothing
            G .= 2 .* x
        end
        if F === nothing
            return nothing
        else
            return sum(x->x^2,x)
        end
    end
    fgh(x) = f(x), g(x), h(x)
    fdf!_real = only_fg!(just_fg!)
    fdf_real = only_fg(fg)

    function just_hv!(Hv, x, v)
        Hv .= 2 .* v
    end
    function just_fghv!(F, G, Hv, x, v)
        if G  !== nothing
            G .= 2 .* x
        end
        if Hv !== nothing
            Hv .= 2 .* v
        end
        if F === nothing
            return nothing
        else
            return sum(x->x^2, x)
        end
    end

    df_fdf_real = only_g_and_fg(g, fg)
    Random.seed!(3259)
    x = rand(10)
    G_cache = similar(x)
    G_cache2 = similar(G_cache)

    @test fdf_real.df === nothing
    @test df_fdf_real.df === g

    @test fdf!_real.fdf === just_fg!
    @test fdf_real.fdf === fg
    @test df_fdf_real.df === g
    @test df_fdf_real.fdf === fg

    for FDF in (fdf_real, fdf!_real)
        @test make_f(FDF, x, x[1])(x) == f(x)
        make_df(FDF, x, x[1])(G_cache, x)
        g!(G_cache2, x)
        @test G_cache == G_cache2
        f1 = make_fdf(FDF, x, x[1])(G_cache, x.*2)
        f2 = fg!(G_cache2, x.*2)
        @test G_cache == G_cache2
        @test f1 == f2
    end

    # Only gradient/Jacobian
    df!_real = make_df(NLSolversBase.NotInplaceObjective(; df = g), x, x[1])
    df!_real(fill!(G_cache, NaN), x)
    g!(G_cache2, x)
    @test G_cache == G_cache2
    jac!_array = make_df(NLSolversBase.NotInplaceObjective(; df = x -> Diagonal(fill!(similar(x), 1))), x, x)
    J = fill!(similar(x, length(x), length(x)), NaN)
    jac!_array(J, x)
    @test J == Diagonal(ones(length(x)))

    # Only function value, gradient, and objective 
    df!_real = make_df(only_fghv!(just_fghv!), x, x[1])
    df!_real(fill!(G_cache, NaN), x)
    g!(G_cache2, x)
    @test G_cache == G_cache2

    nd_fg = NonDifferentiable(only_fg(fg), x)
    nd_fg! = NonDifferentiable(only_fg!(just_fg!), x)
    for ND in (nd_fg, nd_fg!)
        value!(ND, x)
        value(ND) == f(x)
    end
    od_fg = OnceDifferentiable(only_fg(fg), x)
    od_fg! = OnceDifferentiable(only_fg!(just_fg!), x)
    od_fgh = OnceDifferentiable(NLSolversBase.NotInplaceObjective(; fgh), x)
    od_fgh! = TwiceDifferentiable(only_fgh!(just_fgh!), x)
    _F = zero(eltype(x))
    od_fgh! = TwiceDifferentiable(only_fgh!(just_fgh!), x, _F)
    od_fgh! = TwiceDifferentiable(only_fgh!(just_fgh!), x, _F, similar(x))
    od_fgh! = TwiceDifferentiable(only_fgh!(just_fgh!), x, _F, similar(x), NLSolversBase.alloc_H(x, _F))
    for OD in (od_fg, od_fg!, od_fgh, od_fgh!)
        value!(OD, x)
        @test value(OD) == f(x)
        gradient!(OD, x)
        @test gradient(OD) == g(x)
        @test gradient(OD, x) == g(x)
        value_gradient!(OD, 2 .* x)
        @test value(OD) == f(2 .* x)
        @test gradient(OD) == g(2 .* x)
    end

    # Incomplete TwiceDifferentiable with Hessian-vector product
    v = randn(10)
    od_fg_and_hv = TwiceDifferentiable(only_fg_and_hv!(just_fg!, just_hv!), x)
    od_fghv      = TwiceDifferentiable(only_fghv!(just_fghv!), x)
    ndtdhv = NonDifferentiable(od_fghv, v)
    @test value(ndtdhv, v) === value(od_fghv, v)

    for OD in (od_fg_and_hv, od_fghv)
        gradient!(OD, x)
        @test gradient(OD) == g(x)
        hv_product!(OD, x, v)
        @test OD.Hv == 2v
        OD.f(x) == f(x)
        _g = similar(x)
        OD.fdf(_g, x)
        @test _g == g(x)
        @test OD.fdf(_g, x) == f(x)
    end
end
@testset "incomplete objectives vectors" begin
    function tf(x)
        x.^2
    end
    function tf(F, x)
        copyto!(F, tf(x))
    end

    tj!(J, x) = copyto!(J, Diagonal(x))
    tj(x) = Matrix(Diagonal(x))

    function tfj!(F, J, x)
        copyto!(J, Diagonal(x))
        F .= x.^2
    end
    function just_tfj!(F, J, x)
        if J !== nothing
            copyto!(J, Diagonal(x))
        end
        if F === nothing
            return nothing
        else
            F .= x.^2
            return F
        end
    end
    tfj(x) = tf(x), tj(x)

    fdf!_real = only_fj!(just_tfj!)
    fdf_real = only_fj(tfj)

    df_fdf_real = only_j_and_fj(tj, tfj)
    Random.seed!(3259)
    x = rand(10)
    J_cache = similar(Matrix(Diagonal(x)))
    J_cache2 = similar(Matrix(Diagonal(x)))
    F_cache = similar(x)
    F_cache2 = similar(x)

    @test fdf_real.df === nothing
    @test df_fdf_real.df === tj

    @test fdf!_real.fdf === just_tfj!
    @test fdf_real.fdf === tfj
    @test df_fdf_real.df === tj
    @test df_fdf_real.fdf == tfj

    for FDF in (fdf_real, fdf!_real)
        @test make_f(FDF, x, x)(F_cache, x) == tf(x)
        make_df(FDF, x, x)(J_cache, x)
        tj!(J_cache2, x)
        @test J_cache == J_cache2
        make_fdf(FDF, x, x)(F_cache, J_cache, x.*2)
        tfj!(F_cache2, J_cache2, x.*2)
        @test F_cache == F_cache2
        @test J_cache == J_cache2
    end

    nd_fj = NonDifferentiable(only_fj(tfj), x, x)
    nd_fj! = NonDifferentiable(only_fj!(just_tfj!), x, x)
    for ND in (nd_fj, nd_fj!)
        value!(ND, x)
        value(ND) == tf(x)
    end
    od_fj = OnceDifferentiable(only_fj(tfj), x, x)
    od_fj! = OnceDifferentiable(only_fj!(just_tfj!), x, x)
    for OD in (od_fj, od_fj!)
        value!(OD, x)
        @test value(OD) == tf(x)
        jacobian!(OD, x)
        @test jacobian(OD) == tj(x)
        @test jacobian(OD, x) == tj(x)
        value_jacobian!(OD, 2 .* x)
        @test value(OD) == tf(2 .* x)
        @test jacobian(OD) == tj(2 .* x)
    end

end

@testset "https://github.com/JuliaNLSolvers/Optim.jl/issues/718" begin
    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    function g!(G, x)
      G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
      G[2] = 200.0 * (x[2] - x[1]^2)
    end
    function h!(H, x)
      H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
      H[1, 2] = -400.0 * x[1]
      H[2, 1] = -400.0 * x[1]
      H[2, 2] = 200.0
    end

    function fg!(F,G,x)
      if G !== nothing
        g!(G,x)
      end
      if F === nothing
        return nothing
      else
        return f(x)
      end
    end
    function fgh!(F,G,H,x)
      if G !== nothing
        g!(G,x)
      end
      if H !== nothing
        h!(H,x)
      end
      if F === nothing
        return nothing
      else
        return f(x)
      end
    end
    
    gx = [0.0,0.0]
    x=[0.0,0.0]

    @test NLSolversBase.make_f(only_fgh!(fgh!),[0.0,0.0],0.0)(x) == 1.0
    @test NLSolversBase.make_df(only_fgh!(fgh!),[0.0,0.0],0.0)(gx, x) === nothing
    @test gx == [-2.0, 0.0]

    gx = [0.0,0.0]
    @test NLSolversBase.make_fdf(only_fgh!(fgh!),[0.0,0.0],0.0)(gx, x) == 1.0
    @test gx == [-2.0, 0.0]

end

@testset "Error messages in case of missing functions" begin
    x = [1.0, 2.0]

    @testset "Inplace objectives" begin
        # Evaluation of objective function not possible
        f = make_f(NLSolversBase.InplaceObjective(; hv = (Hv, x, v) -> fill!(Hv, 0)), x, x[1])
        @test_throws ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.") f(x)
        f! = make_f(NLSolversBase.InplaceObjective(), x, x)
        @test_throws ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.") f!(similar(x), x)

        # Evaluation of gradient/Jacobian not possible
        df! = make_df(NLSolversBase.InplaceObjective(; hv = (Hv, x, v) -> fill!(Hv, 0)), x, x[1])
        @test_throws ArgumentError("Cannot evaluate the gradient of the objective function: No suitable Julia function available.") df!(similar(x), x)
        jac! = make_df(NLSolversBase.InplaceObjective(), x, x)
        @test_throws ArgumentError("Cannot evaluate the Jacobian of the objective function: No suitable Julia function available.") jac!(similar(x, length(x), length(x)), x)

        # Combined evaluation of objective function + its gradient/Jacobian not possible
        fdf! = make_fdf(NLSolversBase.InplaceObjective(; hv = (Hv, x, v) -> fill!(Hv, 0)), x, x[1])
        @test_throws ArgumentError("Cannot evaluate the objective function and its gradient: No suitable Julia function available.") fdf!(similar(x), x)
        fjac! = make_fdf(NLSolversBase.InplaceObjective(), x, x)
        @test_throws ArgumentError("Cannot evaluate the objective function and its Jacobian: No suitable Julia function available.") fjac!(similar(x), similar(x, length(x), length(x)), x)

        # Combined evaluation of gradient and Hessian
        dfh!_1 = NLSolversBase.make_dfh(NLSolversBase.InplaceObjective(; fdf = (DF, x) -> (fill!(DF, 1); sum(x))), x, x[1])
        dfh!_2 = NLSolversBase.make_dfh(NLSolversBase.InplaceObjective(; fghv = (DF, Hv, x, v) -> (fill!(DF, 1); copyto!(Hv, 0); sum(x))), x, x[1])
        dfh!_3 = NLSolversBase.make_dfh(NLSolversBase.InplaceObjective(; hv = (Hv, x, v) -> fill!(Hv, 0)), x, x[1])
        for dfh! in (dfh!_1, dfh!_2, dfh!_3)
            @test_throws ArgumentError("Cannot evaluate the gradient and Hessian of the objective function: No suitable Julia function available.") dfh!(similar(x), similar(x, length(x), length(x)), x[1])
        end

        # Combined evaluation of objective function, gradient and Hessian
        fdfh!_1 = NLSolversBase.make_fdfh(NLSolversBase.InplaceObjective(; fdf = (DF, x) -> (fill!(DF, 1); sum(x))), x, x[1])
        fdfh!_2 = NLSolversBase.make_fdfh(NLSolversBase.InplaceObjective(; fghv = (DF, Hv, x, v) -> (fill!(DF, 1); copyto!(Hv, 0); sum(x))), x, x[1])
        fdfh!_3 = NLSolversBase.make_fdfh(NLSolversBase.InplaceObjective(; hv = (Hv, x, v) -> fill!(Hv, 0)), x, x[1])
        for fdfh! in (fdfh!_1, fdfh!_2, fdfh!_3)
            @test_throws ArgumentError("Cannot evaluate the objective function, its gradient and its Hessian: No suitable Julia function available.") fdfh!(similar(x), similar(x, length(x), length(x)), x[1])
        end

        # Evaluation of Hessian
        h!_1 = NLSolversBase.make_h(NLSolversBase.InplaceObjective(; fdf = (DF, x) -> (fill!(DF, 1); sum(x))), x, x[1])
        h!_2 = NLSolversBase.make_h(NLSolversBase.InplaceObjective(; fghv = (DF, Hv, x, v) -> (fill!(DF, 1); copyto!(Hv, 0); sum(x))), x, x[1])
        h!_3 = NLSolversBase.make_h(NLSolversBase.InplaceObjective(; hv = (Hv, x, v) -> fill!(Hv, 0)), x, x[1])
        for h! in (h!_1, h!_2, h!_3)
            @test_throws ArgumentError("Cannot evaluate the Hessian of the objective function: No suitable Julia function available.") h!(similar(x, length(x), length(x)), x[1])
        end

        # Evaluation of Hessian-vector product
        hv!_1 = NLSolversBase.make_hv(NLSolversBase.InplaceObjective(; fdf = (DF, x) -> (fill!(DF, 1); sum(x))), x, x[1])
        hv!_2 = NLSolversBase.make_hv(NLSolversBase.InplaceObjective(; fgh = (DF, H, x) -> (fill!(DF, 1); copyto!(H, 0); sum(x))), x, x[1])
        for hv! in (hv!_1, hv!_2)
            @test hv! === nothing
        end
    end

    @testset "Non-mutating objectives" begin
        # Evaluation of objective function
        f = make_f(NLSolversBase.NotInplaceObjective(; df = x -> fill!(similar(x), 1)), x, x[1])
        @test_throws ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.") f(x)
        f! = make_f(NLSolversBase.NotInplaceObjective(; df = x -> copyto!(similar(x, length(x), length(x)), I)), x, x)
        @test_throws ArgumentError("Cannot evaluate the objective function: No suitable Julia function available.") f!(similar(x), x)

        # Evaluation of gradient/Jacobian
        df! = make_df(NLSolversBase.NotInplaceObjective(), x, x[1])
        @test_throws ArgumentError("Cannot evaluate the gradient of the objective function: No suitable Julia function available.") df!(similar(x), x)
        jac! = make_df(NLSolversBase.NotInplaceObjective(), x, x)
        @test_throws ArgumentError("Cannot evaluate the Jacobian of the objective function: No suitable Julia function available.") jac!(similar(x, length(x), length(x)), x)

        # Combined evaluation of objective function and gradient/Jacobian
        fdf! = make_fdf(NLSolversBase.NotInplaceObjective(; df = x -> fill!(similar(x), 1)), x, x[1])
        @test_throws ArgumentError("Cannot evaluate the objective function and its gradient: No suitable Julia function available.") fdf!(similar(x), x)
        fjac! = make_fdf(NLSolversBase.NotInplaceObjective(; df = x -> copyto!(similar(x, length(x), length(x)), I)), x, x)
        @test_throws ArgumentError("Cannot evaluate the objective function and its Jacobian: No suitable Julia function available.") fjac!(similar(x), similar(x, length(x), length(x)), x)
    end
end