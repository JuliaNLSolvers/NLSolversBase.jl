@testset "incomplete objectives" begin
    import NLSolversBase: df, fdf, make_f, make_df, make_fdf
    function f(x)
        sum(x->x^2,x)
    end

    g!(G, x) = copyto!(G, 2 .* x)
    g(x) = 2 .* x

    function fg!(G, x)
        copyto!(G, 2 .* x)
        sum(x->x^2,x)
    end
    function just_fg!(F, G, x)
        !(G == nothing) && copyto!(G, 2 .* x)
        !(F == nothing) && sum(x->x^2,x)
    end
    fg(x) = f(x), g(x)

    fdf!_real = only_fg!(just_fg!)
    fdf_real = only_fg(fg)

    df_fdf_real = only_g_and_fg(g, fg)
    srand(3259)
    x = rand(10)
    G_cache = similar(x)
    G_cache2 = similar(G_cache)

    @test df(fdf!_real) === nothing
    @test df(fdf_real) === nothing
    @test df(df_fdf_real) === g
    @test df(df_fdf_real)(x) == g(x)

    @test fdf(fdf!_real) === just_fg!
    @test fdf(fdf_real) === fg
    @test df(df_fdf_real) == g
    @test fdf(df_fdf_real) == fg
    @test df(df_fdf_real)(x) == g(x)
    @test fdf(df_fdf_real)(x) == fg(x)

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

    nd_fg = NonDifferentiable(only_fg(fg), x)
    nd_fg! = NonDifferentiable(only_fg!(just_fg!), x)
    for ND in (nd_fg, nd_fg!)
        value!(ND, x)
        value(ND) == f(x)
    end
    od_fg = OnceDifferentiable(only_fg(fg), x)
    od_fg! = OnceDifferentiable(only_fg!(just_fg!), x)
    for OD in (od_fg, od_fg!)
        value!(OD, x)
        @test value(OD) == f(x)
        gradient!(OD, x)
        @test gradient(OD) == g(x)
        @test gradient(OD, x) == g(x)
        value_gradient!(OD, 2 .* x)
        @test value(OD) == f(2 .* x)
        @test gradient(OD) == g(2 .* x)
    end

end
@testset "incomplete objectives vectors" begin
    import NLSolversBase: OnceDifferentiable, df, fdf, make_f, make_df, make_fdf, only_fj!, only_fj, only_j_and_fj
    import NLSolversBase: value!, value, jacobian, jacobian!, value_jacobian!
    import Compat: copyto!
    function tf(x)
        x.^2
    end
    function tf(F, x)
        copyto!(F, tf(x))
    end

    tj!(J, x) = copyto!(J, Matrix(Diagonal(x)))
    tj(x) = Matrix(Diagonal(x))

    function tfj!(F, J, x)
        copyto!(J, Matrix(Diagonal(x)))
        copyto!(F, tf(x))
    end
    function just_tfj!(F, J, x)
        !(J == nothing) && copyto!(J, Matrix(Diagonal(x)))
        !(F == nothing) && copyto!(F, tf(x))
    end
    tfj(x) = tf(x), tj(x)

    fdf!_real = only_fj!(just_tfj!)
    fdf_real = only_fj(tfj)

    df_fdf_real = only_j_and_fj(tj, tfj)
    srand(3259)
    x = rand(10)
    J_cache = similar(Matrix(Diagonal(x)))
    J_cache2 = similar(Matrix(Diagonal(x)))
    F_cache = similar(x)
    F_cache2 = similar(x)

    @test df(fdf!_real) === nothing
    @test df(fdf_real) === nothing
    @test df(df_fdf_real) === tj
    @test df(df_fdf_real)(x) == tj(x)

    @test fdf(fdf!_real) === just_tfj!
    @test fdf(fdf_real) === tfj
    @test df(df_fdf_real) == tj
    @test fdf(df_fdf_real) == tfj
    @test df(df_fdf_real)(x) == tj(x)
    @test fdf(df_fdf_real)(x) == tfj(x)

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
