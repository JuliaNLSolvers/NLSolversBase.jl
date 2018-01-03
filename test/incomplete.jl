@testset "incomplete objectives" begin
    import NLSolversBase: df, fdf, make_f, make_df, make_fdf
    function f(x)
        sum(x->x^2,x)
    end

    g!(G, x) = copy!(G, 2.*x)
    g(x) = 2.*x

    function fg!(G, x)
        copy!(G, 2.*x)
        sum(x->x^2,x)
    end
    function just_fg!(F, G, x)
        !(G == nothing) && copy!(G, 2.*x)
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
    nd_fg! = NonDifferentiable(only_fg(just_fg!), x)
    for ND in (nd_fg, nd_fg!)
        value!(ND, x)
        value(ND) == f(x)
    end
    od_fg = NonDifferentiable(only_fg(fg), x)
    od_fg! = NonDifferentiable(only_fg(just_fg!), x)
    for OD in (od_fg, od_fg!)
        value!(OD, x)
        value(OD) == f(x)
        gradient!(OD, x)
        gradient(OD) == f(x)
        value_gradient!(OD, 2.*x)
        value(OD), gradient(OD) == f(2.*x), g(2.*x)
    end

end