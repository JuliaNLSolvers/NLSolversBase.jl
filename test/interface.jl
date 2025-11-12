for T in (Float32, Float64, BigFloat)
    @testset "interface" begin
        @testset "single-valued" begin
            x_seed = zeros(T, 2)
            g_seed = zeros(T, 2)
            h_seed = zeros(T, 2, 2)
            f_x_seed = exponential(x_seed)
            g_x_seed = exponential_gradient!(zeros(T, 2), x_seed)
            h_x_seed = exponential_hessian!(zeros(T, 2, 2), x_seed)

            x_alt = ones(T, 2)
            f_x_alt = exponential(x_alt)
            g_x_alt = exponential_gradient!(zeros(T, 2), x_alt)
            h_x_alt = exponential_hessian!(zeros(T, 2, 2), x_alt)

            # Construct instances
            nd = NonDifferentiable(exponential, x_seed)
            od = OnceDifferentiable(
                exponential,
                exponential_gradient!,
                exponential_value_gradient!,
                x_seed,
                T(0),
                g_seed,
            )
            td = TwiceDifferentiable(
                exponential,
                exponential_gradient!,
                exponential_value_gradient!,
                exponential_hessian!,
                x_seed,
                T(0),
                g_seed,
                h_seed,
            )

            # Test call counters: Initially all are zero
            for f in (nd, od, td)
                @test iszero(f_calls(f))
                @test iszero(g_calls(f))
                @test iszero(jvp_calls(f))
                @test iszero(h_calls(f))
                @test iszero(hv_calls(f))
            end

            # Force evaluation
            value!!(nd, x_seed)
            value_gradient!!(od, x_seed)
            value_gradient!!(td, x_seed)
            hessian!!(td, x_seed)

            # Test that the call counters got incremented:
            # - One function evaluation for nd, od and td
            # - One gradient evaluation for od and td
            # - One Hessian evaluation for td
            @test f_calls(nd) == f_calls(od) == f_calls(td) == 1
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 1
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 1
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            # Test that values are the same, and that values match those
            # calculated by the value(obj, x) methods
            @test value(nd) == value(od) == value(td) ≈ f_x_seed
            @test f_calls(nd) == f_calls(od) == f_calls(td) == 1
            @test value(nd, x_seed) == value(od, x_seed) == value(td, x_seed)
            @test f_calls(nd) == f_calls(od) == f_calls(td) == 2

            # Test that the gradients match the intended values
            @test gradient(od) == gradient(td) == g_x_seed
            # Test that the Hessian matches the intended value
            @test hessian(td) == h_x_seed

            # Test hv_product!
            v = T.([0.111, -1234])
            @test hv_product!(td, x_seed, v) == h_x_seed * v

            # Test that the call counter is not incremented:
            # `hv_product!` falls back to computing Hessian, but the Hessian at `x_seed` is cached
            @test h_calls(td) == 1
            @test iszero(hv_calls(td))

            # Test that the call counters are not incremented with single-"bang" methods...
            value!(nd, x_seed)
            value_gradient!(od, x_seed)
            value_gradient!(td, x_seed)
            hessian!(td, x_seed)

            @test f_calls(nd) == f_calls(od) == f_calls(td) == 2
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 1
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 1
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            # ... and that they do with double-"bang" methods
            # - One additional function evaluation for nd, od and td
            # - One additional gradient evaluation for od and td
            # - One additional Hessian evaluation for td
            value!!(nd, x_seed)
            value_gradient!!(od, x_seed)
            value_gradient!!(td, x_seed)
            hessian!!(td, x_seed)

            @test f_calls(nd) == f_calls(od) == f_calls(td) == 3
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 2
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 2
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            # Test that gradient doesn't work for NonDifferentiable, but does otherwise
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 2
            @test_throws MethodError gradient!(nd, x_alt)
            gradient!(od, x_alt)
            gradient!(td, x_alt)
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 3

            @test value(nd) == value(od) == value(td) ≈ f_x_seed
            @test gradient(td) == g_x_alt
            @test gradient(td) == [gradient(td, i) for i = 1:length(x_seed)]
            @test hessian(td) == h_x_seed

            @test f_calls(nd) == f_calls(od) == f_calls(td) == 3
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 3
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 2
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            @test_throws MethodError hessian!(nd, x_alt)
            @test_throws MethodError hessian!(od, x_alt)
            hessian!(td, x_alt)
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 3

            @test value(nd) == value(od) == value(td) ≈ f_x_seed
            @test gradient(td) == g_x_alt
            @test hessian(td) == h_x_alt

            @test f_calls(nd) == f_calls(od) == f_calls(td) == 3
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 3
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 3
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            value!(nd, x_alt)
            value!(od, x_alt)
            value!(td, x_alt)
            @test value(nd) == value(od) == value(td) == f_x_alt
            @test gradient(td) == g_x_alt
            @test hessian(td) == h_x_alt

            # One additional function evaluation for nd, od and td
            @test f_calls(nd) == f_calls(od) == f_calls(td) == 4
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 3
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 3
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            @test_throws MethodError value_gradient!(nd, x_seed)
            value_gradient!(od, x_seed)
            value_gradient!(td, x_seed)
            @test value(od) == value(td) ≈ f_x_seed
            # change x_f manually to test branch
            od.x_f .*= 0
            td.x_f .*= 0
            value_gradient!(od, x_seed)
            value_gradient!(td, x_seed)
            @test value(od) == value(td) ≈ f_x_seed
            # change x_df manually to test branch
            # only df is meant to be re-calculated as
            # d.x_f == x_seed
            # and that the hessian counter doesn't increment
            od.x_df .*= 0
            td.x_df .*= 0
            value_gradient!(od, x_seed)
            value_gradient!(td, x_seed)
            @test value(od) == value(td) ≈ f_x_seed
            @test gradient(td) == g_x_seed
            @test hessian(td) == h_x_alt

            # Updated call counters: 1 additional function + gradient evaluation for od and td
            @test f_calls(nd) == 4
            @test f_calls(od) == f_calls(td) == 5
            @test iszero(g_calls(nd))
            @test g_calls(od) == g_calls(td) == 4
            @test iszero(h_calls(nd))
            @test iszero(h_calls(od))
            @test h_calls(td) == 3
            for f in (nd, od, td)
                @test iszero(jvp_calls(f))
                @test iszero(hv_calls(f))
            end

            # test the non-mutating gradient() function
            gradient!(od, fill(T(1), 2))
            gradient!(td, fill(T(1), 2))
            od_df_old = copy(gradient(od))
            td_df_old = copy(gradient(td))
            gradient!(od, fill(T(-1), 2))
            gradient!(td, fill(T(-1), 2))
            @test od_df_old == gradient(od, fill(T(1), 2))
            @test td_df_old == gradient(td, fill(T(1), 2))
            # Mutate obj.DF directly to check that the gradient
            # is not recalculated if the same x is reused.
            # We obviously cannot just rerun for the same x twice
            # as we would then not be able to tell if it was just
            # calculated again, or the cache was simply returned
            # as intended.
            fill!(od.DF, T(0))
            fill!(td.DF, T(0))
            @test gradient(od) == zeros(T, 2)
            @test gradient(td) == zeros(T, 2)

            # Reset all call counters and set all cached values to `NaN`
            clear!(nd)
            clear!(od)
            clear!(td)
            for f in (nd, od, td)
                @test iszero(f_calls(f))
                @test iszero(g_calls(f))
                @test iszero(jvp_calls(f))
                @test iszero(h_calls(f))
                @test iszero(hv_calls(f))
            end
            @test isnan(nd.F)
            @test isnan(od.F)
            @test all(isnan, od.DF)
            @test isnan(od.JVP)
            @test isnan(td.F)
            @test all(isnan, td.DF)
            @test isnan(td.JVP)
            @test all(isnan, td.H)
            @test all(isnan, td.Hv)

            @testset "Hessian-vector product" begin
                for (name, prob) in MultivariateProblems.UnconstrainedProblems.examples
                    if prob.istwicedifferentiable
                        hv!(storage::Vector, x::Vector, v::Vector) = begin
                            n = length(x)
                            H = x*x'
                            MVP.hessian(prob)(H, x)
                            mul!(storage, H, v)
                        end
                        fg!(F, G, x::Vector) = begin
                            if G !== nothing
                                MVP.gradient(prob)(G, x)
                            end
                            if F !== nothing
                                return T(MVP.objective(prob)(x))
                            else
                                return nothing
                            end
                        end
                        xT = T.(prob.initial_x)
                        nxT = length(xT)
                        ddf = TwiceDifferentiable(only_fg_and_hv!(fg!, hv!), xT)
                        x = rand(xT, nxT)
                        v = rand(xT, nxT)
                        G = NLSolversBase.alloc_DF(x, T(0))
                        H = NLSolversBase.alloc_H(x, T(0))
                        MVP.hessian(prob)(H, x)
                        @test hv_product!(ddf, x, v) == H*v
                        @test ddf.Hv == H*v
                        F = fg!(0.0, G, x)
                        @test gradient!(ddf, x) == G
                        @test value!(ddf, x) == F
                        @test f_calls(ddf) == 1
                        @test g_calls(ddf) == 1
                        @test h_calls(ddf) == 0
                        @test hv_calls(ddf) == 1
                        clear!(ddf)
                        @test f_calls(ddf) == 0
                        @test g_calls(ddf) == 0
                        @test h_calls(ddf) == 0
                        @test hv_calls(ddf) == 0
                    end
                end
            end
        end
        @testset "multivalued" begin
            # Test example: Rosenbrock MINPACK
            function f!(F::Vector, x::Vector)
                F[1] = 1 - x[1]
                F[2] = 10(x[2]-x[1]^2)
                F
            end
            function j!(J::Matrix, x::Vector)
                J[1, 1] = -1
                J[1, 2] = 0
                J[2, 1] = -20x[1]
                J[2, 2] = 10
                J
            end

            x_seed = zeros(T, 2)
            F_seed = zeros(T, 2)
            J_seed = zeros(T, 2, 2)
            F_x_seed = T.([1.0, 0.0])
            J_x_seed = T.([-1.0 0.0; -0.0 10.0])

            x_alt = T.([0.5, 0.5])
            F_x_alt = T.([0.5, 2.5])
            J_x_alt = T.([-1.0 0.0; -10.0 10.0])

            # Construct instances
            nd = NonDifferentiable(f!, x_seed, F_seed)
            od = OnceDifferentiable(f!, j!, x_seed, F_seed, J_seed)

            @testset "forwarddiff with different types" begin
                od_ad = OnceDifferentiable(f!, x_seed, F_seed; autodiff = AutoForwardDiff())
                value!!(od_ad, x_seed)
                value_jacobian!!(od_ad, x_seed)
            end
            # Force evaluation
            value!!(nd, x_seed)
            value_jacobian!!(od, x_seed)

            # Test that values are the same, and that values match those
            # calculated by the value(obj, x) methods
            @test value(nd) == value(od) ≈ F_x_seed
            @test value(nd, x_seed) == value(od, x_seed)
            @test value(nd, x_alt) == value(od, x_alt)

            # Test that the Jacobians match the intended values
            @test jacobian(od) == J_x_seed

            # Test that the call counters got incremented
            @test nd.f_calls == od.f_calls == 3
            @test od.df_calls == 1

            # Test that the call counters do not get incremented
            # with single-"bang" methods...
            value!(nd, x_seed)
            value_jacobian!(od, x_seed)

            @test nd.f_calls == od.f_calls == 3
            @test od.df_calls == 1

            # ... and that they do with double-"bang" methods
            value!!(nd, x_seed)
            value_jacobian!!(od, x_seed)

            @test nd.f_calls == od.f_calls == 4
            @test od.df_calls == 2

            # Test that jacobian doesn't work for NonDifferentiable, but does otherwise
            @test_throws MethodError jacobian!(nd, x_alt)
            jacobian!(od, x_alt)

            @test value(nd) == value(od) ≈ F_x_seed
            @test jacobian(od) == J_x_alt
            @test nd.f_calls == od.f_calls == 4
            @test od.df_calls == 3

            @test value(nd) == value(od) ≈ F_x_seed
            @test jacobian(od) == J_x_alt
            @test nd.f_calls == od.f_calls == 4
            @test od.df_calls == 3

            value!(nd, x_alt)
            value!(od, x_alt)
            @test value(nd) == value(od) == F_x_alt
            @test jacobian(od) == J_x_alt
            @test nd.f_calls == od.f_calls == 5
            @test od.df_calls == 3

            @test_throws MethodError value_jacobian!(nd, x_seed)
            value_jacobian!(od, x_seed)
            @test value(od) ≈ F_x_seed
            # change x_f manually to test branch
            od.x_f .*= 0
            value_jacobian!(od, x_seed)
            @test value(od) ≈ F_x_seed
            # change x_df manually to test branch
            # only df is meant to be re-calculated as
            # d.x_f == x_seed
            od.x_df .*= 0
            value_jacobian!(od, x_seed)
            @test value(od) ≈ F_x_seed
            @test jacobian(od) == J_x_seed
            @test od.f_calls == 6
            @test od.df_calls == 4

            clear!(nd)
            clear!(od)
            @test nd.f_calls == 0
            @test od.f_calls == 0
            @test od.df_calls == 0

            # Test that the correct branch gets called if jacobian hasn't been
            # calculated yet
            xx = rand(T, 2)
            @test jacobian!(od, xx) == j!(NLSolversBase.alloc_DF(xx, xx), xx)
            @test od.x_df == xx

            # Test the branching in value_jacobian! works as expected
            xxx = rand(T, 2)
            value_jacobian!(od, xxx)
            @test xxx == od.x_f == od.x_df
            xxx2 = rand(T, 2)
            value!(od, xxx2)
            @test xxx2 == od.x_f
            @test xxx == od.x_df
            xxx3 = rand(T, 2)
            jacobian!(od, xxx3)
            @test xxx2 == od.x_f
            @test xxx3 == od.x_df

        end
    end
end
