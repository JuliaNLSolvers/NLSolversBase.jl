# Used for objectives and solvers where the gradient is available/exists
mutable struct OnceDifferentiable{TF, TDF, TX} <: AbstractObjective
    f # objective
    df # (partial) derivative of objective
    fdf # objective and (partial) derivative of objective
    F::TF # cache for f output
    DF::TDF # cache for df output
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    f_calls::Vector{Int}
    df_calls::Vector{Int}
end

### Only the objective
# Ambiguity
OnceDifferentiable(f, x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F); inplace = true, autodiff = :finite,  
                   chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x)) =
    OnceDifferentiable(f, x, F, DF, autodiff, chunk)
#OnceDifferentiable(f, x::AbstractArray, F::AbstractArray; autodiff = :finite) =
#    OnceDifferentiable(f, x::AbstractArray, F::AbstractArray, alloc_DF(x, F))
function OnceDifferentiable(f, x::AbstractArray,
                   F::AbstractArray, DF::AbstractArray = alloc_DF(x, F);
                   inplace = true, autodiff = :finite)
    f! = f!_from_f(f, F, inplace)

    OnceDifferentiable(f!, x::AbstractArray, F::AbstractArray, DF, autodiff)
end


function OnceDifferentiable(f, x_seed::AbstractArray{T},
                            F::Real,
                            DF::AbstractArray,
                            autodiff, chunk) where T
    # When here, at the constructor with positional autodiff, it should already
    # be the case, that f is inplace.
    if  typeof(f) <: Union{InplaceObjective, NotInplaceObjective}

        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)

        return OnceDifferentiable(fF, dfF, fdfF, x_seed, F, DF)
    else
        if is_finitediff(autodiff)

            # Figure out which Val-type to use for FiniteDiff based on our
            # symbol interface.
            fdtype = finitediff_fdtype(autodiff)
            df_array_spec = DF
            x_array_spec = x_seed
            return_spec = typeof(F)
            gcache = FiniteDiff.GradientCache(df_array_spec, x_array_spec, fdtype, return_spec)

            function g!(storage, x)
                FiniteDiff.finite_difference_gradient!(storage, f, x, gcache)
                return
            end
            function fg!(storage, x)
                g!(storage, x)
                return f(x)
            end
        elseif is_forwarddiff(autodiff)
            gcfg = ForwardDiff.GradientConfig(f, x_seed, chunk)
            g! = (out, x) -> ForwardDiff.gradient!(out, f, x, gcfg)

            fg! = (out, x) -> begin
                gr_res = DiffResults.DiffResult(zero(T), out)
                ForwardDiff.gradient!(gr_res, f, x, gcfg)
                DiffResults.value(gr_res)
            end
        else
            error("The autodiff value $autodiff is not supported. Use :finite or :forward.")
        end

        return OnceDifferentiable(f, g!, fg!, x_seed, F, DF)
    end
end

has_not_dep_symbol_in_ad = Ref{Bool}(true)
OnceDifferentiable(f, x::AbstractArray, F::AbstractArray, autodiff::Symbol, chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x)) =
OnceDifferentiable(f, x, F, alloc_DF(x, F), autodiff, chunk)
function OnceDifferentiable(f, x::AbstractArray, F::AbstractArray,
                            autodiff::Bool, chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x))
    if autodiff == false
        throw(ErrorException("It is not possible to set the `autodiff` keyword to `false` when constructing a OnceDifferentiable instance from only one function. Pass in the (partial) derivative or specify a valid `autodiff` symbol."))
    elseif has_not_dep_symbol_in_ad[]
        @warn("Setting the `autodiff` keyword to `true` is deprecated. Please use a valid symbol instead.")
        has_not_dep_symbol_in_ad[] = false
    end
    OnceDifferentiable(f, x, F, alloc_DF(x, F), :forward, chunk)
end
function OnceDifferentiable(f, x_seed::AbstractArray, F::AbstractArray, DF::AbstractArray,
    autodiff::Symbol , chunk::ForwardDiff.Chunk = ForwardDiff.Chunk(x_seed))
    if  typeof(f) <: Union{InplaceObjective, NotInplaceObjective}
        fF = make_f(f, x_seed, F)
        dfF = make_df(f, x_seed, F)
        fdfF = make_fdf(f, x_seed, F)
        return OnceDifferentiable(fF, dfF, fdfF, x_seed, F, DF)
    else
        if is_finitediff(autodiff)
            # Figure out which Val-type to use for FiniteDiff based on our
            # symbol interface.
            fdtype = finitediff_fdtype(autodiff)
            # Apparently only the third input is aliased.
            j_finitediff_cache = FiniteDiff.JacobianCache(copy(x_seed), copy(F), copy(F), fdtype)
            if autodiff == :finiteforward
                # These copies can be done away with if we add a keyword for
                # reusing arrays instead for overwriting them.
                Fx = copy(F)
                DF = copy(DF)

                x_f, x_df = x_of_nans(x_seed), x_of_nans(x_seed)
                f_calls, j_calls = [0,], [0,]
                function j_finiteforward!(J, x)
                    # Exploit the possibility that it might be that x_f == x
                    # then we don't have to call f again.

                    # if at least one element of x_f is different from x, update
                    if any(x_f .!= x)
                        f(Fx, x)
                        f_calls .+= 1
                    end

                    FiniteDiff.finite_difference_jacobian!(J, f, x, j_finitediff_cache, Fx)
                end
                function fj_finiteforward!(F, J, x)
                    f(F, x)
                    FiniteDiff.finite_difference_jacobian!(J, f, x, j_finitediff_cache, F)
                end


                return OnceDifferentiable(f, j_finiteforward!, fj_finiteforward!, Fx, DF, x_f, x_df, f_calls, j_calls)
            end

            function fj_finitediff!(F, J, x)
                f(F, x)
                FiniteDiff.finite_difference_jacobian!(J, f, x, j_finitediff_cache)
                F
            end
            function j_finitediff!(J, x)
                F_cache = copy(F)
                fj_finitediff!(F_cache, J, x)
            end

            return OnceDifferentiable(f, j_finitediff!, fj_finitediff!, x_seed, F, DF)

        elseif is_forwarddiff(autodiff)

            jac_cfg = ForwardDiff.JacobianConfig(f, F, x_seed, chunk)
            ForwardDiff.checktag(jac_cfg, f, x_seed)

            F2 = copy(F)
            function j_forwarddiff!(J, x)
                ForwardDiff.jacobian!(J, f, F2, x, jac_cfg, Val{false}())
            end
            function fj_forwarddiff!(F, J, x)
                jac_res = DiffResults.DiffResult(F, J)
                ForwardDiff.jacobian!(jac_res, f, F2, x, jac_cfg, Val{false}())
                DiffResults.value(jac_res)
            end

            return OnceDifferentiable(f, j_forwarddiff!, fj_forwarddiff!, x_seed, F, DF)
        else
            error("The autodiff value $(autodiff) is not supported. Use :finite or :forward.")
        end
    end
end

### Objective and derivative
function OnceDifferentiable(f, df,
                   x::AbstractArray,
                   F::Real = real(zero(eltype(x))),
                   DF::AbstractArray = alloc_DF(x, F);
                   inplace = true)


    df! = df!_from_df(df, F, inplace)

    fdf! = make_fdf(x, F, f, df!)

    OnceDifferentiable(f, df!, fdf!, x, F, DF)
end

function OnceDifferentiable(f, j,
                   x::AbstractArray,
                   F::AbstractArray,
                   J::AbstractArray = alloc_DF(x, F);
                   inplace = true)

    f! = f!_from_f(f, F, inplace)
    j! = df!_from_df(j, F, inplace)
    fj! = make_fdf(x, F, f!, j!)

    OnceDifferentiable(f!, j!, fj!, x, F, J)
end


### Objective, derivative and combination
function OnceDifferentiable(f, df, fdf,
    x::AbstractArray,
    F::Real = real(zero(eltype(x))),
    DF::AbstractArray = alloc_DF(x, F);
    inplace = true)

    # f is never "inplace" since F is scalar
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f, x_df = x_of_nans(x), x_of_nans(x)

    OnceDifferentiable{typeof(F),typeof(DF),typeof(x)}(f, df!, fdf!,
    copy(F), copy(DF),
    x_f, x_df,
    [0,], [0,])
end

function OnceDifferentiable(f, df, fdf,
                            x::AbstractArray,
                            F::AbstractArray,
                            DF::AbstractArray = alloc_DF(x, F);
                            inplace = true)

    f = f!_from_f(f, F, inplace)
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)

    x_f, x_df = x_of_nans(x), x_of_nans(x)

    OnceDifferentiable(f, df!, fdf!, copy(F), copy(DF), x_f, x_df, [0,], [0,])
end
