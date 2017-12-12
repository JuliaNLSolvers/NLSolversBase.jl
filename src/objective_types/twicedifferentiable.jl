# Used for objectives and solvers where the gradient and Hessian is available/exists
mutable struct TwiceDifferentiable{T,TDF,TH,TX} <: AbstractObjective
    f
    df
    fdf
    h
    F::T
    DF::TDF
    H::TH
    x_f::TX
    x_df::TX
    x_h::TX
    f_calls::Vector{Int}
    df_calls::Vector{Int}
    h_calls::Vector{Int}
end
iscomplex(obj::TwiceDifferentiable) = false
# compatibility with old constructor
function TwiceDifferentiable(f,g!,fg!,h!,f_x::T, df::TDF, H::TH, x::TX) where {T, TDF, TH, TX}
    x_f = x_of_nans(x)
    x_df = x_of_nans(x)
    x_h = x_of_nans(x)
    TwiceDifferentiable{T,TDF, TH, TX}(f, g!, fg!, h!, f_x,
                                        df, H,
                                        x_f, x_df, x_h,
                                        [0,], [0,], [0,])
end
# The user friendly/short form TwiceDifferentiable constructor
function TwiceDifferentiable(f, g!, fg!, h!, f_x::Real, x_seed::AbstractVector)
    g = similar(x_seed)
    n = length(x_seed)
    H = similar(x_seed, n, n)
    x = similar(x_seed)
    TwiceDifferentiable(f, g!, fg!, h!, f_x, g, H, x)
end

function TwiceDifferentiable(f, g!, h!, F::Real, x_seed::AbstractVector)
    function fg!(storage, x)
        g!(storage, x)
        return f(x)
    end
    return TwiceDifferentiable(f, g!, fg!, h!, F, x_seed)
end

function TwiceDifferentiable(td::TwiceDifferentiable, x::AbstractArray)
    value_gradient!(td, x)
    hessian!(td, x)
    td
end