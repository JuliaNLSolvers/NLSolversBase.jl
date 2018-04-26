mutable struct LinSolveFactorize{F}
  factorization::F
  A
end
LinSolveFactorize(factorization = factorize) = LinSolveFactorize(factorization,nothing)
function (p::LinSolveFactorize)(x,A,b,update_matrix=false)
    if update_matrix
      p.A = p.factorization(A)
    end
    copy!(x, p.A\b)
end
