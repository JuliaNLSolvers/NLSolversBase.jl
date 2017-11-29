abstract type AbstractObjective end
real_to_complex(d::AbstractObjective, x) = iscomplex(d) ? real_to_complex(x) : x
complex_to_real(d::AbstractObjective, x) = iscomplex(d) ? complex_to_real(x) : x
