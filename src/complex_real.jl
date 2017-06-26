function real_to_complex(A::Array{T}) where T<:Real
    @assert size(A)[1] == 2
    sizeB = size(A)[2:end]
    return reinterpret(Complex{T}, A, sizeB)
end
function complex_to_real(B::Array{Complex{T}}) where T<:Real
    sizeA = tuple(2,size(B)...)
    return reinterpret(T, B, sizeA)
end
