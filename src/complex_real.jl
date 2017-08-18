"Convert a real array of size 2 x dims to a complex array of size dims"
function real_to_complex(A::Array{T}) where T<:Real
    @assert size(A)[1] == 2
    stripfirst(a, b...) = b
    sizeB = stripfirst(size(A)...) #type-stable way of doing sizeB = size(A)[2:end]
    return reinterpret(Complex{T}, A, sizeB)
end
"Convert a complex array of size dims to a real array of size 2 x dims"
function complex_to_real(B::Array{Complex{T}}) where T<:Real
    sizeA = tuple(2,size(B)...)
    return reinterpret(T, B, sizeA)
end
