function real_to_complex(A::AbstractArray{T}) where T<:Real
    @assert size(A)[1] == 2
    sizeB = size(A)[2:end]
    return unsafe_wrap(Array, convert(Ptr{Complex{T}}, pointer(A)), sizeB)
end
function complex_to_real(B::AbstractArray{Complex{T}}) where T<:Real
    sizeA = tuple(2,size(B)...)
    return unsafe_wrap(Array, convert(Ptr{T}, pointer(B)), sizeA)
end
