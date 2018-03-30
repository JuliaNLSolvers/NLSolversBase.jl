"Convert a real array of size 2 x dims to a complex array of size dims"
function real_to_complex(A::AbstractArray{T}) where T<:Real
    @assert size(A)[1] == 2
    stripfirst(a, b...) = b
    sizeB = stripfirst(size(A)...) #type-stable way of doing sizeB = size(A)[2:end]

    @static if VERSION >= v"0.7.0-DEV.4000"
        return reshape(reinterpret(Complex{T}, A), sizeB)
    else
        return reinterpret(Complex{T}, A, sizeB)
    end
end
"Convert a complex array of size dims to a real array of size 2 x dims"
function complex_to_real(B::AbstractArray{Complex{T}}) where T<:Real
    sizeA = tuple(2,size(B)...)

    @static if VERSION >= v"0.7.0-DEV.4000"
        return reshape(reinterpret(T, B), sizeA)
    else
        return reinterpret(T, B, sizeA)
    end
end
