"Convert a real array of size 2 x dims to a complex array of size dims"
function real_to_complex(A::AbstractArray{T}) where T<:Real
    @assert size(A)[1] == 2
    stripfirst(a, b...) = b
    sizeB = stripfirst(size(A)...) #type-stable way of doing sizeB = size(A)[2:end]

    @static if VERSION >= v"0.7.0-DEV.393"
        # Given performance issues of reinterpreted Arrays that haven't been fixed as of this PR,
        # and that a few tests are failing due to failure to convert reinterpreted arrays to regular arrays
        # I figured a suitable hack was to suck it up and copy.
        # In a benchmark, this was faster than copy(reshape(reinterpret(Complex{T}, A),sizeB))
        B = Array{Complex{T}}(undef, sizeB)
        @inbounds for i ∈ eachindex(B)
            B[i] = A[2i-1] + im*A[2i]
        end
        return B
    else
        return reinterpret(Complex{T}, A, sizeB)
    end
end
"Convert a complex array of size dims to a real array of size 2 x dims"
function complex_to_real(B::AbstractArray{Complex{T}}) where T<:Real
    sizeA = tuple(2,size(B)...)

    @static if VERSION >= v"0.7.0-DEV.393"
        A = Array{T}(undef, sizeA)
        @inbounds for i ∈ eachindex(B)
            A[2i-1] = real(B[i])
            A[2i]   = imag(B[i])
        end
        return A
    else
        return reinterpret(T, B, sizeA)
    end
end
