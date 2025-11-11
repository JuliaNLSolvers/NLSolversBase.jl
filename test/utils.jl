@testset "utililties" begin
    @testset "x_of_nans" begin
        for T in (Int, Float32, Float64)
            for x in (zeros(T), zeros(T, 2), zeros(T, 3, 2))
                x_nans = @inferred(NLSolversBase.x_of_nans(x))
                @test x_nans isa Array{float(T),ndims(x)}
                @test size(x_nans) == size(x)
                @test all(isnan, x_nans)
            end

            for x in (SArray{Tuple{}}(zeros(T)), SArray{Tuple{2}}(zeros(T, 2)), SArray{Tuple{3,2}}(zeros(T, 3, 2)))
                x_nans = @inferred(NLSolversBase.x_of_nans(x))
                @test x_nans isa MArray{Tuple{size(x)...},float(T)}
                @test size(x_nans) == size(x)
                @test all(isnan, x_nans)
            end

            for x in (MArray{Tuple{}}(zeros(T)), MArray{Tuple{2}}(zeros(T, 2)), MArray{Tuple{3,2}}(zeros(T, 3, 2)))
                x_nans = @inferred(NLSolversBase.x_of_nans(x))
                @test x_nans isa MArray{Tuple{size(x)...},float(T)}
                @test size(x_nans) == size(x)
                @test all(isnan, x_nans)
            end
        end
    end
end
