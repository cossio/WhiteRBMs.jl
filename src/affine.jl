struct Affine{M,V}
    A::M
    u::V
    """
        Affine(A, b)

    Represents an (inverse) affine transform of the form `x -> A * (x - u)`.
    """
    function Affine(A::AbstractMatrix, u::AbstractVector)
        @assert size(A, 2) == length(u)
        return new{typeof(A), typeof(u)}(A, u)
    end
end

function Affine(A::AbstractMatrix)
    u = zeros(eltype(A), size(A, 2))
    return Affine(A, u)
end

function Affine(u::AbstractVector)
    A = Diagonal(ones(eltype(u), length(u)))
    return Affine(A, u)
end

Affine(::Type{T}, n::Int) where {T} = Affine(zeros(T, n))
Affine(n::Int) = Affine(Float64, n)

"""
    one(t::Affine)

Returns the identity affine transformation, which maps all points to themselves.
"""
Base.one(t::Affine) = Affine(one(t.A), zero(t.u))

"""
    zero(t::Affine)

Returns the zero affine transformation, which maps all points to zero.
"""
Base.zero(t::Affine) = Affine(zero(t.A), zero(t.u))

function Base.copy!(dst::Affine, src::Affine)
    dst.A .= src.A
    dst.u .= src.u
    return dst
end

Base.:*(t::Affine, x::AbstractVecOrMat) = t.A * (x .- t.u)
Base.:\(t::Affine, y::AbstractVecOrMat) = t.A \ y .+ t.u
Base.:*(s::Affine, t::Affine) = Affine(s.A * t.A, t \ s.u)
Base.:\(s::Affine, t::Affine) = Affine(s.A \ t.A, t.u - t.A \ s.A * s.u)

Base.:*(t::Affine, c::Number) = Affine(t.A * c, t.u / c)
Base.:*(c::Number, t::Affine) = Affine(c * t.A, t.u)
Base.:-(t::Affine) = Affine(-t.A, t.u)

Base.:+(s::Affine, t::Affine) = Affine(s.A + t.A, (s.A + t.A) \ (s.A * s.u + t.A * t.u))
Base.:-(s::Affine, t::Affine) = Affine(s.A - t.A, (s.A - t.A) \ (s.A * s.u - t.A * t.u))

Base.inv(t::Affine) = Affine(inv(t.A), -t.A * t.u)

"""
    whitening_transform(μ, C)

Returns the `Affine` transform that whitens data with mean `μ` and covariance `C`.
"""
whitening_transform(u::AbstractVector, C::AbstractMatrix) = Affine(inv(cholesky(C).L), u)
whitening_transform(C::AbstractMatrix) = Affine(inv(cholesky(C).L))
whitening_transform(u::AbstractVector) = Affine(u)

function whitening_transform!(affine::Affine, μ::AbstractVector, C::AbstractMatrix)
    affine.A .= inv(cholesky(C).L)
    affine.u .= μ
    return affine
end

function whitening_transform!(affine::Affine, μ::AbstractVector)
    copyto!(affine.A, I)
    affine.u .= μ
    return affine
end

function whitening_transform!(affine::Affine, C::AbstractMatrix)
    affine.A .= inv(cholesky(C).L)
    affine.u .= 0
    return affine
end
