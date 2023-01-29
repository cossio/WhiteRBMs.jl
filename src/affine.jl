abstract type AbstractAffine end

# Represents a general (inverse) affine transform of the form `x -> A * (x - u)`.
struct Affine{M,V} <: AbstractAffine
    A::M
    u::V
    function Affine(A::AbstractMatrix, u::AbstractVector)
        @assert size(A, 1) == size(A, 2) == length(u)
        return new{typeof(A), typeof(u)}(A, u)
    end
end
Base.one(a::Affine) = Affine(one(a.A), zero(a.u))
Base.zero(a::Affine) = Affine(zero(a.A), zero(a.u))

# Represents a centering (inverse) affine transform of the form `x -> x - u`.
struct CenterAffine{V} <: AbstractAffine
    u::V
    CenterAffine(u::AbstractVector) = new{typeof(u)}(u)
end
Base.propertynames(::CenterAffine) = (:A, :u)
Base.getproperty(a::CenterAffine, s::Symbol) = s === :A ? I : getfield(a, s)
Base.one(a::CenterAffine) = CenterAffine(zero(a.u))

# Represents a standardization (inverse) affine transform of the form `x -> A * (x - u)`, where `A` is diagonal.
struct StdizeAffine{M,V} <: AbstractAffine
    A::M
    u::V
    StdizeAffine(A::Diagonal, u::AbstractVector) = new{typeof(A), typeof(u)}(A, u)
end
Base.one(a::StdizeAffine) = StdizeAffine(one(a.A), zero(a.u))
Base.zero(a::StdizeAffine) = StdizeAffine(zero(a.A), zero(a.u))

function Base.copy!(dst::AbstractAffine, src::AbstractAffine)
    copyto!(dst.A, src.A)
    copyto!(dst.u, src.u)
    return dst
end

Base.:*(t::AbstractAffine, x::AbstractVecOrMat) = t.A * (x .- t.u)
Base.:\(t::AbstractAffine, y::AbstractVecOrMat) = t.A \ y .+ t.u
Base.:*(s::AbstractAffine, t::AbstractAffine) = Affine(s.A * t.A, t \ s.u)
Base.:\(s::AbstractAffine, t::AbstractAffine) = Affine(s.A \ t.A, t.u - t.A \ s.A * s.u)
Base.:*(t::AbstractAffine, c::Number) = Affine(t.A * c, t.u / c)
Base.:*(c::Number, t::AbstractAffine) = Affine(c * t.A, t.u)
Base.:-(t::AbstractAffine) = Affine(-t.A, t.u)
Base.:+(s::AbstractAffine, t::AbstractAffine) = Affine(s.A + t.A, (s.A + t.A) \ (s.A * s.u + t.A * t.u))
Base.:-(s::AbstractAffine, t::AbstractAffine) = Affine(s.A - t.A, (s.A - t.A) \ (s.A * s.u - t.A * t.u))
Base.inv(t::AbstractAffine) = Affine(inv(t.A), -t.A * t.u)

"""
    whitening_transform(μ, C)

Returns the `Affine` transform that whitens data with mean `μ` and covariance `C`.
"""
whitening_transform(u::AbstractVector, C::AbstractMatrix) = Affine(inv(cholesky(C).L), u)
whitening_transform(u::AbstractVector, C::Diagonal) = StdizeAffine(inv(cholesky(C).L), u)
whitening_transform(u::AbstractVector) = CenterAffine(u)
