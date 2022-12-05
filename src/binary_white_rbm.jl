"""
    BinaryWhiteRBM(a, b, w, affine_v, affine_h)
    BinaryWhiteRBM(a, b, w)

Construct a whitened RBM with binary visible and hidden units.
"""
function BinaryWhiteRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    affine_v::Affine, affine_h::Affine
)
    rbm = BinaryRBM(a, b, w)
    return WhiteRBM(rbm, affine_v, affine_h)
end

function BinaryWhiteRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    rbm = BinaryRBM(a, b, w)
    return WhiteRBM(rbm)
end
