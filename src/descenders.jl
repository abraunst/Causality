export AdamDescender, SignDescender

"""
`AdamDescender{T}` \n
Concrete type for the gradient descender with adam method.\n
See `step!`.
"""
struct AdamDescender{T}
    m::T
    v::T
    β1::Float64
    β2::Float64
    ε::Float64
    η::Float64
end

"""
`AdamDescender(w, η)`\n
Constructor for concrete type `AdamDescender{T}` \n
`w` is the parameter vector to update. \n
`η` is the desired learning rate. \n
See `step!`.
"""
AdamDescender(w, η) = AdamDescender(zero(w), zero(w), 0.9, 0.999, 1e-8, η)

"""
`step!(w, ∇Q, a)`\n
Performs a step of descent, by updating `w`.\n
`w` is the parameters vector.\n
`∇Q` is the gradient with which `w` is updated.\n
`a` is the method type, can either be a `AdamDescender` or a `SignDescender`
"""
function step!(w, ∇Q, a::AdamDescender)
    a.m .= a.β1 .* a.m .+ (1 - a.β1) .* ∇Q
    a.v .= a.β2 .* a.v .+ (1-a.β2) .* ∇Q.^2
    w .-= a.η .* (a.m ./ (1-a.β1)) ./ (sqrt.(a.v ./ (1-a.β2)) .+ a.ε)
end

"""
`SignDescender(η)` \n
Concrete type for the sign descender with SignDescender method.\n
`η` is the desired learning rate.\n
See `step!`.\n
"""
struct SignDescender
    η::Float64
end

function step!(w, ∇Q, s::SignDescender)
    w .-= s.η .* abs.(w) .* sign.(∇Q)
end
