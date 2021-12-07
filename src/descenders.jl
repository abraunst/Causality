export AdamDescender, SignDescender

struct AdamDescender{T}
    m::T
    v::T
    β1::Float64
    β2::Float64
    ε::Float64
    η::Float64
end

AdamDescender(w, η) = AdamDescender(zero(w), zero(w), 0.9, 0.999, 1e-8, η)

function step!(w, ∇Q, a::AdamDescender)
    a.m .= a.β1 .* a.m .+ (1 - a.β1) .* ∇Q
    a.v .= a.β2 .* a.v .+ (1-a.β2) .* ∇Q.^2
    w .-= a.η .* (a.m ./ (1-a.β1)) ./ (sqrt.(a.v ./ (1-a.β2)) .+ a.ε)
end

struct SignDescender
    η::Float64
end

function step!(w, ∇Q, s::SignDescender)
    w .-= s.η .* abs.(w) .* sign.(∇Q)
end
