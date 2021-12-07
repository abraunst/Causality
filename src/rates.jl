using SpecialFunctions, IntervalUnionArithmetic

export ConstantRate, GaussianRate, MaskedRate


abstract type RateContinuous end

struct GaussianRate{A,B,C} <: RateContinuous
    a::A
    b::B
    c::C
end

function density(g::GaussianRate, t)
    a, b, c = g.a[], g.b[], g.c[]
    a*exp(-((t-b)/c)^2)
end

function cumulated(g::GaussianRate, t)
    a, b, c = g.a[], g.b[], g.c[]
    a*c*0.5*sqrt(π)*(erfc(-(t-b)/c)-erfc(b/c))
end

function infect(g::GaussianRate, tj)
    a, b, c = g.a[], g.b[], g.c[]
    y=2/(a*c*sqrt(π))*(cumulated(g, tj) - log(rand())) + erfc(b/c);
    if y > 2
        return Inf;
    end
    return -c * erfcinv(y) + b
end


struct ConstantRate{T} <: RateContinuous
    c::T
end

cumulated(c::ConstantRate, Δt) = c.c*Δt
infect(c::ConstantRate, tj) = tj - log(rand())/c.c
density(c::ConstantRate, t) = c.c


struct MaskedRate{R} <: RateContinuous
    rate::R
    mask::IntervalUnion{Float64}
end


density(m::MaskedRate, t) = t ∈ m.mask ? density(m.rate, t) : 0.0

function cumulated(m::MaskedRate, t)
    s = 0.0
    for ab in m.mask.v
        if right(ab) < t
            s += cumulated(m.rate, right(ab)) - cumulated(m.rate, left(ab))
        elseif left(ab) < t
            s += cumulated(m.rate, t) - cumulated(m.rate, left(ab))
        else
            break
        end
    end
    return s
end

function infect(m::MaskedRate, tj)
    for ab in m.mask.v
        tj > right(ab) && continue
        t = infect(m.rate, max(left(ab),tj))
        t ∈ ab  && return t
    end
    return Inf
end

Base.:*(m::MaskedRate, g::RateContinuous) = MaskedRate(m.rate*g, m.intervals)
Base.:*(m::MaskedRate, n::MaskedRate) = MaskedRate(m.rate*n.rate, m.intervals ∩ n.interfals)
