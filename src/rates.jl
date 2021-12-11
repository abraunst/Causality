using SpecialFunctions, IntervalUnionArithmetic

export ConstantRate, GaussianRate, MaskedRate, UnitRate


abstract type RateContinuous end




struct GaussianRate{T} <: RateContinuous
    a::T
    b::T
    c::T
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



function Base.:*(g1::GaussianRate, g2::GaussianRate)
    a1, b1, c1 = g1.a[], g1.b[], g1.c[]
    a2, b2, c2 = g2.a[], g2.b[], g2.c[]
    peak = a1 * a2 * exp(- (b1 - b2)^2 / (c1^2 + c2^2))
    avg  = (b1 * c2^2 + b2 * c1^2)/(c1^2+c2^2)
    var  = (c1^2 * c2^2) / (c1^2 + c2^2)
    GaussianRate(peak,avg,sqrt(var))
end

shift(g::GaussianRate, tj) = GaussianRate(g.a,g.b+tj,g.c)

struct ConstantRate{T} <: RateContinuous
    c::T
end

Base.:*(g1::ConstantRate, g2::ConstantRate) = ConstantRate(g1.c*g2.c)
Base.:*(g1::ConstantRate, g2::GaussianRate) = GaussianRate(g1.c*g2.a,g2.b,g2.c)
Base.:*(g2::GaussianRate, g1::ConstantRate) = g1*g2
cumulated(c::ConstantRate, Δt) = c.c*Δt
infect(c::ConstantRate, tj) = tj - log(rand())/c.c
density(c::ConstantRate, t) = c.c
shift(c::ConstantRate, tj) = c

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

Base.:*(m::MaskedRate, g::RateContinuous) = MaskedRate(m.rate*g, m.mask)
Base.:*(g::RateContinuous, m::MaskedRate) = MaskedRate(m.rate*g, m.mask)
Base.:*(m::MaskedRate, n::MaskedRate) = MaskedRate(m.rate*n.rate, m.mask ∩ n.mask)

struct UnitRate <: RateContinuous end

cumulated(::UnitRate, Δt) = Δt
infect(::UnitRate, tj) = tj - log(rand())
density(::UnitRate, t) = 1.0

Base.:*(u::UnitRate, r::RateContinuous) = r
Base.:*(r::RateContinuous, u::UnitRate) = r

nparams(::Type{UnitRate}) = 0
nparams(::Type{<: GaussianRate}) = 3
nparams(::Type{<: ConstantRate}) = 1
nparams(::Type{MaskedRate{R}}) where R = nparams(R)
