using SpecialFunctions, IntervalUnionArithmetic

export ConstantRate, GaussianRate, MaskedRate, UnitRate


abstract type RateContinuous end

nparams(::Type{R}) where R = length(R.types)

####### GaussianRate

struct GaussianRate{T} <: RateContinuous
    a::T
    b::T
    c::T
end

function density(g::GaussianRate, t)
    a, b, c = g.a, g.b, g.c
    a*exp(-((t-b)/c)^2)
end

logdensity(g::RateContinuous, t) = log(density(g, t))

logdensity(g::GaussianRate, t) = -((t-g.b)/g.c)^2 + log(g.a)

function cumulated(g::GaussianRate, t)
    a, b, c = g.a, g.b, g.c
    a*c*0.5*sqrt(π)*(erfc(-(t-b)/c)-erfc(b/c))
end

function delay(g::GaussianRate, tj)
    a, b, c = g.a, g.b, g.c
    y=2/(a*c*sqrt(π))*(cumulated(g, tj) - log(rand())) + erfc(b/c);
    if y > 2
        return Inf;
    end
    return -c * erfcinv(y) + b
end

function Base.:*(g1::GaussianRate, g2::GaussianRate)
    a1, b1, c1 = g1.a, g1.b, g1.c
    a2, b2, c2 = g2.a, g2.b, g2.c
    peak = a1 * a2 * exp(- (b1 - b2)^2 / (c1^2 + c2^2))
    avg  = (b1 * c2^2 + b2 * c1^2)/(c1^2+c2^2)
    var  = (c1^2 * c2^2) / (c1^2 + c2^2)
    GaussianRate(peak,avg,sqrt(var))
end

shift(g::GaussianRate, tj) = GaussianRate(g.a,g.b+tj,g.c)

nparams(::Type{GaussianRate}) = 3


#### ConstantRate

struct ConstantRate{T} <: RateContinuous
    c::T
end

Base.:*(g1::ConstantRate, g2::ConstantRate) = ConstantRate(g1.c*g2.c)
Base.:*(g1::ConstantRate{T}, g2::GaussianRate) where T = GaussianRate{T}(g1.c*g2.a,g2.b,g2.c)
Base.:*(g2::GaussianRate, g1::ConstantRate) = g1*g2
cumulated(c::ConstantRate, Δt) = c.c*Δt
delay(c::ConstantRate, tj) = tj - log(rand())/c.c
density(c::ConstantRate, t) = c.c
shift(c::ConstantRate, tj) = c

nparams(::Type{ConstantRate}) = 1

#### MaskedRate

struct MaskedRate{R} <: RateContinuous
    rate::R
    mask::IntervalUnion{Float64}
end


density(m::MaskedRate, t) = t ∈ m.mask ? density(m.rate, t) : 0.0

logdensity(m::MaskedRate, t) = t ∈ m.mask ? logdensity(m.rate, t) : -Inf

function cumulated(m::MaskedRate, t)
    s = 0.0
    for ab in m.mask.v
        if right(ab) < t
            s += cumulated(m.rate, max(0,right(ab))) - cumulated(m.rate, max(0,left(ab)))
        elseif left(ab) < t
            s += cumulated(m.rate, t) - cumulated(m.rate, left(ab))
        else
            break
        end
    end
    return s
end

function delay(m::MaskedRate, tj)
    for ab in m.mask.v
        tj > right(ab) && continue
        t = delay(m.rate, max(left(ab),tj))
        t ∈ ab  && return t
    end
    return Inf
end

Base.:*(m::MaskedRate, g::RateContinuous) = MaskedRate(m.rate*g, m.mask)
Base.:*(g::RateContinuous, m::MaskedRate) = MaskedRate(m.rate*g, m.mask)
Base.:*(m::MaskedRate, n::MaskedRate) = MaskedRate(m.rate*n.rate, m.mask ∩ n.mask)
nparams(::Type{MaskedRate{R}}) where R = nparams(R)


####### UnitRate

struct UnitRate <: RateContinuous end

cumulated(::UnitRate, Δt) = Δt
delay(::UnitRate, tj) = tj - log(rand())
density(::UnitRate, t) = 1.0
logdensity(::UnitRate, t) = 0.0

Base.:*(u::UnitRate, r::RateContinuous) = r
Base.:*(r::RateContinuous, u::UnitRate) = r
Base.:*(::UnitRate,::UnitRate) = UnitRate()
Base.:*(m::MaskedRate, n::UnitRate) = m
nparams(::Type{UnitRate}) = 0
nparams(::Type{<: GaussianRate}) = 3
nparams(::Type{<: ConstantRate}) = 1
nparams(::Type{MaskedRate{R}}) where R = nparams(R)

struct StepRate{R, T} <: RateContinuous
    rate::R
    center::T
    width::T    
end
l_ext(s::StepRate) = s.center - s.width/2
r_ext(s::StepRate) = s.center + s.width/2

density(s::StepRate, t) = l_ext(s) < t < r_ext(s) ? density(s.rate, t) : 0.0
logdensity(s::StepRate, t) = l_ext(s) < t < r_ext(s) ? logdensity(s.rate, t) : -Inf
function cumulated(s::StepRate, t)
    sum = 0.0
    if r_ext(s) < t
        return ( cumulated(s.rate, max(0, r_ext(s))) - cumulated(s.rate, max(0,l_ext(s))) )
    elseif l_ext(s) < t
        return ( cumulated(s.rate, t) - cumulated(s.rate, max(0,l_ext(s))) )
    end
    return sum
end

function delay(s::StepRate, tj)
    tj > r_ext(s) && return Inf
    t = delay(s.rate, max(l_ext(s),tj))
    l_ext(s) < t < r_ext(s)  && return t
    return Inf
end

mask(s::StepRate) =  IntervalUnion(l_ext(s), r_ext(s))

Base.:*(s::StepRate, g::RateContinuous) = StepRate(s.rate * g, s.center, s.width)
Base.:*(g::GaussianRate, s::StepRate) = StepRate(s.rate * g, s.center, s.width)
Base.:*(s::StepRate, g::GaussianRate) = g * s
Base.:*(s::StepRate, n::MaskedRate) = MaskedRate(s.rate * n.rate, n.mask ∩ mask(s))

shift(s::StepRate, tj) = StepRate(s.a, s.center+tj, s.width)
Base.:*(s::StepRate, n::UnitRate) = s
nparams(::Type{StepRate{R,T}}) where {R,T} = nparams(R) + 2