import ForwardDiff


export descend!, logQ


function logQi(x, i, Λ, T, pseed, autoinf, inf)
    iszero(x[i]) && return log(pseed)
    s = zero(pseed)
    s += log(1-pseed)
    s -= cumulated(autoinf, x[i])
    s2 = density(autoinf, x[i])
    ci = cumulated(inf, x[i])
    di = density(inf, x[i])
    for j ∈ ∂(Λ,i)
        if x[i] > x[j]
            s -= ci - cumulated(inf, x[j])
            s2 += di
        end
    end
    return x[i] < T ? s + log(s2) : s
end


function logQ(x, M::StochasticModel)
    sum(logQi(x, i, M.Λ, M.T, M.pseed[i], M.autoinf[i], M.inf[i]) for i in eachindex(x))
end


logO(x, O) = sum(log(p + ((x[i] < t) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)

function descend!(Mp, O; M = copy(Mp),
        numiters = 200, numsamples = 1000, ε = 1e-10,
        descender = AdamDescender(M.θ, 1e-3),
        θmin = 1e-5,
        θmax = 1-1e-5)
    N = size(M.Λ,2)
    nt = Threads.nthreads()
    X = [zeros(N) for ti=1:nt]
    dθ = [zero(M.θ) for ti=1:nt]
    Dθ = [zero(M.θ) for ti=1:nt]
    avF = zeros(nt)
    S! = [Sampler(M) for i=1:nt]
    pr = Progress(numiters)
    θ = M.θ

    for t = 1:numiters
        for ti=1:nt
            Dθ[ti] .= 0.0
            avF[ti] = 0.0
        end
        Threads.@threads for s = 1:numsamples
            ti = Threads.threadid()
            x, sample! = X[ti], S![ti]
            sample!(x)
            F = (logQ(x, M) - logQ(x, Mp) - logO(x, O)) / numsamples
            #!isfinite(F) && return x
            gradient!(dθ[ti], x, M)
            Dθ[ti] .+= F .* dθ[ti]
            avF[ti] += F
        end
        #@show Dθ[1]
        for ti = 2:nt
            Dθ[1] .+= Dθ[ti]
        end
        step!(θ, Dθ[1], descender)
        θ .= clamp.(θ, θmin, θmax)
        any(isnan.(θ)) && (@show t; return M)
        ProgressMeter.next!(pr, showvalues=[(:F,sum(avF))])
    end
    M
end


function gradient!(dθ, x, M::StochasticModel{TT,TH,G,A,B,C}) where {TT,TH,G,A,B<:AbstractVector{<:GaussianRate},C<:AbstractVector{<:GaussianRate}}
    for i=1:size(dθ, 2)
        ForwardDiff.gradient!((@view dθ[:,i]), v->logQi(x, i, M.Λ, M.T, v[1], GaussianRate(v[2:4]...), GaussianRate(v[5:7]...)), M.θ[:,i])
    end
end



