import Enzyme, ForwardDiff


export descend!, logQ


function logQi(M, i, ind, x)
    iszero(x[i]) && return log(ind.pseed)
    s = log(1-ind.pseed)
    s -= cumulated(ind.autoinf, x[i])
    s2 = density(ind.autoinf, x[i])
    for (j,rji) ∈ in_neighbors(M, i)
        if x[j] < x[i]
            inf = ind.inf * rji * shift(individual(M,j).out,x[j])
            s -= cumulated(inf, x[i]) - cumulated(inf, x[j])
            s2 += density(inf, x[i])
        end
    end
    if x[i] < M.T
        s += log(s2)
    end
    return s
end


function logQ(x, M::StochasticModel)
    sum(logQi(M, i, individual(M,i), x) for i in eachindex(x); init=0.0)
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
    sum(avF)
end


function gradient!(dθ, x, M::StochasticModel)
    for i=1:size(dθ, 2)
        # Enzyme.autodiff(θi->logQi(M, i, individual(M, θi), x), Enzyme.Duplicated((@view M.θ[:,i]), (@view dθ[:,i])))
        ForwardDiff.gradient!((@view dθ[:,i]), θi->logQi(M, i, individual(M, θi), x), @view M.θ[:,i])
    end
end



