import Enzyme, ForwardDiff


export descend!, logQ


function logQi(M::StochasticModel{<:IndividualSEIR}, i, ind, x::Matrix{Float64})     #x[i] = (tE, tI, tR)
    s = 0.0
    if iszero(x[i,1])
        s += log(ind.pseed)
    else
        s += log(1-ind.pseed)
        s -= cumulated(ind.autoinf, x[i,1])
        sSE = density(ind.autoinf, x[i,1])
        for (j,rji) ∈ in_neighbors(M, i)
            if x[j,2] < x[i,1] 
                inf = ind.inf * rji * shift(individual(M,j).out,x[j,2])
                s -= cumulated(inf, min(x[i,1],x[j,3])) - cumulated(inf, x[j,2])
                if x[i,1] < x[j,3]
                    sSE += density(inf,x[i,1])
                end
            end
        end
        if x[i,1] < M.T
            s += log(sSE)
        end
    end
    lat = shift(ind.lat_delay, x[i,1]) * ind.latency
    s -= cumulated(lat, x[i,2]) - cumulated(lat, x[i,1])
    if x[i,2] < M.T
        s += log(density(lat, x[i,2])) 
    end
    rec = shift(ind.recov_delay, x[i,2]) * ind.recov
    s -= cumulated(rec, x[i,3]) - cumulated(rec, x[i,2]) 
    if x[i,3] < M.T
        s += log(density(rec, x[i,3])) 
    end
    return s
end


function logQi(M::StochasticModel{<:IndividualSI}, i, ind, x)
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
    sum(logQi(M, i, individual(M,i), x) for i = 1:size(x,1); init=0.0)
end

function logQgen(x, M::StochasticModel, θgen)
    sum(logQi(M, i, individual(M, i, θgen), x) for i = 1:size(x,1); init=0.0)
end

logO(x, O, M::StochasticModel{<:IndividualSI}) = sum(log(p + ((x[i] < t) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)

logO(x, O, M::StochasticModel{<:IndividualSEIR}) = sum(log(p + ((x[i,2] < t < x[i,3]) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)

function descend!(Mp, O; M = copy(Mp),
        numiters = 200, numsamples = 1000, ε = 1e-10,
        descender = AdamDescender(M.θ, 1e-3),
        θmin = 1e-5,
        θmax = 1-1e-5,
        θgenmin = 1e-5,
        θgenmax = 1-1e-5)
    
    number_of_states = n_states(M) 
    N = nv(M.G)
    nt = Threads.nthreads()
    X = [zeros(N, number_of_states - 1) for ti=1:nt]
    dθ = [zero(M.θ) for ti=1:nt]
    dθgen = [zero(M.θgen) for ti=1:nt]
    Dθ = [zero(M.θ) for ti=1:nt]
    Dθgen = [zero(M.θgen) for ti=1:nt]
    avF = zeros(nt)
    S! = [Sampler(M) for i=1:nt]
    pr = Progress(numiters)
    θ = M.θ
    θgen = M.θgen
    for t = 1:numiters
        for ti=1:nt
            Dθ[ti] .= 0.0
            avF[ti] = 0.0
        end
        Threads.@threads for s = 1:numsamples
            ti = Threads.threadid()
            x, sample! = X[ti], S![ti]
            sample!(x)
            F = (logQ(x, M) - logQ(x, Mp) - logO(x, O, Mp)) / numsamples
            gradient!(dθ[ti], x, M)
            ForwardDiff.gradient!(dθgen[ti], th->logQgen(x, M, th), M.θgen)
            Dθ[ti] .+= F .* dθ[ti]
            Dθgen[ti] .+= F .* dθgen[ti]
            avF[ti] += F
        end
        for ti = 2:nt
            any(isnan.(Dθ[ti])) && (@show sum(avF) t; return M)
            Dθ[1] .+= Dθ[ti]
            Dθgen[1] .+= Dθgen[ti]
        end        
        step!(θ, Dθ[1], descender)
        step!(θgen, Dθgen[1], descender)
        θ .= clamp.(θ, θmin, θmax) 
        θgen .= clamp.(θgen, θgenmin, θgenmax) 
        ProgressMeter.next!(pr, showvalues=[(:F,sum(avF))])
    end
    sum(avF)
end

function gradient!(dθ, x, M::StochasticModel)
    for i=1:size(dθ, 2)
        ForwardDiff.gradient!((@view dθ[:,i]), θi->logQi(M, i, individual(M, θi), x), @view M.θ[:,i])
    end
end



