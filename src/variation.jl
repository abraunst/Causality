import Enzyme, ForwardDiff


export descend!, logQ, localdescend!



function logQ(x, M::StochasticModel)
    sum(logQi(M, i, individual(M,i), x) for i = 1:size(x,1); init=0.0)
end


function logQcorr(x, M::StochasticModel)
    N = nv(M.G)
    logQ(x,M) - log(1 - prod(1 - individual(M,i).pseed for i = 1:N))
end


function logQgen(x, M::StochasticModel, θgen)
    sum(logQi(M, i, individual(M, i, @view(M.θ[:,i]), θgen), x) for i = 1:size(x,1); init=0.0)
end



function descend!(Mp, O; M = copy(Mp),
        numiters = 200, numsamples = 1000, ε = 1e-10,
        descender = AdamDescender(M.θ, 1e-3),
        hyperdescender = AdamDescender(M.θgen, 1e-3),
        θmin = 1e-5,
        θmax = 1-1e-5,
        θgenmin = 1e-5,
        θgenmax = 1-1e-5,
        hyper_mask = []  #put here the indices of the hyperparameters NOT to learn
    )
    
    number_of_states = n_states(M) 
    N = nv(M.G)
    nt = Threads.nthreads()
    X = [zeros(N, number_of_states - 1) for ti=1:nt]
    dθ = [zero(M.θ) for ti=1:nt]
    dθgen = [zero(M.θgen) for ti=1:nt]
    Dθ = [zero(M.θ) for ti=1:nt]
    Dθgen = [zero(M.θgen) for ti=1:nt]
    Obs = [[o for o in O if o[1] == i ] for i = 1:N]  #observations grouped for particle number
    avF = zeros(nt)
    S! = [Sampler(M) for i=1:nt]
    pr = Progress(numiters)
    θ = M.θ
    θgen = M.θgen
    #@show Threads.nthreads()
    for t = 1:numiters
        for ti=1:nt
            Dθ[ti] .= 0.0
            Dθgen[ti] .= 0.0
            avF[ti] = 0.0
        end
        a = prod([1 - individual(M,i).pseed for i = 1:N])
        b = a / (1-a)
        Threads.@threads for s = 1:numsamples
            ti = Threads.threadid()
            x, sample! = X[ti], S![ti]
            sample!(x)
            F1 = logQcorr(x, M) - logQ(x, Mp) 
            gradient!(dθ[ti], x, M)
            F = (F1 - logO(x, O, Mp)) / numsamples
            avF[ti] += F
            for i = 1:N
                Dθ[ti][:,i] .+= F .* dθ[ti][:,i] 
                Dθ[ti][1,i] -= F * b/(1 - individual(M,i).pseed)
            end
            ForwardDiff.gradient!(dθgen[ti], th->logQgen(x, M, th), θgen)
            Dθgen[ti] .+= (F .* dθgen[ti] .- ForwardDiff.gradient(th -> logQgen(x, Mp, th), θgen))       
        end
        for ti = 2:nt
            any(isnan.(Dθ[ti])) && (@show sum(avF) t; return M)
            Dθ[1] .+= Dθ[ti]
            Dθgen[1] .+= Dθgen[ti]
        end
        #@show Dθ[1][:,1] 
        step!(θ, Dθ[1], descender)
        θ .= clamp.(θ, θmin, θmax) 
        Dθgen[1][hyper_mask] .= 0
        step!(θgen, Dθgen[1], hyperdescender)
        θgen .= clamp.(θgen, θgenmin, θgenmax) 
        ProgressMeter.next!(pr, showvalues=[(:F,sum(avF))])
    end
    sum(avF)
end


function localdescend!(Mp, O; M = copy(Mp),
        numiters = 200, numsamples = 1000, ε = 1e-10,
        descender = AdamDescender(M.θ, 1e-3),
        hyperdescender = AdamDescender(M.θgen, 1e-3),
        θmin = 1e-5,
        θmax = 1-1e-5,
        θgenmin = 1e-5,
        θgenmax = 1-1e-5,
        hyper_mask = []  #put here the indices of the hyperparameters NOT to learn
    )
    
    number_of_states = n_states(M) 
    N = nv(M.G)
    nt = Threads.nthreads()
    X = [zeros(N, number_of_states - 1) for ti=1:nt]
    dθ = [zero(M.θ) for ti=1:nt]
    dθgen = [zero(M.θgen) for ti=1:nt]
    Dθ = [zero(M.θ) for ti=1:nt]
    Dθgen = [zero(M.θgen) for ti=1:nt]
    Obs = [[o for o in O if o[1] == i ] for i = 1:N]  #observations grouped for particle number
    avF = zeros(nt)
    S! = [Sampler(M) for i=1:nt]
    pr = Progress(numiters)
    Floc = zeros(N)
    θ = M.θ
    θgen = M.θgen
    for t = 1:numiters
        for ti=1:nt
            Dθ[ti] .= 0.0
            Dθgen[ti] .= 0.0
            avF[ti] = 0.0
        end
        a = prod([1 - individual(M,i).pseed for i = 1:N])
        b = a / (1-a)
        Threads.@threads for s = 1:numsamples
            ti = Threads.threadid()
            x, sample! = X[ti], S![ti]
            sample!(x)
            logobs = logO(x, O, Mp) / numsamples
            F = (logQ(x, M) - logQ(x, Mp)) / numsamples - logobs
            gradient!(dθ[ti], x, M)
            avF[ti] += F 
            for i = 1:N
               Floc[i] = (logQi(M, i, individual(M,i), x) - logQi(Mp, i, individual(Mp,i), x) ) / numsamples 
            end
            for i = 1:N
                Fi = (Floc[i] + sum([Floc[j] for (j,rij) ∈ in_neighbors(M,i)]) - logobs) 
                Dθ[ti][:,i] .+= Fi .* dθ[ti][:,i]
                Dθ[ti][1,i] -= Fi * b/(1 - individual(M,i).pseed)
            end
            ForwardDiff.gradient!(dθgen[ti], th->logQgen(x, M, th), θgen)
            Dθgen[ti] .+= (F .* dθgen[ti] .- ForwardDiff.gradient(th -> logQgen(x, Mp, th), θgen))       
        end
        for ti = 2:nt
            any(isnan.(Dθ[ti])) && (@show sum(avF) t; return M)
            Dθ[1] .+= Dθ[ti]
            Dθgen[1] .+= Dθgen[ti]
        end
        #@show Dθ[1][:,1] 
        step!(θ, Dθ[1], descender)
        θ .= clamp.(θ, θmin, θmax) 
        Dθgen[1][hyper_mask] .= 0
        step!(θgen, Dθgen[1], hyperdescender)
        θgen .= clamp.(θgen, θgenmin, θgenmax) 
        ProgressMeter.next!(pr, showvalues=[(:F,sum(avF))])
    end
    sum(avF)
end

function gradient!(dθ, x, M::StochasticModel)
    for i=1:size(dθ, 2)
        ForwardDiff.gradient!((@view dθ[:,i]), θi->logQi(M, i, individual(M, i, θi), x), @view M.θ[:,i])
    end
end



