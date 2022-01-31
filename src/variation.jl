import Enzyme, ForwardDiff


export descend!, logQ



function logQ(x, M::StochasticModel)
    sum(logQi(M, i, individual(M,i), x) for i = 1:size(x,1); init=0.0)
end

function logQgen(x, M::StochasticModel, θgen)
    sum(logQi(M, i, individual(M, i, θgen), x) for i = 1:size(x,1); init=0.0)
end



function descend!(Mp, O; M = copy(Mp),
        numiters = 200, numsamples = 1000, ε = 1e-10,
        descender = AdamDescender(M.θ, 1e-3),
        hyperdescender = AdamDescender(M.θgen, 1e-3),
        θmin = 1e-5,
        θmax = 1-1e-5,
        θgenmin = 1e-5,
        θgenmax = 1-1e-5,
        learnhyper=1
    )
    
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
            Dθgen[ti] .= 0.0
            avF[ti] = 0.0
        end
        Threads.@threads for s = 1:numsamples
            ti = Threads.threadid()
            x, sample! = X[ti], S![ti]
            sample!(x)
            F = (logQ(x, M) - logQ(x, Mp) - logO(x, O, Mp)) / numsamples
            gradient!(dθ[ti], x, M)
            Dθ[ti] .+= F .* dθ[ti]
            ForwardDiff.gradient!(dθgen[ti], th->logQgen(x, M, th), θgen)
            Dθgen[ti] .+= (F .* dθgen[ti] .- ForwardDiff.gradient(th -> logQgen(x, Mp, th), θgen))
            #@show Dθgen[ti] dθgen[ti]
            avF[ti] += F
        end
        for ti = 2:nt
            any(isnan.(Dθ[ti])) && (@show sum(avF) t; return M)
            Dθ[1] .+= Dθ[ti]
            Dθgen[1] .+= Dθgen[ti]
        end
        #@show Dθgen[1]
        step!(θ, Dθ[1], descender)
        θ .= clamp.(θ, θmin, θmax) 
        if mod(t,learnhyper) == 0
            step!(θgen, Dθgen[1], hyperdescender)
            θgen .= clamp.(θgen, θgenmin, θgenmax) 
        end
        ProgressMeter.next!(pr, showvalues=[(:F,sum(avF))])
    end
    sum(avF)
end

function gradient!(dθ, x, M::StochasticModel)
    for i=1:size(dθ, 2)
        ForwardDiff.gradient!((@view dθ[:,i]), θi->logQi(M, i, individual(M, θi), x), @view M.θ[:,i])
    end
end



