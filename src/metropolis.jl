using Random, Distributions

export UniformMove, GaussMove, metropolis_hasting

struct UniformMove{T<:AbstractFloat}
    δ::T
end

struct GaussMove{T<:AbstractFloat}
    δ::T
end

function propose_move!(K::UniformMove,x,xnew,i,Mp)
    a = max(0,x[i]-K.δ)
    xnew[i] = rand()*(min(x[i]+K.δ,Mp.T)-a)+a
end

function propose_move!(K::GaussMove,x,xnew,i,Mp)
    xnew[i] = rand(TruncatedNormal(x[i],K.δ^2,0.0,Mp.T))
end

logdensity(K::UniformMove,μ,x,T) = -log((min(μ+K.δ,T)-max(0,μ-K.δ)))
logdensity(K::GaussMove,μ,x,T) = logpdf(TruncatedNormal(μ,K.δ^2,0.0,T),x)

function ΔE(x,xnew,i,Mp,O)
    s = (logQi(Mp,i,individual(Mp,i),x) + logO(x, O, Mp)) - (logO(xnew, O, Mp) + logQi(Mp,i,individual(Mp,i),xnew))
    for (j,_) ∈ in_neighbors(Mp, i)
        s += logQi(Mp,j,individual(Mp,j),x) - logQi(Mp,j,individual(Mp,j),xnew)
    end
    s
end

function metropolis_hasting_mc(Mp, O, K; numsteps=10^3,x = post(Mp,O,numsamples = 1)[:])
    
    N = nv(Mp.G)
    T = Mp.T
    xnew = similar(x)
    acc_ratio = 0.0
    @showprogress for m=1:numsteps

        xnew = copy(x)
        i = rand(1:N)
        propose_move!(K,x,xnew,i,Mp)
        if rand() < exp(-ΔE(x,xnew,i,Mp,O) + logdensity(K,xnew[i],x[i],T) - logdensity(K,x[i],xnew[i],T) )
            x = copy(xnew)
            acc_ratio+=1
        end
    end
    xnew, acc_ratio/numsteps
end

function metropolis_sampling_parallel(Mp, O, K; numsamples = 10^3,numsteps=10^3)

    N = nv(Mp.G)
    nt = Threads.nthreads()
    stats = [Vector{Float64}(undef,0) for t=1:nt]

    #pr = Progress(numsamples)

    #ProgressMeter.update!(pr,0)
    #jj = Threads.Atomic{Int}(0)
    #l = Threads.SpinLock()

    Threads.@threads for m = 1:numsamples
        ti = Threads.threadid()
        append!(stats[ti],metropolis_hasting_mc(Mp,O,K;numsteps = numsteps)[1])
        
        #Threads.atomic_add!(jj, 1)
        #Threads.lock(l)#acc_vec = zeros(length(δvec),numsamples)
        #ProgressMeter.update!(pr, jj[])
        #Threads.unlock(l) 
    end
    
    return collect(reshape(vcat(stats...),(N,numsamples))')
end


function metropolis_sampling_sequential(Mp, O, K; numsamples = 10^3,numsteps=10^3,nfirst = 10^3, x = post(Mp,O,numsamples = 1)[:])

    n_states(Mp)!=2 && @error "only working for SI "
    
    N = nv(Mp.G)
    stats = zeros(numsamples,N)
    pr = Progress(numsamples)

    x,~ = metropolis_hasting_mc(Mp,O,K;numsteps = nfirst,x = x)

    for m = 1:numsamples
        x, acc_ratio = metropolis_hasting_mc(Mp,O,K;numsteps = numsteps,x = x)
        stats[m,:]  = x
        ProgressMeter.next!(pr,showvalues=[(:acc_ratio,acc_ratio)] )
    end
    
    stats
end


