using Random, Distributions

export UniformMove, GaussMove, metropolis_hasting

struct UniformMove{T<:AbstractFloat}
    δ::T
    l::T
    r::T
end

struct GaussMove{T<:AbstractFloat}
    δ::T
    l::T
    r::T
end

struct RectifiedGaussMove{T<:AbstractFloat}
    δ::T # for the moment, support in -Inf, Inf and clamping in 0,T
end


UniformMove(δ) = UniformMove(δ,-Inf,Inf)
GaussMove(δ) = GaussMove(δ,-Inf,Inf)

function propose_move!(K::UniformMove,x,xnew,i,Mp)
    #xmin = max(K.lb,x[i]-K.δ)
    #xnew[i] = rand()*(min(x[i]+K.δ,K.ub)-xmin)+xmin
    xnew[i] = clamp(rand(Uniform(max(K.l,x[i]-K.δ), min(x[i]+K.δ,K.r))),0.0,Mp.T)
end

function propose_move!(K::GaussMove,x,xnew,i,Mp)
    xnew[i] = clamp(rand(TruncatedNormal(x[i],K.δ,K.l,K.r)),0.0,Mp.T)
end


function propose_move!(K::RectifiedGaussMove,x,xnew,i,Mp)
    xnew[i] = clamp(rand(Normal(x[i],K.δ)),0.0,Mp.T)
end


logdensity(K::UniformMove,μ,x,Mp) = logpdf(Uniform(max(K.l,μ-K.δ), min(μ+K.δ,K.r)),x)
logdensity(K::GaussMove,μ,x,Mp) = logpdf(TruncatedNormal(μ,K.δ,K.l,K.r),x)

function logdensity(K::RectifiedGaussMove,μ,x,Mp)
    # what if I have to compute the ratio between an integral and a density?
    if x==0 # integral from -infty to 0
        return logcdf( Normal(μ,K.δ) , 0.0 )
    elseif x==Mp.T
        return log(1.0 - cdf(Normal(μ,K.δ),Mp.T))
    else
        return logpdf(Normal(μ,K.δ),x)
    end
end


function ΔE(x,xnew,i,Mp,O)
    s = (logQi(Mp,i,individual(Mp,i),x) + logO(x, O, Mp)) - (logO(xnew, O, Mp) + logQi(Mp,i,individual(Mp,i),xnew))
    for (j,_) ∈ in_neighbors(Mp, i)
        s += logQi(Mp,j,individual(Mp,j),x) - logQi(Mp,j,individual(Mp,j),xnew)
    end
    s
end

function initial_condition(Mp,O;x0type=:post)
    x0type==:post && return post(Mp,O;numsamples=1)[:]
    x0type==:prior && return prior(Mp;numsamples=1)[:]
    x0type==:rand && return rand(nv(Mp.G))*Mp.T
end

function metropolis_hasting_mc(Mp, O, K; numsteps=10^3,x = prior(Mp,numsamples = 1)[:],hr=true) #deprecated for now
    
    N = nv(Mp.G)
    T = Mp.T
    xnew = similar(x)
    acc_ratio = 0.0
    for m=1:numsteps

        xnew = copy(x)
        i = rand(1:N)
        propose_move!(K,x,xnew,i,Mp)
        if rand() < exp(-ΔE(x,xnew,i,Mp,O) + (logdensity(K,xnew[i],x[i],Mp) - logdensity(K,x[i],xnew[i]),Mp)*hr )
            x = copy(xnew)
            acc_ratio+=1
        end
    end
    xnew, acc_ratio/numsteps
end


function metropolis_hasting_fullswipe(Mp, O, K; x = prior(Mp,numsamples = 1)[:],hr=true)
    
    N = nv(Mp.G)
    T = Mp.T
    xnew = similar(x)
    acc_ratio = 0.0
    xnew = copy(x)
    for i in randperm(N)
        propose_move!(K,x,xnew,i,Mp)
        if rand() < exp(-ΔE(x,xnew,i,Mp,O) + (logdensity(K,xnew[i],x[i],Mp) - logdensity(K,x[i],xnew[i],Mp))*hr )
            x[i] = xnew[i]
            acc_ratio+=1
        else
            xnew[i] = x[i]
        end
    end
    xnew, acc_ratio/N
end



function metropolis_sampling_parallel(Mp, O, K; numsamples = 10^3,numsteps=10^3,hastingratio=true,x0type=:prior) #deprecated

    N = nv(Mp.G)
    nt = Threads.nthreads()
    stats = [Vector{Float64}(undef,0) for t=1:nt]

    pr = Progress(numsamples)

    ProgressMeter.update!(pr,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()

    Threads.@threads for m = 1:numsamples
        ti = Threads.threadid()
        append!(stats[ti],metropolis_hasting_mc(Mp,O,K;numsteps = numsteps,hr=hastingratio,x = initial_condition(Mp,O;x0type = x0type))[1])
        
        Threads.atomic_add!(jj, 1)
        Threads.lock(l) #acc_vec = zeros(length(δvec),numsamples)
        ProgressMeter.update!(pr, jj[])
        Threads.unlock(l) 
    end
    
    return collect(reshape(vcat(stats...),(N,numsamples))')
end


function metropolis_sampling_sequential(Mp, O, K; numsamples = 10^3,numsteps=10,nfirst = 10^3, x = prior(Mp,numsamples = 1)[:],hastingratio=true)

    n_states(Mp)!=2 && @error "only working for SI "
    
    N = nv(Mp.G)
    stats = zeros(numsamples,N)
    pr = Progress(numsamples)

    for _ = 1:nfirst
        x,~ = metropolis_hasting_fullswipe(Mp,O,K;x = x,hr = hastingratio)
    end

    #start collecting samples
    acc_ratio = -1.0
    
    for m = 1:numsamples
        for _= 1:numsteps
            x, acc_ratio = metropolis_hasting_fullswipe(Mp,O,K;x = x,hr = hastingratio)
        end
        stats[m,:]  = copy(x)
        ProgressMeter.next!(pr,showvalues=[(:acc_ratio,acc_ratio)] )
    end
    
    return stats
end


