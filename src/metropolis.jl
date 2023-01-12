using Random, Distributions

export UniformMove, GaussMove, metropolis_hasting

struct RectifiedGaussMove{T<:AbstractFloat}
    δ::T # for the moment, support in -Inf, Inf and clamping in 0,T
end

struct NewRectifiedGaussMove{T<:AbstractFloat}
    δ::T # for the moment, support in -Inf, Inf and clamping in 0,T
end

function propose_move!(K::RectifiedGaussMove,x,xnew,i,Mp)
    xnew[i] = clamp(rand(Normal(x[i],K.δ)),0.0,Mp.T)
end

function propose_move!(K::NewRectifiedGaussMove,x,xnew,i,Mp)
    if x[i]==0.0
        xnew[i] = min(rand(TruncatedNormal(0.0,K.δ,0,Inf)),Mp.T)
    elseif x[i]==Mp.T
        xnew[i] = max(0,rand(TruncatedNormal(Mp.T,K.δ,-Inf,Mp.T)))
    else
        xnew[i] = clamp(rand(Normal(x[i],K.δ)),0.0,Mp.T)
    end
end

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

function logdensity(K::NewRectifiedGaussMove,ti,tinew,Mp)
    if ti==0
        if tinew==Mp.T
            return log(2 * (1 - cdf(Normal(0,K.δ),Mp.T)))
        else
            return log(2*cdf(Normal(0,K.δ),Mp.T) -1 ) + logpdf(TruncatedNormal(0,K.δ,0.0,Mp.T),tinew)
        end
    elseif ti==Mp.T
        if tinew==0.0
            return log(2*cdf(Normal(Mp.T,K.δ),0.0))
        else
            return log(1 - 2*cdf(Normal(Mp.T,K.δ),0.0)) + logpdf(TruncatedNormal(Mp.T,K.δ,0.0,Mp.T),tinew)
        end

    else # ti in the interval (0,T)
        if tinew==0.0
            return logcdf( Normal(ti,K.δ), 0.0 )
        elseif tinew==Mp.T
            return log(1.0 - cdf(Normal(ti,K.δ),Mp.T))
        else
            #return logpdf(Normal(ti,K.δ),tinew) same as below line
            return log( cdf(Normal(ti,K.δ),Mp.T) - cdf(Normal(ti,K.δ),0.0) ) + logpdf(TruncatedNormal(ti,K.δ,0.0,Mp.T),tinew)
        end
    end
end

function ΔE(x,xnew,i,Mp,O) # old - new
    s = (logQi(Mp,i,individual(Mp,i),x) + logO(x, O, Mp)) - (logO(xnew, O, Mp) + logQi(Mp,i,individual(Mp,i),xnew))
    for (j,_) ∈ out_neighbors(Mp, i)
        s += logQi(Mp,j,individual(Mp,j),x) - logQi(Mp,j,individual(Mp,j),xnew)
    end
    return s

    # debug
    #s2 = (logQi(Mp,i,individual(Mp,i),x) + logO(x, O, Mp)) - (logO(xnew, O, Mp) + logQi(Mp,i,individual(Mp,i),xnew))
    #for (j,_) ∈ in_neighbors(Mp, i)
    #    s2 += logQi(Mp,j,individual(Mp,j),x) - logQi(Mp,j,individual(Mp,j),xnew)
    #end
    #@assert isapprox(s,s2)
end

function metropolis_hasting(Mp, O, K; x = prior(Mp,numsamples = 1)[:],hr=true)
    
    N = nv(Mp.G)
    T = Mp.T
    xnew = similar(x)
    acc_ratio = 0.0
    xnew = copy(x)
    for i in randperm(N)
        propose_move!(K,x,xnew,i,Mp)
        pacc = exp(-ΔE(x,xnew,i,Mp,O) + (logdensity(K,xnew[i],x[i],Mp) - logdensity(K,x[i],xnew[i],Mp))*hr )
        if rand() < pacc
            x[i] = xnew[i]
            acc_ratio+=1
        else
            xnew[i] = x[i]
        end
    end
    xnew, acc_ratio/N
end

function metropolis_sampling_sequential(Mp, O, K; numsamples = 10^3,numsteps=10,nfirst = 10^3, x = prior(Mp,numsamples = 1)[:],hastingratio=true)

    n_states(Mp)!=2 && @error "only working for SI "
    
    N = nv(Mp.G)
    stats = zeros(numsamples,N)
    pr = Progress(numsamples)

    for _ = 1:nfirst
        x,~ = metropolis_hasting(Mp,O,K;x = x,hr = hastingratio)
    end

    #start collecting samples
    acc_ratio = -1.0
    
    for m = 1:numsamples
        for _= 1:numsteps
            x, acc_ratio = metropolis_hasting(Mp,O,K;x = x,hr = hastingratio)
        end
        stats[m,:]  = copy(x)
        ProgressMeter.next!(pr,showvalues=[(:acc_ratio,acc_ratio)] )
    end
    
    return stats
end