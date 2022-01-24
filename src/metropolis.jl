using Random, Distributions

export UniformMove, GaussMove, metropolis_hasting

struct UniformMove{T<:AbstractFloat}
    δ::T
end

struct GaussMove{T<:AbstractFloat}
    σ::T
end

function propose_move(K::UniformMove,x,T)
    a = max(0,x-K.δ)
    rand()*(min(x+K.δ,T)-a)+a
end

function logdensity(K::UniformMove,μ,x,T)
    -log((min(μ+K.δ,T)-max(0,μ-K.δ)))
end

function propose_move(K::GaussMove,x,T)
    rand(TruncatedNormal(x,K.σ,0.0,T))
end

function logdensity(K::GaussMove,μ,x,T)
    logpdf(TruncatedNormal(μ,K.σ,0.0,T),x)
end


function metropolis_hasting(Mp, O, K; numsteps=10^3,nfirst = numsteps,x = post(Mp,O,numsamples = 1)[:])
    
    N = nv(Mp.G)
    T = Mp.T
    xnew = similar(x)
    #stats = copy(x)
    stats = zeros(N,numsteps)
    perm = collect(1:N)

    @showprogress for m=1:(numsteps+nfirst)
        
        randperm!(perm)
        xnew = copy(x)
        for i in perm
            xnew[i] = propose_move(K,x[i],T)
            if rand() < exp(-ΔE(x,xnew,i,Mp,O) + logdensity(K,xnew[i],x[i],T) - logdensity(K,x[i],xnew[i],T) )
                x = copy(xnew)
            end
        end
        if m > nfirst
            stats[:,m-nfirst] = xnew
        end
    end
    stats
end

function ΔE(x,xnew,i,Mp,O)
    s = (logQi(Mp,i,individual(Mp,i),x) + logO(x, O, Mp)) - (logO(xnew, O, Mp) + logQi(Mp,i,individual(Mp,i),xnew))
    for (j,_) ∈ in_neighbors(Mp, i)
        s += logQi(Mp,j,individual(Mp,j),x) - logQi(Mp,j,individual(Mp,j),xnew)
    end
    s
end


