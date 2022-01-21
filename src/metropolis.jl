using Random, Distributions

export UniformMove, GaussMove, metropolis_hasting

struct UniformMove{T<:AbstractFloat}
    δ::T
end

struct GaussMove{T<:AbstractFloat}
    σ::T
end

function propose_move(K::UniformMove,xi,T)
    a = (xi-K.δ)>=0 ? xi-K.δ : 0.0
    b = (xi+K.δ)<=T ? xi+K.δ : T
    rand()*(b-a)+a
end

function propose_move(K::GaussMove,xi,T)
    rand(TruncatedNormal(xi,K.σ,0.0,T))    
end


function metropolis_hasting(Mp, O, K; numsteps=10^3,nfirst = numsteps,x = post(Mp,O,numsamples = 1)[:])
    
    N = nv(Mp.G)
    T = Mp.T
    xnew = similar(x)
    stats = copy(x)
    
    perm = collect(1:N)

    @showprogress for m=1:(numsteps+nfirst)
        
        randperm!(perm)
        for i in perm
            xnew .= copy(x)
            xnew[i] = propose_move(K,x[i],T)
            if rand() < exp(-ΔE(x,xnew,i,Mp,O)) 
                x = copy(xnew)
                if m > nfirst
                    stats = hcat(stats,xnew)
                end
            end
        end
    end
    stats[:,2:end]
end

function ΔE(x,xnew,i,Mp,O)
    s = (logQi(Mp,i,individual(Mp,i),x) + logO(x, O, Mp)) - (logO(xnew, O, Mp) + logQi(Mp,i,individual(Mp,i),xnew))
    for (j,_) ∈ in_neighbors(Mp, i)
        s += logQi(Mp,j,individual(Mp,j),x) - logQi(Mp,j,individual(Mp,j),xnew)
    end
    s
end


