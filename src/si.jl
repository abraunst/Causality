using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, TrackingHeaps

export  Sampler, GenerativeSI, InferentialSI

# For the SI
#θi = pseed,autoinf, inf_in
#θgen = pseed, autoinf, inf_out

abstract type SI end

struct IndividualSI{P,Rauto,Rinf,Rout}
    pseed::P
    autoinf::Rauto
    inf::Rinf
    out::Rout
end


struct InferentialSI{Rauto, Rinf, Rout} <: SI end

@individual InferentialSI{Rauto, Rinf, Rout}(θi, θgen) =
    IndividualSI(θi,
        Rauto(θi),
        Rinf(θi),
        Rout(θgen))


struct GenerativeSI{Rauto, Rout} <: SI end

@individual GenerativeSI{Rauto, Rout}(θi, θgen) =
    IndividualSI(θgen,
        Rauto(θgen),
        UnitRate(),
        Rout(θgen))

#General functions for SI

compatibility(x, O, Mp::StochasticModel{<:SI}) = all((x[i,1] < t) == s for (i,s,t,p) in O)


function Sampler(M::StochasticModel{<:SI})
    N::Int = nv(M.G)
    s::BitVector = falses(N)
    Q::TrackingHeap{Int, Float64, 2, MinHeapOrder, NoTrainingWheels} = TrackingHeap(Float64, S=NoTrainingWheels)
    function sample!(x)
        @assert N == length(x)
        empty!(Q)
        x .= M.T
        s .= false
        for i in eachindex(x)
            ind = individual(M, i)
            Q[i] = min(M.T, rand() < ind.pseed ? zero(M.T) : delay(ind.autoinf, zero(M.T)))
        end
        while !isempty(Q)
            i, t = pop!(Q)
            s[i] = true
            x[i] = t
            for (j,rij) ∈ out_neighbors(M,i)
                if !s[j]
                    Q[j] = min(Q[j], delay(shift(individual(M,i).out,t) * rij * individual(M,j).inf, t))
                end
            end
        end
        return x
    end
end


function logQi(M::StochasticModel{<:SI}, i, ind, x)
    iszero(x[i]) && return log(ind.pseed)
    s = log(1-ind.pseed)
    s -= cumulated(ind.autoinf, x[i])
    s2 = density(ind.autoinf, x[i])
    for (j,rji) ∈ in_neighbors(M, i)
        if x[j] < x[i]
            inf = ind.inf * rji * shift(ind.out,x[j])  # we use ind.out because all the out are the same
            s -= cumulated(inf, x[i]) - cumulated(inf, x[j])
            s2 += density(inf, x[i])
        end
    end
    if x[i] < M.T
        s += log(s2)
    end
    return s
end

#function logO(x, O, M::StochasticModel{<:SI}) 
 #   sum(log(p + ((x[i] < t) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)
#end


function logO(x, O, M::StochasticModel{<:SI})
    su = 0.
    for (i,s,t,p) in O
        if s == 0 
            if t < x[i]
                su += 0.
            else
                su += x[i] - t
            end
        elseif s == 1
            if t > x[i]
                su += 0.
            else
                su += t - x[i]
            end
        end
    end
    su
end


n_states(M::StochasticModel{<:SI}) = 2
trajectorysize(M::StochasticModel{<:SI}) = (nv(M.G))