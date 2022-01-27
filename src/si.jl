using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, TrackingHeaps

export  Sampler, GenerativeSI, InferencialSI

abstract type IndividualSI end

# For the inferencial SI
#θi = pseed,autoinf,inf
#θgen = out

struct InferencialSI{T,Rauto,Rinf,Rout}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
end

InferencialSI{Rauto, Rinf, Rout}(θi, θgen) where {Rauto, Rinf, Rout} = @views InferencialSI(θi[1], Rauto(θi[2:1+nparams(Rauto)]...), Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...), 
Rout(θgen[1:nparams(Rout)]...)
)

# For the generative SI
#θi = inf
#θgen = pseed, autoinf, out

struct GenerativeSI{T,Rauto,Rinf,Rout}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
end

GenerativeSI{Rauto, Rout}(θgen) where {Rauto, Rinf, Rout} = @views 
GenerativeSI(θgen[1], 
             Rauto(θgen[2:1+nparams(Rauto)]...), 
             UnitRate(), 
             Rout(θgen[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rout)]...)
)

#General functions for SI

compatibility(x, O, Mp::StochasticModel{<:IndividualSI}) = all((x[i,1] < t) == s for (i,s,t,p) in O)


function Sampler(M::StochasticModel{<:IndividualSI})
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



n_states(M::StochasticModel{<: IndividualSI}) = 2
trajectorysize(M::StochasticModel{<: IndividualSI}) = (nv(M.G))
