using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, TrackingHeaps

export  GenerativeSEIR, InferencialSEIR

abstract type IndividualSEIR end

# For the inferencial SEIR
#θi = pseed,autoinf,inf,latency,recovery
#θgen = out, lat_delay, recov_delay

struct InferencialSEIR{T,Rauto,Rinf,Rout,Rlat,Rrec,Rgenlat,Rgenrec} <: IndividualSEIR
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
    latency::Rlat
    lat_delay::Rgenlat
    recov::Rrec
    recov_delay::Rgenrec
end

InferencialSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec}(θi, θgen) where {Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} = @views InferencialSEIR(θi[1],
    Rauto(θi[2:1+nparams(Rauto)]...),
    Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...),
    Rout(θgen[1:nparams(Rout)]...),
    Rlat(θi[2+nparams(Rauto)+nparams(Rinf):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)]...),
    Rgenlat(θgen[nparams(Rout)+1:nparams(Rout)+nparams(Rgenlat)]...),
    Rrec(θi[2+nparams(Rauto)+nparams(Rinf)+nparams(Rlat):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)+nparams(Rrec)]...),
    Rgenrec(θgen[nparams(Rout)+nparams(Rgenlat)+1:nparams(Rout)+nparams(Rgenlat)+nparams(Rgenrec)]...))

# For the generative SEIR
#θi = inf, latency, recovery
#θgen = pseed, autoinf, out, lat_delay, recov_delay

struct GenerativeSEIR{T,Rauto,Rinf,Rout,Rlat,Rrec,Rgenlat,Rgenrec} <: IndividualSEIR
    pseed::T
    autoinf::Rauto 
    inf::Rinf 
    out::Rout 
    latency::Rlat 
    lat_delay::Rgenlat
    recov::Rrec 
    recov_delay::Rgenrec
end

GenerativeSEIR{Rauto, Rout, Rgenlat, Rgenrec}(θgen) where {Rauto, Rout, Rgenlat, Rgenrec} = 
    @views GenerativeSEIR(θgen[1],
    Rauto(θgen[2:1+nparams(Rauto)]...),
    UnitRate(),
    Rout(θgen[2+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)]...),
    UnitRate(),
    Rgenlat(θgen[2+nparams(Rout)+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)]...),
    UnitRate(),
    Rgenrec(θgen[2+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)+nparams(Rgenrec)]...))

#General functions for SEIR

compatibility(x, O, Mp::StochasticModel{<:IndividualSEIR}) = all((x[i,2]<t<x[i,3]) == s for (i,s,t,p) in O)


function Sampler(M::StochasticModel{<:IndividualSEIR})  #0=S  1=E  2=I  3=R
    N::Int = nv(M.G)
    s::Vector{Int} = zeros(Int, N)
    Q::TrackingHeap{Int, Float64, 2, MinHeapOrder, NoTrainingWheels} = TrackingHeap(Float64, S=NoTrainingWheels)
    function updateQ!(i, t)
        if t < M.T
            Q[i] = haskey(Q, i) ? min(Q[i], t) : t
        end
    end
    function sample!(x)
        @assert N == size(x,1)
        empty!(Q)
        x .= M.T
        s .= 0
        for i = 1:N
            ind = individual(M, i)
            updateQ!(i, rand() < ind.pseed ? zero(M.T) : delay(ind.autoinf, zero(M.T)))
        end
        while !isempty(Q)
            i, t = pop!(Q)
            s[i] += 1
            x[i,s[i]] = t
            if s[i] == 1
                updateQ!(i, delay(shift(individual(M,i).lat_delay,t) * individual(M,i).latency, t))
            elseif s[i] == 2
                trec = min(M.T, delay(shift(individual(M,i).recov_delay,t) * individual(M,i).recov, t))
                for (j,rij) ∈ out_neighbors(M,i)
                    if s[j] == 0
                        tij = delay(shift(individual(M,i).out,t) * rij * individual(M,j).inf, t)
                        tij < trec && updateQ!(j, tij)
                    end
                end
                updateQ!(i, trec)
            end
        end
        return x
    end
end

n_states(M::StochasticModel{<: IndividualSEIR}) = 4
trajectorysize(M::StochasticModel{<: IndividualSEIR}) = (nv(M.G), 3)


