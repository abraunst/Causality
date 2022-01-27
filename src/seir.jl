using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, TrackingHeaps

export  GenerativeSEIR, InferentialSEIR

abstract type SEIR end

# For the SEIR model
#θi = pseed,autoinf,inf,latency,recovery
#θgen = pseed, autoinf, out, lat_delay, recov_delay

struct IndividualSEIR{P,Rauto,Rinf,Rout,Rlat,Rrec,Rgenlat,Rgenrec}
    pseed::P
    autoinf::Rauto
    inf::Rinf
    out::Rout
    latency::Rlat
    lat_delay::Rgenlat
    recov::Rrec
    recov_delay::Rgenrec
end


struct InferentialSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} <: SEIR end

individual(::Type{InferentialSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec}}, θi, θgen) where {Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} = @views IndividualSEIR(θi[1],
    Rauto(θi[2:1+nparams(Rauto)]...),
    Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...),
    Rout(θgen[2+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)]...),
    Rlat(θi[2+nparams(Rauto)+nparams(Rinf):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)]...),
    Rgenlat(θgen[2+nparams(Rout)+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)]...),
    Rrec(θi[2+nparams(Rauto)+nparams(Rinf)+nparams(Rlat):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)+nparams(Rrec)]...),
    Rgenrec(θgen[2+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)+nparams(Rgenrec)]...))


struct GenerativeSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} <: SEIR end

individual(::Type{GenerativeSEIR{Rauto, Rout, Rgenlat, Rgenrec}}, θi, θgen) where {Rauto, Rout, Rgenlat, Rgenrec} = @views IndividualSEIR(
    θgen[1],
    Rauto(θgen[2:1+nparams(Rauto)]...),
    UnitRate(),
    Rout(θgen[2+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)]...),
    UnitRate(),
    Rgenlat(θgen[2+nparams(Rout)+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)]...),
    UnitRate(),
    Rgenrec(θgen[2+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)+nparams(Rgenrec)]...))

#General functions for SEIR

compatibility(x, O, Mp::StochasticModel{<:SEIR}) = all((x[i,2]<t<x[i,3]) == s for (i,s,t,p) in O)


function Sampler(M::StochasticModel{<:SEIR})  #0=S  1=E  2=I  3=R
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


function logQi(M::StochasticModel{<:SEIR}, i, ind, x::Matrix{Float64})     #x[i] = (tE, tI, tR)
    s = 0.0
    if iszero(x[i,1])
        s += log(ind.pseed)
    else
        s += log(1-ind.pseed)
        s -= cumulated(ind.autoinf, x[i,1])
        sSE = density(ind.autoinf, x[i,1])
        for (j,rji) ∈ in_neighbors(M, i)
            if x[j,2] < x[i,1]
                inf = ind.inf * rji * shift(ind.out,x[j,2]) # we use ind.out because all the out are the same
                s -= cumulated(inf, min(x[i,1],x[j,3])) - cumulated(inf, x[j,2])
                if x[i,1] < x[j,3]
                    sSE += density(inf,x[i,1])
                end
            end
        end
        if x[i,1] < M.T
            s += log(sSE)
        end
    end
    lat = shift(ind.lat_delay, x[i,1]) * ind.latency
    s -= cumulated(lat, x[i,2]) - cumulated(lat, x[i,1])
    if x[i,2] < M.T
        s += log(density(lat, x[i,2]))
    end
    rec = shift(ind.recov_delay, x[i,2]) * ind.recov
    s -= cumulated(rec, x[i,3]) - cumulated(rec, x[i,2])
    if x[i,3] < M.T
        s += log(density(rec, x[i,3]))
    end
    return s
end

logO(x, O, M::StochasticModel{<:SEIR}) = sum(log(p + ((x[i,2] < t < x[i,3]) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)

n_states(M::StochasticModel{<:SEIR}) = 4
trajectorysize(M::StochasticModel{<:SEIR}) = (nv(M.G), 3)