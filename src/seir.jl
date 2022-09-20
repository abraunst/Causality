using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, Distributions, TrackingHeaps

export  GenerativeSEIR, StepInferentialSEIR, GaussianInferentialSEIR

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

struct GaussianInferentialSEIR <: SEIR end
individual(::Type{GaussianInferentialSEIR}, θi, θg) = @views IndividualSEIR(θi[1], #pseed 
    GaussianRate(θi[2:4]...),   #autoinf
    GaussianRate(θi[5:7]...),   #infection_in
    GaussianRate(θg[5:7]...),   #infection_out delay
    GaussianRate(θi[8:10]...),  #latency in time
    GaussianRate(θg[8:10]...),  #latency delay
    GaussianRate(θi[11:13]...), #recovery in time
    GaussianRate(θg[11:13]...), #recovery delay
)

struct StepInferentialSEIR <: SEIR end
individual(::Type{StepInferentialSEIR}, θi, θg) = @views IndividualSEIR(θi[1], #pseed
    StepRate(ConstantRate(θi[2]), θi[3], θi[4]),    #autoinf
    StepRate(ConstantRate(θi[5]), θi[6], θi[7]),    #infection_in
    GaussianRate(θg[5:7]...),                       #infection_out delay
    StepRate(ConstantRate(θi[8]), θi[9], θi[10]),   #latency in time
    GaussianRate(θg[8:10]...),                      #latency delay
    StepRate(ConstantRate(θi[11]), θi[12], θi[13]), #recovery in time
    GaussianRate(θg[11:13]...),                     #recovery delay
)



struct InferentialSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} <: SEIR end

@individual InferentialSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec}(θi, θgen) =
    IndividualSEIR(θi,
        Rauto(θi),
        Rinf(θi),
        Rout(θgen),
        Rlat(θi),
        Rgenlat(θgen),
        Rrec(θi),
        Rgenrec(θgen))

struct GenerativeSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} <: SEIR end

@individual GenerativeSEIR{Rauto, Rout, Rgenlat, Rgenrec}(θi, θgen) =
    IndividualSEIR(θgen,
        Rauto(θgen),
        UnitRate(),
        Rout(θgen),
        UnitRate(),
        Rgenlat(θgen),
        UnitRate(),
        Rgenrec(θgen))

#General functions for SEIR

compatibility(x, O, Mp::StochasticModel{<:SEIR}) = all((x[i,2]<t<x[i,3]) == s for (i,s,t,p) in O)


function Sampler(M::StochasticModel{<:SEIR})  #0=S  1=E  2=I  3=R
    N::Int = nv(M.G)
    s::Vector{Int} = zeros(Int, N)
    Q::TrackingHeap{Int, Float64, 2, MinHeapOrder, NoTrainingWheels} = TrackingHeap(Float64, S=NoTrainingWheels)
    function updateQ!(i, t)
        Q[i] = haskey(Q, i) ? min(Q[i], t) : min(M.T,t)
    end
    function sample!(x)
        @assert N == size(x,1)
        x .= M.T
        empty!(Q)
        s .= 0
        flag = 0
        Z1 = prod(1 - individual(M,i).pseed for i = 1:N)
        for i = 1:N
            ind = individual(M, i)
            eff_seed = (flag == 0 ? ind.pseed / (1-Z1) : ind.pseed)
            updateQ!(i, rand() < eff_seed ? zero(M.T) : delay(ind.autoinf, zero(M.T)))
            Z1 /= (1-ind.pseed)
            flag += (Q[i] == zero(M.T))
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
        s += logdensity(lat, x[i,2])
    end
    rec = shift(ind.recov_delay, x[i,2]) * ind.recov
    s -= cumulated(rec, x[i,3]) - cumulated(rec, x[i,2])
    if x[i,3] < M.T
        s += logdensity(rec, x[i,3])
    end
    return s
end

logO(x, O, M::StochasticModel{<:SEIR}) = sum(log(p + ((t > x[i,1]) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)


#=function logO(x, O, M::StochasticModel{<:SEIR}) 
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == 0
            (x[i,1] < t < x[i,3]) && (su += log(p) - 1 + 4*((t - (x[i,1]+x[i,3])/2)^2)/(x[i,1]-x[i,3])^2 )
        elseif s==1            
            if t < x[i,1] 
                su += log(p) - (x[i,1] - t)^2 / x[i,1]^2
            elseif t > x[i,3]
                su += log(p) - (t - x[i,3])^2 / (T - x[i,3])^2  
            end
        end
    end
    su
end=#

#=function logO(x, O, M::StochasticModel{<:SEIR}) 
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == 0
            (x[i,1] < t < x[i,3]) && (su += log(p) - 100 + 400*((t - (x[i,1]+x[i,3])/2)^2)/(x[i,1]-x[i,3])^2 )
        elseif s==1            
            if t < x[i,1] 
                su += log(p) - 100*(x[i,1] - t)^2 / x[i,1]^2
            elseif t > x[i,3]
                su += log(p) - 100*(t - x[i,3])^2 / (T - x[i,3]^2)^2  
            end
        end
    end
    su
end=#

#Sierological Test function
#=function logO(x, O, M::StochasticModel{<:SEIR}) 
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == 0
            (x[i,1] < t ) && (su += log(p) - 100 * (t - x[i,1])^2/ (T - x[i,1])^2  )
        elseif s==1             
            (t < x[i,1] ) && (su += log(p) - 100 * (x[i,1] - t)^2 / x[i,1]^2)
        end
    end
    su
end=#



n_states(M::StochasticModel{<:SEIR}) = 4
trajectorysize(M::StochasticModel{<:SEIR}) = (nv(M.G), 3)
