using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, TrackingHeaps, Distributions

export  Sampler, GenerativeSI, InferentialSI, GaussianInferentialSI, IndividualSI, SI

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


struct GaussianInferentialSI <: SI end
individual(::Type{GaussianInferentialSI}, θi, θg) = @views IndividualSI(θi[1], GaussianRate(θi[2:4]...), GaussianRate(θi[5:7]...), GaussianRate(θg[5:7]...))



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

#=function logO(x, O, M::StochasticModel{<:SI}) 
    sum(log(p + ((x[i] < t) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)
end=#


#=function logO(x, O, M::StochasticModel{<:SI})
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == 0
            if x[i] < t
            su += log(p/10 + p * (x[i]/t))
            end
        elseif s==1
            if x[i] > t
                su += log(p/10 + (T-x[i])/(T-t)*p )
            end
        end
    end
    su
end=#


function logO(x, O, M::StochasticModel{<:SI})
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == 0
            if x[i] < t
                su += log(p) + (10 * x[i]/t)^2 - 100 
            end
        elseif s==1
            if x[i] > t
                su += log(p) - ( 10 * (x[i] - t) / (T-t) )^2
            end
        end
    end
    su
end

#=function logO(x, O, M::StochasticModel{<:SI})
    gauss = Distributions.Gaussian(0.,0.1)
    su = 0.
    for (i,s,t,p) in O
        if s == 0 
            su += logccdf(gauss, t - x[i]) 
        elseif s == 1
            su += logccdf(gauss, x[i] - t) 
        end
    end
    su
end =#

#=function logO(x, O, M::StochasticModel{<:SI})
    su = 0.
    for (i,s,t,p) in O
        Δt = t - x[i]
        sigma = 1/10
        if s == 0 
            su += log( ( (1-p)*erfc(sigma * Δt ) + p*erfc(-Δt * sigma) - 1e-3 * Δt ) / 2 ) 
        elseif s == 1
            su += log( ( (1-p)*erfc(-Δt * sigma) + p*erfc(Δt * sigma) +  1e-3 * Δt ) / 2 ) 
        end
    end
    su
end=#

n_states(M::StochasticModel{<:SI}) = 2
trajectorysize(M::StochasticModel{<:SI}) = (nv(M.G))
