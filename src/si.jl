using SparseArrays, IndexedGraphs, DataStructures, ProgressMeter, SparseArrays, TrackingHeaps, Distributions, IntervalUnionArithmetic

export  Sampler, GenerativeSI, InferentialSI, GaussianInferentialSI, IndividualSI, SI, StepInferentialSI

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

struct StepInferentialSI <: SI end
individual(::Type{StepInferentialSI}, θi, θg) = @views IndividualSI(θi[1],
    StepRate(ConstantRate(θi[2]), θi[3], θi[4]), 
    StepRate(ConstantRate(θi[5]), θi[6], θi[7]), 
    GaussianRate(θg[5:7]...))



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
        Z1 = prod(1 - individual(M,i).pseed for i = 1:N)
        flag = 0 #a flag to see if there is a zero patient
        for i in eachindex(x)
            ind = individual(M, i)
            eff_seed = (flag == 0 ? ind.pseed / (1-Z1) : ind.pseed)
            Q[i] = min(M.T, rand() < eff_seed ? zero(M.T) : delay(ind.autoinf, zero(M.T)))
            Z1 /= (1-ind.pseed)
            flag += (Q[i] == zero(M.T)) #if i is a zero patient the flag is no more 0
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



function logO(x, O, M::StochasticModel{<:SI}) 
    sum(log(p + ((x[i] < t) == s)*(1-2p)) for (i,s,t,p) in O; init=0.0)
end


#=function logO(x, O, M::StochasticModel{<:SI})
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == (x[i] < t)
            su += log(1-p)
        end
        if s == 0
            if x[i] < t
            su += log(p) - 10 + 10*(x[i]/t)
            end
        elseif s==1
            if x[i] > t
                su += log(p) + 10*(T-x[i])/(T-t) - 10
            end
        end
    end
    su
end=#


#=function logO(x, O, M::StochasticModel{<:SI})
    su = 0.
    T = M.T
    for (i,s,t,p) in O
        if s == (x[i] < t)
            su += log(1-p)
        end
        if s == 0
            if x[i] < t
                su += log(p) +  (10 * x[i]/t)^2 - 100 
            end
        elseif s==1
            if x[i] > t
                su += log(p) - (10 * (x[i] - t) / (T-t) )^2
            end
        end
    end
    su
end=#

n_states(M::StochasticModel{<:SI}) = 2
trajectorysize(M::StochasticModel{<:SI}) = nv(M.G)
