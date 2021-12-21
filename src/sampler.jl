using DataStructures, ProgressMeter, SparseArrays, TrackingHeaps

export Sampler, prior, post, reweighted_post



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


function Sampler(M::StochasticModel{<:IndividualSEIR})  #0=S  1=E  2=I  3=R
    N = nv(M.G)
    s = zeros(Int, N)
    Q::TrackingHeap{Int, Float64, 2, MinHeapOrder, NoTrainingWheels} = TrackingHeap(Float64, S=NoTrainingWheels)
    function sample!(x)
        @assert N == size(x,1)
        empty!(Q)
        x .= M.T
        s .= 0
        for i = 1:N
            ind = individual(M, i)
            Q[i] = min(M.T, rand() < ind.pseed ? zero(M.T) : delay(ind.autoinf, zero(M.T)))
        end
        while !isempty(Q)
            i, t = pop!(Q)
            s[i] += 1
            x[i,s[i]] = t
            if s[i] == 1
                Q[i] = min(M.T, delay(individual(M,i).latency,zero(M.T))+t)
            else
                x[i,3] = min(M.T, delay(individual(M,i).recov,zero(M.T))+t)
                for (j,rij) ∈ out_neighbors(M,i)
                    if s[j] == 0
                        tij = delay(shift(individual(M,i).out,t) * rij * individual(M,j).inf, t)
                        Q[j] = min( Q[j], (tij < x[i,3] ? tij : M.T) )
                    end
                end
                s[i] = 3
            end
        end
        return x
    end
end



compatibility(x, O, Mp::StochasticModel{<:IndividualSI}) = all((x[i,1] < t) == s for (i,s,t,p) in O)
compatibility(x, O, Mp::StochasticModel{<:IndividualSEIR}) = all((x[i,2]<t<x[i,3]) == s for (i,s,t,p) in O)



prior(M::StochasticModel{<:IndividualSI}; numsamples=10^5) = prior(M, zeros(nv(M.G)); numsamples=numsamples)
prior(M::StochasticModel{<:IndividualSEIR}; numsamples=10^5) = prior(M, zeros(nv(M.G), n_states(M)), numsamples=numsamples)

function prior(M::StochasticModel, x; numsamples)
    sample! = Sampler(M)
    N = sample!.N
    stats = zeros(numsamples, size(x)...)
    @showprogress for i=1:numsamples
        sample!(x)
        stats[i, :] .= x
    end
    stats
end

post(M::StochasticModel{<:IndividualSI}, O; numsamples=10^5) = post(M, O, zeros(nv(M.G)); numsamples=numsamples)
post(M::StochasticModel{<:IndividualSEIR}, O; numsamples=10^5) = post(M, O, zeros(nv(M.G), n_states(M)), numsamples=numsamples)
function post(Mp, O, x; numsamples=10^4)
    sample! = Sampler(Mp)
    N = nv(Mp.G)
    stats = zeros(numsamples, size(x)...)
    @showprogress for m=1:numsamples
        ok = false
        while !ok
            sample!(x)
            ok = compatibility(x, O, Mp)
        end
        stats[m, :] .= x
    end
    stats
end


function reweighted_post(Mp, M, O; numsamples=10^4, stats = zeros(numsamples, nv(Mp.G)))
    weights = zeros(numsamples)
    sample! = Sampler(M)
    N = nv(Mp.G)
    x = zeros(N);
    @showprogress for m=1:numsamples
        sample!(x)
        weights[m] = logQ(x, Mp) + logO(x, O, Mp) - logQ(x, M)
        stats[m, :] .= x
    end
    weights .= exp.(weights .- minimum(weights))
    weights ./= sum(weights)
    stats, weights
end


function reweighted_post(Mp, M, O; numsamples=10^4, )
    weights = zeros(numsamples)
    sample! = Sampler(M)
    N = size(Mp.Λ,1)
    stats = zeros(N, n_states(M) -1, numsamples)
    x = zeros(N, n_states(M) -1);
    @showprogress for m=1:numsamples
        sample!(x)
        weights[m] = logQ(x, Mp) + logO(x, O, Mp) - logQ(x, M)
        stats[:, :,  m] .= x
    end
    weights .= exp.(weights .- minimum(weights))
    weights ./= sum(weights)
    stats, weights
end
