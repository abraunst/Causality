using DataStructures, ProgressMeter

export Sampler, prior, post

function Sampler(M::StochasticModel)
    N = size(Λ,1)
    s = falses(N)
    Q = PriorityQueue{Int,Float64}()
    function sample!(x)
        empty!(Q)
        x .= M.T
        s .= false
        for i in eachindex(x)
            Q[i] = min(M.T, rand() < M.pseed[i] ? zero(M.T) : infect(M.autoinf[i], zero(M.T)))
        end
        while !isempty(Q)
            i, t = dequeue_pair!(Q)
            s[i] && continue
            s[i] = true
            x[i] = t
            for j ∈ ∂(Λ,i)
                if !s[j]
                    Q[j] = min(Q[j], infect(M.inf[j], t))
                end
            end
        end
        return x
    end
end


function prior(sample!, numsamples=10^5)
    x = zeros(N);
    stats = zeros(N, numsamples)
    @showprogress for i=1:numsamples
        sample!(x)
        stats[:, i] .= x
    end
    stats
end

function post(Mp, O; numsamples=10^4, stats = zeros(N, numsamples))
    sample! = Sampler(Mp)
    N = size(Mp.Λ,1)
    x = zeros(N);
    @showprogress for m=1:numsamples
        ok = false
        while !ok
            sample!(x)
            ok = all((x[i] < t) == s for (i,s,t,p) in O)
        end
        stats[:, m] .= x
    end
    stats
end
