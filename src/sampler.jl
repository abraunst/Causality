using DataStructures, ProgressMeter, SparseArrays, TrackingHeaps

export prior, post, reweighted_post , softpost, softpostnoise


function prior(M::StochasticModel; numsamples=10^4)
    sample! = Sampler(M)
    x = zeros(trajectorysize(M))
    N = sample!.N
    stats = zeros(numsamples, size(x)...)
    @showprogress for i=1:numsamples
        sample!(x)
        stats[i, :, :] .= x
    end
    stats
end


function post(Mp, O; numsamples=10^4)
    sample! = Sampler(Mp)
    x = zeros(trajectorysize(Mp))
    N = nv(Mp.G)
    stats = zeros(numsamples, size(x)...)
    @showprogress for m=1:numsamples
        ok = false
        while !ok
            sample!(x)
            ok = compatibility(x, O, Mp)
        end
        stats[m, :, :] .= x
    end
    stats
end

function softpost(Mp, O; numsamples=10^4)
    sample! = Sampler(Mp)
    x = zeros(trajectorysize(Mp))
    N = nv(Mp.G)
    stats = zeros(numsamples, size(x)...)
    @showprogress for m=1:numsamples
        sample!(x)
        ok = compatibility(x, O, Mp)
        if ok
           stats[m, :, :] .= x             
        end        
    end
    stats
    stats[vec(mapslices(col -> any(col .!= 0), stats, dims = 2)),:]
end

function softpostnoise(Mp, O; numsamples=10^4)
    sample! = Sampler(Mp)
    x = zeros(trajectorysize(Mp))
    weights = zeros(numsamples)
    N = nv(Mp.G)
    stats = zeros(numsamples, size(x)...)
    @showprogress for m=1:numsamples
        sample!(x)
        weights[m] = exp(logO(x,O,Mp))
        stats[m, :, :] .= x                           
    end
    stats, weights
end


function reweighted_post(Mp, M, O; numsamples=10^4)
    weights = zeros(numsamples)
    sample! = Sampler(M)
    N = nv(Mp.G)
    x = zeros(trajectorysize(Mp));
    stats = zeros(numsamples, size(x)...)
    @showprogress for m=1:numsamples
        sample!(x)
        weights[m] = logQ(x, Mp) + logO(x, O, Mp) - logQ(x, M)
        stats[m, :, :] .= x
    end
    weights .= exp.(weights .- minimum(weights))
    weights ./= sum(weights)
    stats, weights
end

