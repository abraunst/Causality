using SparseArrays, IndexedGraphs

export StochasticModel

struct StochasticModel{I,GT,VT}
    T::Float64
    θ::Matrix{Float64}
    G::GT
    θgen::Matrix{Float64}
    V::VT
end

StochasticModel(::Type{I}, T, θ, G::GT, θgen, V::VT = fill(UnitRate(), ne(G))) where {I,GT,VT} = StochasticModel{I,GT,VT}(T,θ,G,θgen,V)

individual(M::StochasticModel{I}, θi) where {I} = individual(I, θi, M.θgen)
individual(M::StochasticModel{I}, i::Int, θgen = M.θgen) where {I} = individual(I, (@view M.θ[:,i]), θgen)
in_neighbors(M::StochasticModel, i::Int) = ((e.src, M.V[e.idx]) for e ∈ inedges(M.G, i))
out_neighbors(M::StochasticModel, i::Int) = ((e.dst, M.V[e.idx]) for e ∈ outedges(M.G, i))
