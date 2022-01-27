using SparseArrays, IndexedGraphs

export StochasticModel

abstract type AbstractStochasticModel end


struct StochasticModel{I,GT,VR} <: AbstractStochasticModel
    T::Float64
    θ::Matrix{Float64}
    G::GT
    θgen::Matrix{Float64}
    V::VR
end

StochasticModel(::Type{I}, T, θ, G::GT, θgen, V::VR = fill(UnitRate(), ne(G))) where {I,GT,VR} = StochasticModel{I,GT,VR}(T,θ,G,θgen,V)

individual(M::StochasticModel{I}, θi) where I = I(θi, M.θgen)
individual(M::StochasticModel{I}, i::Int, θgen) where I = I((@view M.θ[:,i]), θgen)   
individual(M::StochasticModel, i::Int) = individual(M, @view M.θ[:,i])
in_neighbors(M::StochasticModel, i::Int) = ((e.src, M.V[e.idx]) for e ∈ inedges(M.G, i))
out_neighbors(M::StochasticModel, i::Int) = ((e.dst, M.V[e.idx]) for e ∈ outedges(M.G, i))
