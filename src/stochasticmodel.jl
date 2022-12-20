using SparseArrays, IndexedGraphs

export StochasticModel

"""
`StochasticModel{I,GT,VT}` 


"""
struct StochasticModel{I,GT,VT}
    T::Float64
    θ::Matrix{Float64}
    G::GT
    θgen::Matrix{Float64}
    V::VT
end

StochasticModel(::Type{I}, T, θ, G::GT, θgen, V::VT = fill(UnitRate(), ne(G))) where {I,GT,VT} = StochasticModel{I,GT,VT}(T,θ,G,θgen,V)


macro individual(x)
    D = Dict()
    v = Expr[]
    @assert x.head === :(=)
    a = x.args[1]
    @assert a.head == :call
    θ1 = a.args[2]
    θ2 = a.args[3]
    D[θ1] = 0
    D[θ2] = 0
    J = a.args[1]
    @assert J.head == :curly
    types = J.args[2:end]
    y = x.args[2].args[2]
    @assert y.head == :call
    I = y.args[1]
    params = y.args[2:end]
    for p in params
        if p === θ1 || p === θ2
            D[p] = :($(D[p]) + 1)
            push!(v, :($p[$(D[p])]))
        elseif p.head === :call
            T = p.args[1]
            if length(p.args) > 1 && p.args[2] ∈ (θ1,θ2) 
                θ = p.args[2]
                push!(v, :($T($θ[$(D[θ]) .+ (1:Causality.nparams($T))]...)))
                D[θ] = :($(D[θ]) + Causality.nparams($T))
            else
                push!(v, p)
            end
        end
    end
    :(individual(::Type{$J}, $θ1, $θ2) where {$(types...)} = @views $I($(v...)))
end

"""
`individual(M::StochasticModel{I}, i::Int, θi = @view(M.θ[:,i]), θgen = M.θgen) where {I} = individual(I, θi, θgen)` \n
Returns the individual associated to label i. 
"""
individual(M::StochasticModel{I}, i::Int, θi = @view(M.θ[:,i]), θgen = M.θgen) where {I} = individual(I, θi, θgen)

"""
`in_neighbors(M::StochasticModel, i::Int)` \n
Returns the label of all the individuals linked to i. \n
This is the toal numbers of neighbours of i in an undirected graph.\n
See also `out_neighbors`. 
"""
in_neighbors(M::StochasticModel, i::Int) = ((e.src, M.V[e.idx]) for e ∈ inedges(M.G, i))

"""
`out_neighbors(M::StochasticModel, i::Int)` \n
Returns the label of all the individuals which i is linked to. \n
This is the toal numbers of neighbours of i in an undirected graph. \n
see also `in_neighbors`. 
"""
out_neighbors(M::StochasticModel, i::Int) = ((e.dst, M.V[e.idx]) for e ∈ outedges(M.G, i))
