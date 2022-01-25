using SparseArrays, IndexedGraphs

export StochasticModel, IndividualSEIR, IndividualSI, GenericStaticSM, GenericDynamicSM

abstract type AbstractStochasticModel end

### some specific models follow

# An Individual with pseed, autoinf, inf and out. Note that out infection is fixed

struct IndividualSI{T,Rauto,Rinf,Rout}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
end

IndividualSI{Rauto, Rinf, Rout}(θi, θgen) where {Rauto, Rinf, Rout} = @views IndividualSI(θi[1], Rauto(θi[2:1+nparams(Rauto)]...), Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...), 
Rout(θgen[1:nparams(Rout)]...)
)



# An Individual SEIR with pseed, autoinf, inf, out, latency and recovery. Note that out infection is fixed

struct IndividualSEIR{T,Rauto,Rinf,Rout,Rlat,Rrec,Rgenlat,Rgenrec}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
    latency::Rlat
    lat_delay::Rgenlat
    recov::Rrec
    recov_delay::Rgenrec
end

IndividualSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec}(θi, θgen) where {Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} = @views IndividualSEIR(θi[1],
    Rauto(θi[2:1+nparams(Rauto)]...),
    Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...),
    Rout(θgen[1:nparams(Rout)]...),
    Rlat(θi[2+nparams(Rauto)+nparams(Rinf):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)]...),
    Rgenlat(θgen[nparams(Rout)+1:nparams(Rout)+nparams(Rgenlat)]...),
    Rrec(θi[2+nparams(Rauto)+nparams(Rinf)+nparams(Rlat):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)+nparams(Rrec)]...),
    Rgenrec(θgen[nparams(Rout)+nparams(Rgenlat)+1:nparams(Rout)+nparams(Rgenlat)+nparams(Rgenrec)]...))

struct StochasticModel{I,GT,VR} <: AbstractStochasticModel
    T::Float64
    θ::Matrix{Float64}
    G::GT
    θgen::Matrix{Float64}
    V::VR
end

StochasticModel(::Type{I}, T, θ, G::GT, θgen, V::VR = fill(UnitRate(), ne(G))) where {I,GT,VR} = StochasticModel{I,GT,VR}(T,θ,G,θgen,V)

individual(M::StochasticModel{I}, θi) where I = I(θi, M.θgen)
individual(M::StochasticModel{I}, i::Int, θgen) where I = I(M.θ[:,i], θgen)   #  MI RACCOMANDO CHIEDI!!
individual(M::StochasticModel, i::Int) = individual(M, @view M.θ[:,i])
in_neighbors(M::StochasticModel, i::Int) = ((e.src, M.V[e.idx]) for e ∈ inedges(M.G, i))
out_neighbors(M::StochasticModel, i::Int) = ((e.dst, M.V[e.idx]) for e ∈ outedges(M.G, i))
n_states(M::StochasticModel{<: IndividualSI}) = 2
n_states(M::StochasticModel{<: IndividualSEIR}) = 4

trajectorysize(M::StochasticModel{<: IndividualSI}) = (nv(M.G))
trajectorysize(M::StochasticModel{<: IndividualSEIR}) = (nv(M.G), 3)
