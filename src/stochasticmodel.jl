using SparseArrays

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

IndividualSI{Rauto, Rinf}(θi, rout) where {Rauto, Rinf} = @views IndividualSI(θi[1], Rauto(θi[2:1+nparams(Rauto)]...), Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...), rout)



# An Individual SEIR with pseed, autoinf, inf, out, latency and recovery. Note that out infection is fixed

struct IndividualSEIR{T,Rauto,Rinf,Rout,Rlat,Rrec}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
    latency::Rlat
    recov::Rrec
end


IndividualSEIR{Rauto, Rinf, Rlat, Rrec}(θi, rout) where {Rauto, Rinf, Rlat, Rrec} = @views IndividualSEIR(θi[1], Rauto(θi[2:1+nparams(Rauto)]...), Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...), rout,
Rlat(θi[2+nparams(Rauto)+nparams(Rinf):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)]...), 
Rrec(θi[2+nparams(Rauto)+nparams(Rinf)+nparams(Rlat):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)+nparams(Rrec)]...))



struct StochasticModel{I,GT,Rout,VR} <: AbstractStochasticModel
    T::Float64
    θ::Matrix{Float64}
    G::GT
    out::Rout
    V::VR
end
StochasticModel(::Type{I}, T, θ, G::GT, out::Rout, V::VR = fill(UnitRate(), ne(G))) where {I,GT,Rout,VR} = StochasticModel{I,GT,Rout,VR}(T,θ,G,out,V)

individual(M::StochasticModel{I}, θi) where I = I(θi, M.out)
individual(M::StochasticModel, i::Int) = individual(M, @view M.θ[:,i])
in_neighbors(M::StochasticModel, i::Int) = ((e.src, M.V[e.idx]) for e ∈ inedges(M.G, i))
out_neighbors(M::StochasticModel, i::Int) = ((e.dst, M.V[e.idx]) for e ∈ outedges(M.G, i))
n_states(M::StochasticModel{<: IndividualSI}) = 2
n_states(M::StochasticModel{<: IndividualSEIR}) = 4

trajectorysize(M::StochasticModel{<: IndividualSI}) = (N,)   
trajectorysize(M::StochasticModel{<: IndividualSEIR}) = (N, 3)  