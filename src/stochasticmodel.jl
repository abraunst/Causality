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



struct StochasticModel{I,Rout,VR} <: AbstractStochasticModel
    T::Float64
    θ::Matrix
    Λ::SparseMatrixCSC{Bool, Int}  
    Λ2::SparseMatrixCSC{Int, Int}  
    out::Rout
    V::VR
end


function StochasticModel{I}(T,θ,G,out::Rout,V::VR = fill(UnitRate(), nnz(G))) where {I,Rout,VR}
    G1=SparseMatrixCSC(G.m, G.n, G.colptr, G.rowval, collect(1:nnz(G)))
    G2=sparse(G1')
    StochasticModel{I,Rout,VR}(T,θ,G,G2,out,V)
end


individual(M::StochasticModel{I}, θi) where I = I(θi, M.out)
individual(M::StochasticModel, i::Int) = individual(M, @view M.θ[:,i])
in_neighbors(M::StochasticModel, i::Int) = ((M.Λ.rowval[k], M.V[k]) for k ∈ nzrange(M.Λ,i))
out_neighbors(M::StochasticModel, i::Int) = ((M.Λ2.rowval[k], M.V[M.Λ2.nzval[k]]) for k ∈ nzrange(M.Λ2,i))
