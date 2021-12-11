using SparseArrays

export StochasticModel, Individual, GenericStaticSM, GenericDynamicSM

abstract type StochasticModel end


### some specific models follow

# An Individual with pseed, autoinf, inf and out. Note that out infection is fixed

struct Individual{T,Rauto,Rinf,Rout}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
end

Individual{Rauto, Rinf}(θi, rout) where {Rauto, Rinf} = @views Individual(θi[1], Rauto(θi[2:1+nparams(Rauto)]...), Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...), rout)


# GenericStaticSM: here out infection is common to all individuals, no per link infection

struct GenericStaticSM{I,Rout} <: StochasticModel
    T::Float64  #epidemic time
    θ::Matrix{Float64}  #parameters
    Λ::SparseMatrixCSC{Bool,Int}  #contact graph
    out::Rout  #outgoing infection
end

GenericStaticSM{I}(T,θ,Λ,out::Rout) where {I,Rout} = GenericStaticSM{I,Rout}(T,θ,Λ,out)

individual(M::GenericStaticSM{I}, θi) where I = I(θi, M.out)
individual(M::GenericStaticSM, i::Int) = individual(M, @view M.θ[:,i])
neighbors(M::GenericStaticSM, i::Int) = ((M.Λ.rowval[k], UnitRate()) for k ∈ nzrange(M.Λ,i))


# GenericDynamicSM: similar but with per link infection rates (typically just masks)

struct GenericDynamicSM{I,Rout,VR} <: StochasticModel
    T::Float64
    θ::Matrix
    Λ::SparseMatrixCSC{Bool, Int}  #adjacency
    Λp::SparseMatrixCSC{Bool, Int} #transposed adjecency
    out::Rout
    V::VR
end

GenericDynamicSM{I}(T,θ,Λ,out::Rout,V::VR) where {I,Rout,VR} = GenericDynamicSM{I,Rout,VR}(T,θ,Λ,(sparse(Λ')),out,V)

individual(M::GenericDynamicSM{I}, θi) where I = I(θi, M.out)
individual(M::GenericDynamicSM, i::Int) = individual(M, @view M.θ[:,i])
in_neighbors(M::GenericDynamicSM, i::Int) = ((M.Λ.rowval[k], M.V[k]) for k ∈ nzrange(M.Λ,i))
out_neighbors(M::GenericDynamicSM, i::Int) = ((M.Λp.rowval[k], M.V[k]) for k ∈ nzrange(M.Λp,i))
