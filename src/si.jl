using SparseArrays, IndexedGraphs

export  GenerativeSI, InferencialSI

abstract type IndividualSI end

# For the inferencial SI
#θi = pseed,autoinf,inf
#θgen = out

struct InferencialSI{T,Rauto,Rinf,Rout}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
end

InferencialSI{Rauto, Rinf, Rout}(θi, θgen) where {Rauto, Rinf, Rout} = @views InferencialSI(θi[1], Rauto(θi[2:1+nparams(Rauto)]...), Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...), 
Rout(θgen[1:nparams(Rout)]...)
)

# For the generative SI
#θi = inf
#θgen = pseed, autoinf, out

struct GenerativeSI{T,Rauto,Rinf,Rout}
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
end

GenerativeSI{Rauto, Rout}(θgen) where {Rauto, Rinf, Rout} = @views 
GenerativeSI(θgen[1], 
             Rauto(θgen[2:1+nparams(Rauto)]...), 
             UnitRate(), 
             Rout(θgen[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rout)]...)
)

n_states(M::StochasticModel{<: IndividualSI}) = 2
trajectorysize(M::StochasticModel{<: IndividualSI}) = (nv(M.G))
