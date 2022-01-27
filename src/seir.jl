using SparseArrays, IndexedGraphs

export  GenerativeSEIR, InferencialSEIR

abstract type IndividualSEIR end

# For the inferencial SEIR
#θi = pseed,autoinf,inf,latency,recovery
#θgen = out, lat_delay, recov_delay

struct InferencialSEIR{T,Rauto,Rinf,Rout,Rlat,Rrec,Rgenlat,Rgenrec} <: IndividualSEIR
    pseed::T
    autoinf::Rauto
    inf::Rinf
    out::Rout
    latency::Rlat
    lat_delay::Rgenlat
    recov::Rrec
    recov_delay::Rgenrec
end

InferencialSEIR{Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec}(θi, θgen) where {Rauto, Rinf, Rout, Rlat, Rgenlat, Rrec, Rgenrec} = @views InferencialSEIR(θi[1],
    Rauto(θi[2:1+nparams(Rauto)]...),
    Rinf(θi[2+nparams(Rauto):1+nparams(Rauto)+nparams(Rinf)]...),
    Rout(θgen[1:nparams(Rout)]...),
    Rlat(θi[2+nparams(Rauto)+nparams(Rinf):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)]...),
    Rgenlat(θgen[nparams(Rout)+1:nparams(Rout)+nparams(Rgenlat)]...),
    Rrec(θi[2+nparams(Rauto)+nparams(Rinf)+nparams(Rlat):1+nparams(Rauto)+nparams(Rinf)+nparams(Rlat)+nparams(Rrec)]...),
    Rgenrec(θgen[nparams(Rout)+nparams(Rgenlat)+1:nparams(Rout)+nparams(Rgenlat)+nparams(Rgenrec)]...))

# For the generative SEIR
#θi = inf, latency, recovery
#θgen = pseed, autoinf, out, lat_delay, recov_delay

struct GenerativeSEIR{T,Rauto,Rinf,Rout,Rlat,Rrec,Rgenlat,Rgenrec} <: IndividualSEIR
    pseed::T
    autoinf::Rauto 
    inf::Rinf 
    out::Rout 
    latency::Rlat 
    lat_delay::Rgenlat
    recov::Rrec 
    recov_delay::Rgenrec
end

GenerativeSEIR{Rauto, Rout, Rgenlat, Rgenrec}(θgen) where {Rauto, Rout, Rgenlat, Rgenrec} = 
    @views GenerativeSEIR(θgen[1],
    Rauto(θgen[2:1+nparams(Rauto)]...),
    UnitRate(),
    Rout(θgen[2+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)]...),
    UnitRate(),
    Rgenlat(θgen[2+nparams(Rout)+nparams(Rauto):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)]...),
    UnitRate(),
    Rgenrec(θgen[2+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat):1+nparams(Rout)+nparams(Rauto)+nparams(Rgenlat)+nparams(Rgenrec)]...))

#General functions for SEIR
n_states(M::StochasticModel{<: IndividualSEIR}) = 4
trajectorysize(M::StochasticModel{<: IndividualSEIR}) = (nv(M.G), 3)
