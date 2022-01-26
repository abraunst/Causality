using IndexedGraphs
import Graphs

makeGNP(N, rho) = Graphs.erdos_renyi(N, rho) |> IndexedBiDiGraph

makeBarabasi(N; k=1) = Graphs.barabasi_albert(N, k) |> IndexedBiDiGraph

makeProximity(N, R2) = Graphs.euclidean_graph(N, 2; seed=-1, L=1., p=2., cutoff=sqrt(R2), bc=:open) |> first |> IndexedBiDiGraph
