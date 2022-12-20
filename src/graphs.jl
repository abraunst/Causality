using IndexedGraphs
import Graphs

"""
`makeGNP(N, rho)` \n
Initializes an Erdos Renyi graph .\n
`N` is the number of nodes. \n
`rho` is the density. \n
"""
makeGNP(N, rho) = Graphs.erdos_renyi(N, rho) |> IndexedBiDiGraph


"""
`makeBarabasi(N; k=1)` \n
Initializes a Barabasi Albert graph with parameters `N` and `k` .
"""
makeBarabasi(N; k=1) = Graphs.barabasi_albert(N, k) |> IndexedBiDiGraph


"""
`makeProximity(N, R2)` \n
Initializes a Proximity graph. \n
`N` is the number of nodes.\n
`R2` is the radius of proximity. 
"""
makeProximity(N, R2) = Graphs.euclidean_graph(N, 2; seed=-1, L=1., p=2., cutoff=sqrt(R2), bc=:open) |> first |> IndexedBiDiGraph
