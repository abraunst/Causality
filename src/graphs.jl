using SparseArrays, LinearAlgebra
using IndexedGraphs

makeGNP(N, rho) = sprand(N, N, rho, n->fill(true, n)) |> x->tril(x,-1) |> (x->x.|=x') |> dropzeros! |> IndexedBiDiGraph

function makeBarabasi(N; k=1)
    E = Graphs.edges(Graphs.barabasi_albert(N, k))
    A = sparse([e.src for e in E], [e.dst for e in E], fill(true, length(E)), N, N)
    A .+= A'
    return IndexedBiDiGraph(A)
end


function makeProximity(N,R2)
    lattice = rand(N,2)
    edgesL = []
    edgesR = []
    for i=1:N
        for j=1:i-1
            if (lattice[i,1]-lattice[j,1])^2+(lattice[i,2]-lattice[j,2])^2 < R2 
                append!(edgesL,i)
                append!(edgesR,j)
            end
        end
    end
    A = sparse(edgesL, edgesR, fill(true, length(edgesL)), N, N)
    A .+= A'
    return IndexedBiDiGraph(A)
end


