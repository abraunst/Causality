using SparseArrays, LinearAlgebra

makeGNP(N, rho) = sprand(N, N, rho, n->fill(true, n)) |> x->tril(x,-1) |> (x->x.|=x') |> dropzeros!

function makeBarabasi(N; k=1)
    E = Graphs.edges(Graphs.barabasi_albert(N, k))
    A = sparse([e.src for e in E], [e.dst for e in E], fill(true, (length(E)), N, N))
    A .+= A'
    return A
end
