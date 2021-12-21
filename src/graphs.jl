using SparseArrays, LinearAlgebra
using Graphs 

makeGNP(N, rho) = sprand(N, N, rho, n->fill(true, n)) |> x->tril(x,-1) |> (x->x.|=x') |> dropzeros! |> SparseMatrixDiGraph

function makeBarabasi(N; k=1)
    E = Graphs.edges(Graphs.barabasi_albert(N, k))
    A = sparse([e.src for e in E], [e.dst for e in E], fill(true, (length(E)), N, N))
    A .+= A'
    return SparseMatrixDiGraph(A)
end


struct SparseMatrixDiGraph{T} <: AbstractGraph{T}
    A::SparseMatrixCSC{Irrational{:π}, T}
    X::SparseMatrixCSC{T,T}
end

struct IndexedEdge{T}
    src::T
    dst::T
    idx::T
end

function SparseMatrixDiGraph(A::AbstractMatrix)
    size(A,1) != size(A,2) && throw(ArgumentError("Matrix should be square"))
    any(A[i,i] for i=1:size(A,1)) && throw(ArgumentError("Self loops are not allowed"))
    A = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, fill(π, nnz(A)))
    X = sparse(SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, collect(1:nnz(A)))')
    SparseMatrixDiGraph(A, X)
end

Graphs.edges(g::SparseMatrixDiGraph) = (IndexedEdge{Int}(i,g.A.rowval[k],k) for i=1:size(g.A,2) for k=nzrange(g.A,i))

Base.eltype(g::SparseMatrixDiGraph{T}) where T = T

Graphs.edgetype(g::SparseMatrixDiGraph{T}) where T = IndexedEdge{T}

Graphs.has_edge(g::SparseMatrixDiGraph, i, j) = g.A[i,j]

Graphs.has_vertex(g::SparseMatrixDiGraph, i) = i ∈ 1:size(g.A, 2)

Graphs.inneighbors(g::SparseMatrixDiGraph, i) = @view g.A.rowval[nzrange(g.A,i)]

Graphs.outneighbors(g::SparseMatrixDiGraph, i) = @view g.X.rowval[nzrange(g.X,i)]

outedges(g::SparseMatrixDiGraph, i) = (IndexedEdge{Int}(i,g.A.rowval[k],k) for k=nzrange(g.A,i))

inedges(g::SparseMatrixDiGraph, i) = (IndexedEdge{Int}(g.X.rowval[k],i,g.X.nzval[k]) for k=nzrange(g.X,i))

Graphs.ne(g::SparseMatrixDiGraph) = nnz(g.A)

Graphs.nv(g::SparseMatrixDiGraph) = size(g.A,2)

Graphs.vertices(g::SparseMatrixDiGraph) = 1:size(g.A,2)

Graphs.is_directed(::Type{SparseMatrixDiGraph{T}}) where {T} = true

Graphs.is_directed(g::SparseMatrixDiGraph) = true

Base.zero(g::SparseMatrixDiGraph) = SparseMatrixDiGraph(zero(g.A), similar(g.W, 0))

index(e::IndexedEdge) = e.idx
