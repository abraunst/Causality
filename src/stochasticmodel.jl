export StochasticModel

struct StochasticModel{T,G,A,B,C,D}
    θ::T
    Λ::G
    pseed::A
    autoinf::B
    inf::C
    T::D
end

StochasticModel(θ, T, Λ, r1, r2) = StochasticModel(θ, Λ,
        view(θ, 1, 1:N),
        [r1(view(θ, 2, i), view(θ, 3, i), view(θ, 4, i)) for i=1:size(θ,2)],
        [r2(view(θ, 5, i), view(θ, 6, i), view(θ, 7, i)) for i=1:size(θ,2)],
        T);
