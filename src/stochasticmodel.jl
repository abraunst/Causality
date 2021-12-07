export StochasticModel

struct StochasticModel{TT,TH,G,A,B,C}
    T::TT
    θ::TH
    Λ::G
    pseed::A
    autoinf::B
    inf::C
end

StochasticModel(T, θ, Λ, r1, r2) = StochasticModel(T, θ, Λ,
                                                   view(θ, 1, 1:size(θ,2)),
                                                   [r1(view(θ, 2, i), view(θ, 3, i), view(θ, 4, i)) for i=1:size(θ,2)],
                                                   [r2(view(θ, 5, i), view(θ, 6, i), view(θ, 7, i)) for i=1:size(θ,2)]);
