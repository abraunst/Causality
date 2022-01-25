using Causality: shift, cumulated, delay
using IntervalUnionArithmetic

@testset "Rates" begin
	M = 10^7
	g = shift(GaussianRate(0.9,5.0,5.0),4.0)
	mask = Interval(0.1,4.0)∪Interval(5.0,7.0)∪Interval(8.,15.)
	mg = MaskedRate(g, mask)
	@test isapprox(exp(-cumulated(mg,7.5)), sum(delay(mg,0.0)>7.5 for i=1:M)/M, atol=1/sqrt(M))
end
