p = DiscreteVDPTagMDP()

states = Array{TagState}(n_states(p))
for i in 1:n_states(p)
    cs = convert_s(TagState, i, p)
    states[i] = cs
    j = convert_s(Int, cs, p)
    @test i == j
    @test cs == convert_s(TagState, j, p)
end
@test length(unique(states)) == n_states(p)

acts = Array{Float64}(n_actions(p))
for i in 1:n_actions(p)
    ca = convert_a(Float64, i, p)
    acts[i] = ca
    j = convert_a(Int, ca, p)
    @test i == j
    @test ca == convert_a(Float64, j, p)
end
@test length(unique(acts)) == n_actions(p)

dmdp = DiscreteVDPTagMDP()

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
policy = translate_policy(ToNextML(cproblem(dmdp)), cproblem(dmdp), dmdp, dmdp)
hist = simulate(hr, dmdp, policy)

dpomdp = AODiscreteVDPTagPOMDP()
p = dpomdp

observations = Array(Float64, n_observations(p))
for i in 1:n_observations(p)
    co = convert_o(Float64, i, p)
    observations[i] = co
    j = convert_o(Int, co, p)
    @test i == j
    @test co == convert_o(Float64, j, p)
end
@test length(unique(actions(dpomdp))) == n_actions(p)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, dmdp, RandomPolicy(dpomdp))
