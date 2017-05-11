p = DiscreteVDPTagMDP()

states = Array(TagState, n_states(p))
for i in 1:n_states(p)
    cs = convert(TagState, i, p)
    states[i] = cs
    j = convert(Int, cs, p)
    @test i == j
    @test cs == convert(TagState, j, p)
end
@test length(unique(states)) == n_states(p)

actions = Array(Float64, n_actions(p))
for i in 1:n_actions(p)
    ca = convert(Float64, i, p)
    actions[i] = ca
    j = convert(Int, ca, p)
    @test i == j
    @test ca == convert(Float64, j, p)
end
@test length(unique(actions)) == n_actions(p)

dmdp = DiscreteVDPTagMDP()

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
policy = translate_policy(ToNextML(cproblem(dmdp)), cproblem(dmdp), dmdp, dmdp)
hist = simulate(hr, dmdp, policy)