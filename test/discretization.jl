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

obs = Array{Float64}(n_observations(p))
for i in 1:n_observations(p)
    co = convert_o(Float64, i, p)
    obs[i] = co
    j = convert_o(Int, co, p)
    @test i == j
    @test co == convert_o(Float64, j, p)
end
@test length(unique(actions(dpomdp))) == n_actions(p)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, dmdp, RandomPolicy(dpomdp))

# unicodeplots()

N = 10_000
sp = TagState([0.0, 0.0], [-1.0, 1.0])
a = convert_a(Int, TagAction(true, 0.0), dpomdp)
counts = zeros(Int, n_observations(dpomdp))
rng = MersenneTwister(18)
for i in 1:N
    o = generate_o(p, sp, a, sp, rng)
    counts[o] += 1
end
# bar(counts, xlabel="o", ylabel="count", title="Histogram of observations for target at $(sp.target)")
# gui()

os = observations(dpomdp)
weights = Float64[]
sp = TagState([0.0, 0.0], [-1.0, 1.0])
a = convert_a(Int, TagAction(true, 0.0), dpomdp)
for o in os
    push!(weights, obs_weight(dpomdp, a, sp, o))
end
    
# bar(os, weights, xlabel="o", ylabel="weight", title="Weight for target at $(sp.target)")
# gui()

@show counts./sum(counts)
@show weights
for o in os
    @test abs(weights[o] - counts[o]/sum(counts)) < 0.01
end

@test abs(sum(weights) - 1.0) < 0.01

xs = linspace(-10, 10, 50)
weights = Float64[]
o = 3
a = convert_a(Int, TagAction(true, 0.0), dpomdp)
for x in xs
    sp = TagState([0.0, 0.0], [x, 1.0])
    push!(weights, obs_weight(dpomdp, a, sp, o))
end
# plot(xs, weights, xlabel="target x", ylabel="weight", title="Weight for observation $o")
# gui() 
