using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter

frames = Frames(MIME("image/png"), fps=2)

dmdp = DiscreteVDPTagMDP()
mdp = cproblem(dmdp)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
policy = translate_policy(ToNextML(mdp), mdp, dmdp, dmdp)
hist = simulate(hr, dmdp, policy)

cstates = [convert_s(TagState, s, dmdp) for s in state_hist(hist)]
gr()
@showprogress "Creating gif..." for s in cstates
    push!(frames, plot(mdp, s))
end

filename = string(tempname(), "_vdprun.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
