using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter

frames = Frames(MIME("image/png"), fps=2)

mdp = VDPTagMDP()

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
# hist = simulate(hr, mdp, RandomPolicy(mdp, rng=MersenneTwister(2)))
hist = simulate(hr, mdp, ToNextML(mdp))

gr()
@showprogress "Creating gif..." for s in state_hist(hist)
    push!(frames, plot(mdp, s))
end

filename = string(tempname(), "_vdprun.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
