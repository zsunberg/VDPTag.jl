using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using ParticleFilters

frames = Frames(MIME("image/png"), fps=2)

pomdp = VDPTagPOMDP()

filter = SIRParticleFilter(pomdp, 10000, rng=MersenneTwister(100))

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1), show_progress=true)
hist = simulate(hr, pomdp, RandomPolicy(pomdp, rng=MersenneTwister(2)), filter)

gr()
@showprogress "Creating gif..." for i in 1:length(hist)
    push!(frames, plot(pomdp, view(hist, 1:i)))
end

filename = string(tempname(), "_vdprun.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
