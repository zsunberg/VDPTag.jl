using VDPTag
using POMDPs
using Base.Test
using POMDPToolbox
# using Plots
using MCTS

# include("discretization.jl")

pomdp = VDPTagPOMDP()
gen = NextMLFirst(mdp(pomdp), MersenneTwister(31))
s = TagState([1.0, 1.0], [-1.0, -1.0])

struct MyNode end
MCTS.n_children(::MyNode) = rand(1:10)

@inferred next_action(gen, pomdp, s, MyNode())
@inferred next_action(gen, pomdp, initial_state_distribution(pomdp), MyNode())
