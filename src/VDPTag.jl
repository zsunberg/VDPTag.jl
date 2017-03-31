module VDPTag

using POMDPs
using StaticArrays
using Parameters
using Plots

typealias Vec2 SVector{2, Float64}

import POMDPs: generate_s, reward, actions, initial_state_distribution, discount
import POMDPs: action
import Base: rand

export
    TagState,
    TagAction,
    VDPTagMDP,

    ToNextML

immutable TagState
    agent::Vec2
    target::Vec2
end

immutable TagAction
    look::Bool
    angle::Float64
end

@with_kw immutable VDPTagMDP <: MDP{TagState, Float64}
    mu::Float64         = 2.0
    dt::Float64         = 0.1
    step_size::Float64  = 0.2
    tag_radius::Float64 = 0.1
    tag_reward::Float64 = 10.0
    pos_std::Float64    = 0.1
    discount::Float64   = 0.95
end

function next_ml_target(p::VDPTagMDP, pos::Vec2)
    steps = round(Int, p.step_size/p.dt)
    for i in 1:steps
        pos = rk4step(p, pos)
    end
    return pos
end

function generate_s(p::VDPTagMDP, s::TagState, a::Float64, rng::AbstractRNG)
    pos = next_ml_target(p, s.target)
    return TagState(s.agent+p.step_size*[cos(a), sin(a)], pos+p.pos_std*randn(rng, 2))
end

function reward(p::VDPTagMDP, s::TagState, a::Float64)
    if norm(s.agent-s.target) < p.tag_radius
        return p.tag_reward
    else
        return 0.0
    end
end

discount(p::VDPTagMDP) = p.discount

immutable AngleSpace end
rand(rng::AbstractRNG, ::AngleSpace) = 2*pi*rand(rng)

actions(::VDPTagMDP) = AngleSpace()

#=
@with_kw immutable VDPTagPOMDP <: POMDP{TagState, TagAction, Nullable{Float64}}
    mdp::VDPTag             = VDPTag()
    bearing_std::Float64    = deg2rad(10.0)
    c_meas
end

immutable POVDPTagActionSpace end
=#

include("rk4.jl")
include("initial.jl")
include("visualization.jl")
include("heuristics.jl")

end # module
