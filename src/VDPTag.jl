module VDPTag

using POMDPs
using StaticArrays
using Parameters
using Plots
using Distributions
using POMDPToolbox
using ParticleFilters

typealias Vec2 SVector{2, Float64}

import POMDPs: generate_s, generate_sr, reward, actions, initial_state_distribution, discount
import POMDPs: observation, pdf
import POMDPs: action
import Base: rand, eltype, isnull

export
    TagState,
    TagAction,
    VDPTagMDP,
    VDPTagPOMDP,

    ToNextML,
    mdp

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
    step_size::Float64  = 0.4
    tag_radius::Float64 = 0.1
    tag_reward::Float64 = 10.0
    pos_std::Float64    = 0.02
    discount::Float64   = 0.99
end

@with_kw immutable VDPTagPOMDP <: POMDP{TagState, TagAction, Nullable{Float64}}
    mdp::VDPTagMDP          = VDPTagMDP()
    bearing_std::Float64    = deg2rad(5.0)
    bearing_cost::Float64   = 0.0
end

typealias VDPTagProblem Union{VDPTagMDP,VDPTagPOMDP}
mdp(p::VDPTagMDP) = p
mdp(p::VDPTagPOMDP) = p.mdp


function next_ml_target(p::VDPTagMDP, pos::Vec2)
    steps = round(Int, p.step_size/p.dt)
    for i in 1:steps
        pos = rk4step(p, pos)
    end
    return pos
end

function generate_s(pp::VDPTagProblem, s::TagState, a::Float64, rng::AbstractRNG)
    p = mdp(pp)
    pos = next_ml_target(p, s.target)
    return TagState(s.agent+p.step_size*[cos(a), sin(a)], pos+p.pos_std*randn(rng, 2))
end

function generate_sr(p::VDPTagProblem, s::TagState, a::Float64, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    return sp, reward(p, s, a)
end

function reward(pp::VDPTagProblem, s::TagState, a::Float64)
    p = mdp(pp)
    if norm(s.agent-s.target) < p.tag_radius
        return p.tag_reward
    else
        return 0.0
    end
end

discount(pp::VDPTagProblem) = mdp(pp).discount

immutable AngleSpace end
rand(rng::AbstractRNG, ::AngleSpace) = 2*pi*rand(rng)
actions(::VDPTagMDP) = AngleSpace()

generate_s(p::VDPTagPOMDP, s::TagState, a::TagAction, rng::AbstractRNG) = generate_s(p, s, a.angle, rng)

immutable POVDPTagActionSpace end
rand(rng::AbstractRNG, ::POVDPTagActionSpace) = TagAction(rand(rng, Bool), 2*pi*rand(rng))
actions(::VDPTagPOMDP) = POVDPTagActionSpace()

function reward(p::VDPTagPOMDP, s::TagState, a::TagAction)
    return reward(mdp(p), s, a.angle) - a.look*p.bearing_cost
end

immutable NullableAngleNormal
    null::Bool
    mean::Float64
    std::Float64

    NullableAngleNormal() = new(true)
    NullableAngleNormal(mean::Float64, std::Float64) = new(false, mean, std)
end
isnull(n::NullableAngleNormal) = n.null
eltype(::Type{NullableAngleNormal}) = Nullable{Float64}
function pdf(d::NullableAngleNormal, o::Nullable{Float64})
    if isnull(o)
        if isnull(d)
            return 1.0
        else
            return 0.0
        end
    else
        if isnull(d)
            return 0.0
        else
            dir_diff = abs(get(o)-d.mean)
            while dir_diff > pi
                dir_diff -= 2*pi
            end
            dir_diff = abs(dir_diff)
            return exp(-dir_diff^2/(2*d.std^2))
        end
    end
end
function rand(rng::AbstractRNG, d::NullableAngleNormal)
    if isnull(d)
        return Nullable{Float64}()
    else
        return Nullable{Float64}(d.mean + d.std*randn(rng))
    end
end

function observation(p::VDPTagPOMDP, a::TagAction, sp::TagState)
    if a.look
        diff = sp.target-sp.agent
        bearing = atan2(diff[2], diff[1])
        return NullableAngleNormal(bearing, p.bearing_std)
    else
        return NullableAngleNormal()
    end
end

include("rk4.jl")
include("initial.jl")
include("visualization.jl")
include("heuristics.jl")

end # module
