
immutable ToNextML <: Policy
    p::VDPTagMDP
end

function action(p::ToNextML, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next-s.agent
    return atan2(diff[2], diff[1])
end

type NextMLFirst{RNG<:AbstractRNG}
    p::VDPTagMDP
    rng::RNG
end

function next_action(gen::NextMLFirst, mdp::Union{POMDP, MDP}, s::TagState, snode)
    if length(children(snode)) < 1
        return action(ToNextML(gen.p), s)
    else
        return 2*pi*rand(gen.rng)
    end
end

immutable TranslatedPolicy{P<:Policy, T, ST, AT} <: Policy
    policy::P
    translator::T
    S::Type{ST}
    A::Type{AT}
end

function translate_policy(p::Policy, from::Union{POMDP,MDP}, to::Union{POMDP,MDP}, translator)
    return TranslatedPolicy(p, translator, state_type(from), action_type(to))
end

function action(p::TranslatedPolicy, s)
    cs = convert(p.S, s, p.translator)
    ca = action(p.policy, cs)
    return convert(p.A, ca, p.translator)
end

