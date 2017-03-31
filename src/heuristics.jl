
immutable ToNextML <: Policy
    p::VDPTagMDP
end

function action(p::ToNextML, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next-s.agent
    return atan2(diff[2], diff[1])
end
