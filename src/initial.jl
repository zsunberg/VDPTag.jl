immutable VDPInitDist end

function rand(rng::AbstractRNG, d::VDPInitDist)
    return TagState([0.0, 0.0], 4.0*rand(rng, 2))
end

initial_state_distribution(p::VDPTagMDP) = VDPInitDist()
