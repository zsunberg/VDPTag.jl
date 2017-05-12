@with_kw immutable DiscreteVDPTagMDP <: MDP{Int, Int}
    cmdp::VDPTagMDP     = VDPTagMDP()
    n_bins::Int         = 60
    grid_lim::Float64   = 3.0
    n_angles::Int       = 10
end

@with_kw immutable DiscreteVDPTagPOMDP <: POMDP{Int, Int, Int}
    cpomdp::VDPTagPOMDP = VDPTagPOMDP()
    n_bins::Int         = 60
    grid_lim::Float64   = 3.0
    n_angles::Int       = 10
    n_obs_angles::Int   = 10
end

typealias DiscreteVDPTagProblem Union{DiscreteVDPTagMDP, DiscreteVDPTagPOMDP}

mdp(p::DiscreteVDPTagMDP) = p
mdp(p::DiscreteVDPTagPOMDP) = DiscreteVDPTagMDP(p.cpomdp.mdp, p.n_bins, p.grid_lim, p.n_angles)
cproblem(p::DiscreteVDPTagMDP) = p.cmdp
cproblem(p::DiscreteVDPTagPOMDP) = p.cpomdp

# state
function convert_s(::Type{Int}, s::TagState, p::DiscreteVDPTagProblem)
    n = p.n_bins
    factor = n/(2*p.grid_lim)
    ai = clamp(ceil(Int, (s.agent[1]+p.grid_lim)*factor), 1, n)
    aj = clamp(ceil(Int, (s.agent[2]+p.grid_lim)*factor), 1, n)
    ti = clamp(ceil(Int, (s.target[1]+p.grid_lim)*factor), 1, n)
    tj = clamp(ceil(Int, (s.target[2]+p.grid_lim)*factor), 1, n)
    return sub2ind((n,n,n,n), ai, aj, ti, tj)
end
function convert_s(::Type{TagState}, s::Int, p::DiscreteVDPTagProblem)
    n = p.n_bins
    factor = 2*p.grid_lim/n
    ai, aj, ti, tj = ind2sub((n,n,n,n), s)
    return TagState((Vec2(ai, aj)-0.5)*factor-p.grid_lim, (Vec2(ti, tj)-0.5)*factor-p.grid_lim)
end

# action
function convert_a(::Type{Int}, a::Float64, p::DiscreteVDPTagProblem)
    i = ceil(Int, a*p.n_angles/(2*pi))
    while i > p.n_angles
        i -= p.n_angles
    end
    while i < 1
        i += p.n_angles
    end
    return i
end
convert_a(::Type{Float64}, a::Int, p::DiscreteVDPTagProblem) = (a-0.5)*2*pi/p.n_angles

function convert_a(T::Type{Int}, a::TagAction, p::DiscreteVDPTagPOMDP)
    i = convert_a(T, a.angle, p::DiscreteVDPTagPOMDP)
    if a.look
        return i + p.n_angles
    else
        return i
    end
end
function convert_a(::Type{TagAction}, a::Int, p::DiscreteVDPTagPOMDP)
    return TagAction(a > p.n_angles, convert_a(Float64, a % p.n_angles, p))
end

# observation
function convert_o(::Type{Int}, o::Float64, p::DiscreteVDPTagPOMDP)
    i = ceil(Int, o*p.n_obs_angles/(2*pi))
    while i > p.n_obs_angles
        i -= p.n_obs_angles
    end
    while i < 1
        i += p.n_obs_angles
    end
    return i
end
convert_o(::Type{Float64}, o::Int, p::DiscreteVDPTagPOMDP) = (o-0.5)*2*pi/p.n_obs_angles

n_states(p::DiscreteVDPTagProblem) = mdp(p).n_bins^4
n_actions(p::DiscreteVDPTagProblem) = mdp(p).n_angles
n_actions(p::DiscreteVDPTagPOMDP) = 2*p.n_angles
discount(p::DiscreteVDPTagProblem) = mdp(p).cmdp.discount
isterminal(p::DiscreteVDPTagProblem, s::Int) = isterminal(p, convert_s(TagState, s, p))

function generate_s(p::DiscreteVDPTagProblem, s::Int, a::Int, rng::AbstractRNG)
    cs = convert_s(TagState, s, p)
    ca = convert_a(action_type(cproblem(p)), a, p)
    csp = generate_s(cproblem(p), cs, ca, rng)
    return convert_s(Int, csp, p)
end

function generate_sr(p::DiscreteVDPTagProblem, s::Int, a::Int, rng::AbstractRNG)
    cs = convert_s(TagState, s, p)
    ca = convert_a(action_type(cproblem(p)), a, p)
    csp = generate_s(cproblem(p), cs, ca, rng)
    r = reward(cproblem(p), cs, ca, csp)
    return (convert_s(Int, csp, p), r)
end

function generate_sor(p::DiscreteVDPTagPOMDP, s::Int, a::Int, rng::AbstractRNG)
    cs = convert_s(TagState, s, p)
    ca = convert_a(action_type(cproblem(p)), a, p)
    csor = generate_sor(cproblem(p), cs, ca, rng)
    return (convert_s(Int, csor[1], p), convert_o(Int, csor[2], p), csor[3])
end

actions(p::DiscreteVDPTagProblem) = 1:n_actions(p)

immutable DiscreteVDPInitDist
    p::DiscreteVDPTagProblem
end
eltype(::Type{DiscreteVDPInitDist}) = Int
function rand(rng::AbstractRNG, d::DiscreteVDPInitDist)
    cs = rand(rng, VDPInitDist())
    return convert_s(Int, cs, d.p)
end
initial_state_distribution(p::DiscreteVDPTagProblem) = DiscreteVDPInitDist(p)
