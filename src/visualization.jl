@recipe function f(mdp::VDPTagMDP, s::TagState)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    @series begin
        color := :black
        seriestype := :scatter
        label := "target"
        markersize := 0.1
        [s.target[1]], [s.target[2]]
    end
    @series begin
        color --> :blue
        label --> "agent"
        pts = Plots.partialcircle(0, 2*pi, 100, mdp.tag_radius)
        x, y = Plots.unzip(pts)
        x+s.agent[1], y+s.agent[2]
    end
end
