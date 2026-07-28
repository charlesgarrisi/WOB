using Plots
include("dro_lib.jl")

k = 1.2
tau = (k - 1) / (k + 1)

Xi = (
    s_min = -500.0,
    s_max = 4000.0,
    d_min = -14000.0,
    d_max = 10000.0,
)

epsilon = 0.5
plots_dir = "epsilon_$(epsilon)"
mkpath(plots_dir)

# Empirical observations xi_i = (g_i, s_i, delta_i)
g0 = [0.7, 0.9, 0.2]
s0 = [50.0, -100.0, 200.0]
d0 = [0.0, -2000.0, 3000.0]

# Normalization by support width
W = [
    1.0,
    1 / (Xi.s_max - Xi.s_min),
    1 / (Xi.d_max - Xi.d_min),
]

function sensitivity(coord, grid)

    nstar = zeros(length(grid), length(grid))

    for (i, x1) in enumerate(grid)
        for (j, x2) in enumerate(grid)

            g = copy(g0)
            s = copy(s0)
            d = copy(d0)

            if coord == :g
                g[1], g[2] = x1, x2

            elseif coord == :s
                s[1], s[2] = x1, x2

            elseif coord == :d
                d[1], d[2] = x1, x2

            else
                error("coord must be :g, :s, or :d")
            end

            n, _, _ = solve_dro_exact(g, s, d, Xi, W, tau, epsilon, ngrid=101)
            nstar[j, i] = n
        end
    end

    return nstar
end

function make_sensitivity_plot(coord, grid, xlabel_txt, ylabel_txt, title_txt, filename)
    Z = sensitivity(coord, grid)

    p = heatmap(
        grid,
        grid,
        Z,
        xlabel = xlabel_txt,
        ylabel = ylabel_txt,
        colorbar_title = "n*",
        clims = (0.0, 1.0),
        dpi = 300,
        title = title_txt,
    )

    savefig(p, joinpath(plots_dir, filename))
    display(p)
end

make_sensitivity_plot(
    :g,
    range(0.0, 1.0, length=50),
    "g_1", "g_2",
    "Optimal decision n*(g_1, g_2)",
    "sensitivity_g.png"
)

make_sensitivity_plot(
    :s,
    range(Xi.s_min, Xi.s_max, length=50),
    "s_1", "s_2",
    "Optimal decision n*(s_1, s_2)",
    "sensitivity_s.png"
)

make_sensitivity_plot(
    :d,
    range(Xi.d_min, Xi.d_max, length=50),
    "d_1", "d_2",
    "Optimal decision n*(d_1, d_2)",
    "sensitivity_d.png"
)
