using OrdinaryDiffEq, Plots, Optim, Distributions
####################################################### Observation ###############################################################
# Start conditions for the two species in the system
u0 = Float32[2.; 0.]
# Number of evaluations of the neural ODE. It relates to the numbers of layers of the neural net (depth of network).
datasize = 30
# Time span in which of evaluation will be and actual timepoints of evaluations
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
# The true ODE (with the true parameters) which the neural net should learn
true_A = [-0.1 2.0; -2.0 -0.1]
function trueODEfunc(du, u, p, t)
  du .= ((u.^3)'true_A)'
end
# Construction of the ODEProblem and solving the ODEProblem with Tsit5 solver
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
scatter(t, ode_data[1,:], label="data")
function add_noise(in_data, noise_sigma)
    out_data = Array{Float64}(undef, size(in_data)[1], size(in_data)[2])
    if (noise_sigma>0)
        out_data = in_data + rand(Normal(0, noise_sigma), size(in_data)...)
    end
    return out_data
end
noisy_data = add_noise(ode_data, 0.1)
scatter!(t, noisy_data[1,:], label="noisy data")
#multiple boundaries
function convert_time(a, b, inc)
    A = Array{Float64}(undef, length(a:inc:b))
    A[:] = a:inc:b
    return A
end
y=convert_time(tspan[1], tspan[2],tspan[2]/(datasize-1))
boundary_positions = [5,10,15,20,25,30]
tes=y[boundary_positions]
known_pos = noisy_data[:,boundary_positions]
scatter!(tes, known_pos[1,:], label="d's")
function get_solve(try_i, b_positions)
    prob_i = ODEProblem(trueODEfunc, try_i, tspan)
    try_solve_i = Array(solve(prob_i, Tsit5(), saveat=t))
    try_end_i = try_solve_i[:,b_positions]
    return try_end_i
end
function get_loss(r)
    return sum(abs2,r)
end
init_u0 =known_pos[:,3]
get_loss(get_r(get_solve(init_u0, boundary_positions), known_pos))
### now minimize this using optim ###
optim_fuction(xx) = get_loss(get_r(get_solve(xx, boundary_positions), known_pos))
result = Optim.optimize(optim_fuction,  init_u0)


sol = ODEProblem(trueODEfunc, result.minimizer, tspan)
solution = Array(solve(sol, Tsit5(), saveat=t))

solstart = ODEProblem(trueODEfunc, init_u0, tspan)
solutionstart = Array(solve(solstart, Tsit5(), saveat=t))

scatter(t, ode_data[1,:], label="data")
scatter!(t, solutionstart[1,:], label="begin data")
scatter!(t, solution[1,:], label="sol data")
