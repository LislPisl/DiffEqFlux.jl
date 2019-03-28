using OrdinaryDiffEq, Plots, Distributions
####################################################### Observation ###############################################################
# Start conditions for the two species in the system
u0 = Float32[2.; 0.]
# Number of evaluations of the neural ODE. It relates to the numbers of layers of the neural net (depth of network).
datasize = 30
# Time span in which of evaluation will be and actual timepoints of evaluations
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
# The true ODE (with the true parameters) which the neural net should learn
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
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
noisy_data = add_noise(ode_data, 0.4)
scatter!(t, noisy_data[1,:], label="noisy data")
#single boundary
known_end = noisy_data[:,end]
#to guess: u0
######## first try ############
try_u0_1 = [1.;1.]
#define and solve ODE
prob_1 = ODEProblem(trueODEfunc, try_u0_1, tspan)
try_solve_1 = Array(solve(prob_1,Tsit5(),saveat=t))
#get last element of try
try_end_1 = try_solve_1[:,end]
#evaluate try by L2
los_val_1 =  sum(abs2,try_end_1 .- known_end)
scatter(t, ode_data[1,:], label="data")
scatter!(t, try_solve_1[1,:], label="try 1")
######## second try ############
try_u0_2 = [1.8;.1]
#define and solve ODE
prob_2 = ODEProblem(trueODEfunc, try_u0_2, tspan)
try_solve_2 = Array(solve(prob_2,Tsit5(),saveat=t))
#get last element of try
try_end_2 = try_solve_2[:,end]
#evaluate try by L2
los_val_2 =  sum(abs2,try_end_2 .- known_end)
scatter!(t, try_solve_2[1,:], label="try 2")
