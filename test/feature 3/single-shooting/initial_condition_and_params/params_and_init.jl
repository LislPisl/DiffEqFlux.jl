using OrdinaryDiffEq, Plots, Optim, Distributions
u0 = Float32[2.; 0.]
datasize = 30           # resiudual number! = d's
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
true_A = [-0.1 2.0; -2.0 -0.1]
function param_ODEfunc(guess_A)
    function guess_ODEfunc(du, u, p, t)
      du .= ((u.^3)'guess_A)'
    end
    return guess_ODEfunc
end
trueODEfunc = param_ODEfunc(true_A)
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
noisy_data = add_noise(ode_data, 0.3)
scatter!(t, noisy_data[1,:], label="noisy data")

function make_ode(init_temp_A)
    temp_u0 = init_temp_A[1,:]
    temp_A = init_temp_A[2:3,:]
    temp_ODEfunc = param_ODEfunc(temp_A)
    temp_prob = ODEProblem(temp_ODEfunc, temp_u0, tspan)
    temp_ode_data = Array(solve(temp_prob,Tsit5(),saveat=t))
    return temp_ode_data
end


L2_loss_fct(params) = sum(abs2,noisy_data .- make_ode(params))
start_params_one = [3.5 0.;-0.1 1.0; -0.6 0.01]
start_params_two = [1.6 0.1;-0.1 1.0; -0.6 0.01]
result_one = Optim.optimize(L2_loss_fct, start_params_one )
result_two = Optim.optimize(L2_loss_fct, start_params_two )


solution_one = make_ode(result_one.minimizer)
solution_two = make_ode(result_two.minimizer)

solstart_one = make_ode(start_params_one)
solstart_two = make_ode(start_params_two)
species = 1
plot(t, ode_data[species,:], label="real sol")
scatter!(t, noisy_data[species,:], label="train data")
plot!(t, solstart_one[species,:], label="begin 1 data")
plot!(t, solstart_two[species,:], label="begin 2 data")

plot!(t, solution_one[species,:], label="sol 1 data")
plot!(t, solution_two[species,:], label="sol 2 data")

savefig("good.png")

#1.5 1.1 does not work --> fist species too far down --> crash. up is ok.
