using OrdinaryDiffEq, Plots, Optim, Distributions
function param_ODEfunc(guess_A)
    function guess_ODEfunc(du, u, p, t)
      du .= ((u.^3)'guess_A)'
    end
    return guess_ODEfunc
end
function add_noise(in_data, noise_sigma)
    out_data = Array{Float64}(undef, size(in_data)[1], size(in_data)[2])
    if (noise_sigma>0)
        out_data = in_data + rand(Normal(0, noise_sigma), size(in_data)...)
    end
    return out_data
end
function L2_loss_fct(params)
    print(params)
    return sum(abs2,noisy_data .- make_ode(params))
end
function make_ode(init_temp_A)
    temp_u0 = init_temp_A[1,:]
    temp_A = init_temp_A[2:3,:]
    temp_ODEfunc = param_ODEfunc(temp_A)
    temp_prob = ODEProblem(temp_ODEfunc, temp_u0, tspan)
    temp_ode_data = Array(solve(temp_prob,Tsit5(),saveat=t))
    return temp_ode_data
end
function make_sub_ode(init_temp_A, tspan_sub, t_sub)
    temp_u0 = init_temp_A[1,:]
    temp_A = init_temp_A[2:3,:]
    temp_ODEfunc = param_ODEfunc(temp_A)
    temp_prob = ODEProblem(temp_ODEfunc, temp_u0, tspan_sub)
    temp_ode_data = Array(solve(temp_prob,Tsit5(),saveat=t_sub))
    return temp_ode_data
end
########################### example call #############################
u0 = Float32[2.; 0.]
datasize = 30           # resiudual number! = d's
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
true_A = [-0.1 2.0; -2.0 -0.1]
trueODEfunc = param_ODEfunc(true_A)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
noisy_data = add_noise(ode_data, 0.5)
############################# visualise ###############################
plot(t, ode_data[1,:], label = "data")
scatter!(t, noisy_data[1,:], label = "noisy data")
############################# play around #############################
result = Optim.optimize(L2_loss_fct, start_params )
solution = make_ode(result.minimizer)
solstart = make_ode(start_params)
subs = 3
ns = 2
all_start_params = [2.5 1.5;2.5 1.5;2.5 1.5;-0.2 1.0; -0.6 0.01]
create_subs(all_start_params, subs, tspan, datasize, ns)
#to do: make resiudals_per_sub = old_datasize/number_subs robust
function create_subs(all_params, number_subs, old_tspan, old_datasize, number_species)
    out_data = Array{Float64}(undef, old_datasize, number_species)
    resiudals_per_sub = old_datasize/number_subs
    time_per_sub = (old_tspan[2]-old_tspan[1])/number_subs
    for i in 1:number_subs
        start_params_i = hcat(all_params[i,:], all_params[number_subs+1:end,:])
        print("start_params_i",start_params_i)
        tspan_i = (time_per_sub*(i-1),time_per_sub*i)
        t_i = range(tspan_i[1], tspan_i[2], length = resiudals_per_sub)
        out_data[1:number_species,i:i+resiudals_per_sub] = make_sub_ode(start_params_i, tspan_i, t_i)
    end
end
i=1
start_params_i = hcat(all_start_params[i,:], all_start_params[subs+1:end,:])'
all_start_params[i,:]
all_start_params[subs+1:end,:]
all_start_params[subs+1:end,:]'
