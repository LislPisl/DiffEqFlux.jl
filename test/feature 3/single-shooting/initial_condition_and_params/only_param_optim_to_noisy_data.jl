using OrdinaryDiffEq, Plots, Optim, Distributions
u0 = Float32[2.; 0.]
datasize = 30
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


function make_ode(temp_A)
    temp_ODEfunc = param_ODEfunc(temp_A)
    temp_prob = ODEProblem(temp_ODEfunc, u0, tspan)
    temp_ode_data = Array(solve(temp_prob,Tsit5(),saveat=t))
    return temp_ode_data
end


L2_loss_fct(params) = sum(abs2,ode_data .- make_ode(params))


result = Optim.optimize(L2_loss_fct,  [-0.1 1.0; -2.0 -0.01])

solution = make_ode(result.minimizer)
solstart = make_ode([-0.1 1.0; -2.0 -0.01])

scatter(t, ode_data[1,:], label="data")
scatter!(t, solutionstart[1,:], label="begin data")
scatter!(t, solution[1,:], label="sol data")
