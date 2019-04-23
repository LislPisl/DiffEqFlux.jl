using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Optim, Dates
# ERROR: in two_stage_method.jl I assume it does not handle 1-species case.
####################################################### Observation ###############################################################
tspan = (0.0f0, 1.5f0)
ode_data = [2.432,
    7.237,
    5.826,
    3.874,
    6.697,
    7.778,
    6.937,
    5.736]
u0 = Float32[2.432]
datasize = 8
t = range(tspan[1], tspan[2], length = datasize)
################################################### Building a neural ODE ##########################################################
# Derivative is modeled by a neural net. Chain concatinates the functions ode function and two dense layers.
dudt = Chain(x -> x.^3,
       Dense(1,50,tanh),
       Dense(50,1))
# Parameters of the model which are to be learnt. They are: W1 (2x50), b1 (50), W2 (50x2), b2 (2)
ps = Flux.params(dudt)
# Getting loss function from two stage collocation function
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
#  loss function
two_stage_loss_fct()=loss_n_ode.cost_function(ps)
# Defining anonymous function for the neural ODE with the model. in: u0, out: solution with current params.
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)

n_epochs = 5000
data1 = Iterators.repeated((), n_epochs)
opt1 = ADAM(0.1)
# Callback function to observe two stage training.
mutable struct saver
    losses::Array{Float64,1}
    times::Array{Dates.Time,1}
    count_epochs::Int128
end
function saver(n_epochs)
    losses = zeros(n_epochs)
    times = fill(Dates.Time(Dates.now()),n_epochs)
    count_epochs = 0
    return saver(losses,times,count_epochs)
end
function update_saver(saver, loss_i, time_i)
    epoch_i = saver.count_epochs
    saver.losses[epoch_i] = loss_i
    saver.times[epoch_i] = time_i
end

sa = saver(n_epochs)

cb1 = function ()
    sa.count_epochs = sa.count_epochs +  1
    update_saver(sa, Tracker.data(two_stage_loss_fct()),Dates.Time(Dates.now()))
    println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
end
#two stage training call
@time Flux.train!(two_stage_loss_fct, ps, data1, opt1, cb = cb1)
# Call n_ode to get first prediction and to show startpoint for training.
pred = n_ode(u0)
sum(abs2,ode_data .- pred)
scatter(t, ode_data[1,:], label="data")
scatter!(t, Flux.data(pred[1,:]), label="prediction")
scatter!(t, ode_data[2,:], label="data")
scatter!(t, Flux.data(pred[2,:]), label="prediction")

#savefig("500.pdf")
#print("hi")
#create outside references and enclose them if that's what you need

test_u0 = [2.5 , 1.]
pred_t = n_ode(test_u0)

prob_t = ODEProblem(trueODEfunc, test_u0, tspan)
ode_data_t = Array(solve(prob_t,Tsit5(),saveat=t))

scatter(t, ode_data_t[1,:], label="data")
plot!(t, Flux.data(pred_t[1,:]), label="prediction")
scatter!(t, ode_data_t[2,:], label="data")
plot!(t, Flux.data(pred_t[2,:]), label="prediction")


using BSON: @save
@save "DiffEqFlux.jl/test/results/model_twostage_169555_epochs.bson" dudt
