using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Optim, Dates

using DiffEqBase, Flux, DiffResults, DiffEqSensitivity, ForwardDiff
using Flux.Tracker: @grad
using DiffEqSensitivity: adjoint_sensitivities_u0
######################################
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
####################################################### Observation ###############################################################
# Start conditions for the two species in the system
u0 = Float32[2.; 0.]
# Number of evaluations of the neural ODE. It relates to the numbers of layers of the neural net (depth of network).
datasize = 30
# Time span in which of evaluation will be and actual timepoints of evaluations
tspan = (0.0f0, 1.5f0)
t = range(tspan[1], tspan[2], length = datasize)
# The true ODE (with the true parameters) which the neural net should learn
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u.^3)'true_A)'
end
# Construction of the ODEProblem and solving the ODEProblem with Tsit5 solver
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
################################################### Building a neural ODE ##########################################################
# Derivative is modeled by a neural net. Chain concatinates the functions ode function and two dense layers.
dudt = Chain(x -> x.^3,
       Dense(2,50,tanh),
       Dense(50,2))
# Parameters of the model which are to be learnt. They are: W1 (2x50), b1 (50), W2 (50x2), b2 (2)
ps = Flux.params(dudt)
# Build node
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
# To set by user
init_train, number_sets, number_reps, number_epochs_loss1, number_epochs_loss2 = 5000, 1, 1, 1, 1000
# For saving
sa = saver(init_train+number_sets*(number_reps*(number_epochs_loss1+number_epochs_loss2)+init_train))
# Define collocation loss function with callback
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
collocation_loss_fct()=loss_n_ode.cost_function(ps)
cb_collocation = function ()
    sa.count_epochs = sa.count_epochs +  1
    update_saver(sa, Tracker.data(collocation_loss_fct()),Dates.Time(Dates.now()))
    #println("\"",Tracker.data(collocation_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
end
# Define L2 loss function with callback
l2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
cb_l2 = function ()
    sa.count_epochs = sa.count_epochs +  1
    update_saver(sa, Tracker.data(l2_loss_fct()),Dates.Time(Dates.now()))
    #println("\"",Tracker.data(l2_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
end
# Training
data_iter_init_collocation = Iterators.repeated((), init_train)
data_iter_rep_collocation = Iterators.repeated((), number_epochs_loss1)
data_iter_rep_l2 = Iterators.repeated((), number_epochs_loss2)
opt = ADAM(0.1)
opt_l2 = ADAM(0.1)

# init train
Flux.train!(collocation_loss_fct, ps, data_iter_init_collocation, opt, cb = cb_collocation)
for j in 1:number_sets
    # Push in new area
    #for i in 1:number_reps
        Flux.train!(l2_loss_fct, ps, data_iter_rep_l2, opt_l2, cb = cb_l2)
        print("sa.count_epochs ", sa.count_epochs)
        print("\n")
        #Flux.train!(collocation_loss_fct, ps, data_iter_rep_collocation, opt, cb = cb_collocation)
    #end
    # Find local minimum
    #Flux.train!(collocation_loss_fct, ps, data_iter_init_collocation, opt, cb = cb_collocation)
end

# Call n_ode to get first prediction and to show startpoint for training.
pred = n_ode(u0)
scatter(t, ode_data[1,:], label="data")
    scatter!(t, Flux.data(pred[1,:]), label="prediction")
    scatter!(t, ode_data[2,:], label="data")
    scatter!(t, Flux.data(pred[2,:]), label="prediction")

#savefig("sogood.png")
using BSON: @save
name = string(init_train)*"_"*string(number_sets)*"_"*string(number_reps)*"_"*string(number_epochs_loss1)*"_"*string(number_epochs_loss2)
#@save "DiffEqFlux.jl/test/results/mix/model_mix_"*name*"_epochs.bson" dudt
