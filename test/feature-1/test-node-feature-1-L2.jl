using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Optim, Dates
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
# Defining anonymous function for the neural ODE with the model. in: u0, out: solution with current params.
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
#L2 loss
n_epochs = 1
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

sa_l2 = saver(n_epochs)

L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
# Callback function to observe L2 training.
cb = function ()
    sa_l2.count_epochs = sa_l2.count_epochs +  1

    update_saver(sa_l2, Tracker.data(L2_loss_fct()),Dates.Time(Dates.now()))

    println("\"",Tracker.data(L2_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
end
#training call
opt = ADAM(0.1)
data = Iterators.repeated((), n_epochs)
@time Flux.train!(L2_loss_fct, ps, data, opt, cb = cb)


# Call n_ode to get first prediction and to show startpoint for training.
pred = n_ode(u0)
scatter(t, ode_data[1,:], label="data")
scatter!(t, Flux.data(pred[1,:]), label="prediction")
scatter!(t, ode_data[2,:], label="data")
scatter!(t, Flux.data(pred[2,:]), label="prediction")

#savefig("sogood.png")


using BSON: @save
@save "model_l2_1000_epochs.bson" dudt
