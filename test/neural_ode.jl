
using Flux, DiffEqFlux, DifferentialEquations, Plots,Optim

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

# Call n_ode to get first prediction and to show startpoint for training.
pred = n_ode(u0)
scatter(t, ode_data[1,:], label="data")
scatter!(t, Flux.data(pred[1,:]), label="prediction")

# Define predict function. Returns tracked array: U: 2x30, starts at u0 = [2,0]
function predict_n_ode()
  n_ode(u0)
end


loss_n_ode = DiffEqParamEstim.node_two_stage_function(predict_n_ode(), u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
los_fct()=loss_n_ode.cost_function(ps)

cur_pred = Flux.data(predict_n_ode())
pl = scatter(t,ode_data[1,:],label="data")
scatter!(pl,t,cur_pred[1,:],label="prediction")
#display(pl)
#result = Optim.optimize(los, 0.0, 10.0)
#w1=randn(2,50)
#w2=randn(50,2)
#b1=randn(50)
#b2=randn(2)
#los([w1,b1,w2,b2])
old_loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())
#los(destructure(dudt))

data = Iterators.repeated((), 50)
opt = ADAM(0.1)

# Callback function to observe training.
cb = function ()
  println("los_fct() is ", old_loss_n_ode())
  cur_pred = Flux.data(predict_n_ode())
  println("cur_pred is ", cur_pred)
end

# Display the ODE with the initial parameter values.
cb()

# Start training process.
Flux.train!(old_loss_n_ode, ps, data, opt, cb = cb)
