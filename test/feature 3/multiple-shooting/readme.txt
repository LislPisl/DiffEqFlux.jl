params_and_init.jl is motivation for multiple shooting:

start_params_one = [1.9 1.5;0.02 1.0; -0.5 0.01]
start_params_two = [1.9 1.5;0.04 1.0; -0.5 0.01]

(truth: [1.9 1.5 ; 0.05 1.0; -0.5 0.01])

As shown in fig, eaxmple.pdf, starting points for optim minor diff (third position .02 and .04) --> one optim does not work.
