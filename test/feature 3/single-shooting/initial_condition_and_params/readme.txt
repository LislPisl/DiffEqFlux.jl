Inferring params and inits.


motivation: only params
start_params_one = [-0.1 1.0; -0.6 0.01]  -> fit's nicely
start_params_two = [-0.1 1.0; -0.5 0.01]  -> crashes
____
with inits:
init of 1.5 1.1 does not work --> fist species too far down --> crash.
init far up --> ok.
