I am trying to implement multiple shooting for solving odes more robust.
I follow ideas of:
F. Hamilton,
“Parameter estimation in differential equations: A numerical study of shooting methods,”
SIAM Undergraduate Research Online, 2011.


difference is: I observe not only one species but all. (they say only x)

I am doing this:

1. single shooting, init, one residual point                -done
2. single shooting, init, multiple residual point           -done
3. single shooting, init + params, multiple residual point  -done   now: save at is residual points!
4. either: multiple shooting


final goal: table with
  dim 1: optimizer
  dim 2: number shooting nodes
  dim 3: example system
  dim 4: noise level

also to do: test other noise distributions?
further work: somehow optimize number of shooting nodes
