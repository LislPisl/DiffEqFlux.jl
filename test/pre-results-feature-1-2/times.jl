using Statistics, Plots
# two stage, 500 epochs.
time_500 = [0.990310 , 0.973435 ,1.085917 ,1.083980 , 1.064565 ]
print(mean(time_500))
# two stage, training time 500,1000,...,5000 epochs
x = [500,1000,2500,5000]
y = [1.0396414, 2.041506, 4.725699, 8.181069]
scatter(x,y,label = "Training time [s] for number of epochs")
savefig("training_time.pdf")

# difference: 9 to 10 epochs -> one training epoch.
data_sizes = [30, 150, 300,1000]
two_stage_time = [time_30_2, time_150_2, time_300_2,time_1000_2]
l2_time =  [time_30, time_150, time_300,time_1000]

time_30 = 11.8 - 11.3
time_150 = 51.089 - 49.663
time_300 = 03.083- 00.793
time_1000 = 17.73 - 11.828
time_30_2 = 40.305 - 40.303
time_150_2 = 19.282- 19.271
time_300_2 = 04.662- 04.638
time_1000_2 = 28.201 - 28.047

scatter(data_sizes, two_stage_time,label = "data size with time per epoch: two stage")
scatter!(data_sizes, l2_time,label="data size with time per epoch:L2")
savefig("data_size.pdf")
