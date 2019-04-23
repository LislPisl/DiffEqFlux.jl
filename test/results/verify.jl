using BSON,Dates,Plots

time =(Dates.value.(sa.times).*(10^(-9))).-Dates.value(sa.times[1]).*(10^(-9))
losses = sa.losses
l2s = sa.l2s
verified = count(i-> i!= 0,l2s)
verified_l2s = zeros(verified)
verified_times = zeros(verified)
two = BSON.load("DiffEqFlux.jl/test/results/twostage/sa_two_169555_epochs.bson")
l2 = BSON.load("DiffEqFlux.jl/test/results/l2/sa_l2_1000_epochs.bson")
l2_a = (Dates.value.(l2[:times]).*(10^(-9))).-Dates.value(l2[:times][1]).*(10^(-9))
two_a =(Dates.value.(two[:times]).*(10^(-9))).-Dates.value(two[:times][1]).*(10^(-9))
l2_times =
col_times =
l2_b = l2[:losses]
two_b = two[:losses]
end_epoch = [0]
for i in 1:length(l2_a)
    if l2_a[i]>time[end]
        end_epoch[1] = i
        break
    end
end
count_ver=[0]
for i in 1:length(l2s)
    if l2s[i]!= 0
        count_ver[1] = count_ver[1] + 1
        verified_l2s[count_ver[1]] = l2s[i]
        verified_times[count_ver[1]] = time[i]

    end
end
scatter(l2_a[1:17],log.(l2_b[1:17].+1),markersize=3,label = "L2 model: loss")
plot!(two_a[1:2:end_epoch[1]],log.(losses[1:2:end_epoch[1]].+1),linecolor ="#ff8533",xlabel="time",ylabel="error",label = "Collocation model: loss")
scatter!(verified_times,log.(verified_l2s.+1),linecolor ="green",label = "Collocation model: L2 to observation")
two_a[end_epoch[1]]
time[end]
plot!(two_a[1:207:end],log.(two_b[1:207:end].+1),linecolor ="#ff8533",label = "collocation", grid=false, xlabel = "time [sec]", ylabel = " error" )
savefig("1.pdf")

two_a[3800]

l2_a[17]
f(x) = x .!= 4
 x .!= 4
