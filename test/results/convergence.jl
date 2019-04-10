plot(((Dates.value.(sa.times)).*(10^(-9))).-Dates.value(sa.times[1]).*(10^(-9)), sa.losses, label = "twostage loss over time")
plot(sa.times, sa.losses, label = "l2 loss over time")

savefig("l2_conv.pdf")
plot(((Dates.value.(sa_l2.times)).*(10^(-9))).-Dates.value(sa_l2.times[1]).*(10^(-9)), sa_l2.losses, label = "l2 loss over time")

using BSON,Dates,Plots

bson("DiffEqFlux.jl/test/results/mix/sa_mix_"*name*"_epochs.bson", Dict(:times => sa.times, :losses => sa.losses))

l2 = BSON.load("DiffEqFlux.jl/test/results/l2/sa_l2_1000_epochs.bson")
mix = BSON.load("DiffEqFlux.jl/test/results/mix/sa_mix_3000_8_1_40_20_epochs.bson")
two = BSON.load("DiffEqFlux.jl/test/results/twostage/sa_two_169555_epochs.bson")



l2_a =(Dates.value.(l2[:times]).*(10^(-9))).-Dates.value(l2[:times][1]).*(10^(-9))
mix_a =(Dates.value.(mix[:times]).*(10^(-9))).-Dates.value(mix[:times][1]).*(10^(-9))
two_a =(Dates.value.(two[:times]).*(10^(-9))).-Dates.value(two[:times][1]).*(10^(-9))
precompiler_time = mix_a[3001]-mix_a[3000]
for i in 1:length(mix_a)
    if i>3000
        mix_a[i]= mix_a[i]-precompiler_time
    end
end

l2_b = l2[:losses]
mix_b = mix[:losses]
two_b = two[:losses]

plot(two_a[1:207:end],log.(two_b[1:207:end].+1),linecolor ="#ff8533",label = "collocation", grid=false, xlabel = "time [sec]", ylabel = " error" )
plot!(l2_a[1:820],log.(l2_b[1:820].+1),label = "L2",linecolor = "#668cff")
plot!(mix_a[1:5:end],log.(mix_b[1:5:end].+1),label = "two stage",linecolor = "#ffd39b")

length(two_a[1:207:end])
length(l2_a[1:820])
length(mix_a[1:10:8494])
u=((Dates.value.(sa.times)).*(10^(-9))).-Dates.value(sa.times[1]).*(10^(-9))
v=sa.losses

for i in 1:length(mix_a)
    if mix_a[i]>283
        print(i)
        break
    end
end

plot(u[1:207:end],log.(v[1:207:end].+1),linecolor ="#ff8533",label = "collocation", grid=false, xlabel = "time [sec]", ylabel = " error" )
plot!(a[1:820],log.(b[1:820].+1),label = "L2",linecolor = "#668cff")

u[1:207:end]

savefig("280sec_loss_drop_mix.pdf")

plot(u[1:2:end],log.(v[1:2:end].+1),linecolor ="#ff8533",label = "mix", grid=false, xlabel = "time [sec]", ylabel = " error" )


plot(sa.times[1:2:end],log.(sa.losses[1:2:end].+1))
plot(sa.times[1:8000],sa.losses[1:8000])
plot(sa.times[8000:end],sa.losses[8000:end])

log(sa.losses[820]+1)


mix_a =(Dates.value.(sa.times).*(10^(-9))).-Dates.value(sa.times[1]).*(10^(-9))
mix_b = sa.losses

mix_a[8000]
460-mix_a[8001]
