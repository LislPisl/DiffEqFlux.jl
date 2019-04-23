using CSV


data = Array{Float64,2}(undef,2,8)

f =open("test/data/silk.csv")
close(f)
s = open("test/data/silk.csv",",") do file
    read(file, String)
end

data
c=[0]
open("test/data/silk.csv") do f
    for i in eachline(f)
        c[1]=
        data[:,]=
    end
end

a=2

times = Array{Float64,1}()
observations = Array{Float64,1}()

open("test/data/silk.csv") do f
    for i in enumerate(eachline(f))
      println(i[1], ",", i[2])
      append!(times,i[1])
      append!(observations,i[2])
    end
end
open("test/data/silk.csv") do f
    for i in enumerate(eachline(f))
      println(i[1], ": ", i[2])
      line = split(i[2],",")
      t = parse(Float64,line[1])
      o = parse(Float64,line[2])

    end
end
data = (times, observations)
a =split("4.939,2.432",",")

a=reshape(times,1,8)
append!(times,observations)
b=reshape(observations,1,8)
