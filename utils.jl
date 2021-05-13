using CategoricalArrays, JSON3, Random
using MLDataUtils, StatsBase
using KCenters, KNearestCenters, TextSearch, TextClassification


function normstring(t)
    if occursin("&", t)
        t = replace(t, "&gt;" => ">")
        t = replace(t, "&lt;" => "<")
        t = replace(t, "&amp;" => "&")
    end

    replace(t, r"#(USER|URL|HASHTAG)#" => p->"_" * lowercase(p[2:end-1]))
end

function loadjson(filename, normstring=normstring; textkey="text", labelkey=nothing)
    corpus = []
    labels = []

    for line in readlines(filename)
        d = JSON3.read(line)
        t = d[textkey]
        if t isa AbstractString
            t = normstring(t)
        else
            t = [normstring(t[i]) for i in eachindex(t)]
        end
        push!(corpus, t)
        
        if labelkey !== nothing
            push!(labels, d[labelkey])
        end
    end
    
    (corpus=corpus, labels=categorical(labels), filename=filename, textkey=textkey, labelkey=labelkey)
end

function stratified_split(labels; p=0.7, shuffle=true)
    n = length(labels)
    X = collect(1:n)
    (train_X, train_labels), (test_X, test_labels) = stratifiedobs((X, labels), p=p, shuffle=shuffle)
    train_X, test_X
end

function stratified_folds(labels; folds=3, shuffle=true)
    n = length(labels)
    X = collect(1:n)
    shuffle && shuffle!(X)
    p = tuple([round(1/folds, RoundUp, digits=2) for i in 1:(folds-1)]...)
    L = stratifiedobs((X, labels), p=p)
    F = []
    for i in 1:folds
        P = L[[j for j in 1:folds if i != j]]
        itrain = vcat(getindex.(P, 1)...)
        itest = L[i][1]
        push!(F, (itrain, itest))
    end

    F
end

function savemodel(filename, data)
    open(filename, "w") do f
        println(f, string(typeof(data)))
        JSON3.write(f, data)
        println(f)
    end
end

function loadmodel(filename)
    open(filename, "r") do f
        type_ = readline(f)
        T = eval(Meta.parse(type_))
        JSON3.read(f, T)
    end
end

function saveparams(paramsfile, B)
    open(paramsfile, "w") do f
        for b in B
            println(f, b.second, "\t", typeof(b.first), "\t", JSON3.write(b.first))
        end
    end
end

function loadparams(paramsfile)
    L = []

    f = open(paramsfile)
    for line in eachline(f)
        arr = split(line, '\t')
        @assert length(arr) == 3
        score, type_ = parse(Float64, arr[1]), eval(Meta.parse(arr[2]))
        config = JSON3.read(arr[3], type_)
        push!(L, config => score)
    end

    close(f)

    L
end