using CategoricalArrays, JSON3, JLD2, Random
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
    
    (corpus=corpus, labels=length(labels) > 0 ? categorical(labels) : nothing, filename=filename, textkey=textkey, labelkey=labelkey)
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