
#using LIBLINEAR, UnicodePlots
#include("utils.jl")
#include("main.jl")

# USAGE: copy and paste the following line to work with this script
# include("utils.jl"); include("main.jl"); using Revise, LIBLINEAR, UnicodePlots; includet("inspect-models.jl")
#inspect("_entropy_pan20-en.params", "PAN20/en.train.json", "PAN20/en.test.json")

function vocab(config, dataname)
    D = loadjson(dataname, textkey="text", labelkey="klass")
    textconfig = config.textconfig 
    #textconfig = copy(config.textconfig, nlist=[3], qlist=[])
    config = MicroTC_Config(textconfig, config.textmodel, config.cls)
    c = MicroTC(config, D.corpus, D.labels, tok=Tokenizer(config.textconfig))
    Dict(decode(c.tok, k) => v.weight for (k, v) in c.textmodel.tokens)
end

function topk(vec, k=length(vec))
    s = sort!(collect(vec), by=p->p[end], rev=true)
    if k !== length(vec)
        s[1:k]
    else
        s
    end
end

function prune(vec, k::Integer)
    Dict(topk(vec, k))
end

function prune(vec, minw::Real)
    Dict(k => w for (k, w) in vec if w >= minw)
end

function merge(A, B)
    if length(B) < length(A)
        A, B = B, A
    end

    Dict(k => sqrt(v * B[k]) for (k, v) in A if haskey(B, k))
end

function plotweights(vec, k=0)
    L = sort!(collect(vec), by=p->p[end], rev=true)
    if k > 0
        L = L[1:k]
    end
    lineplot(last.(L))
end

function inspect(paramsfile, trainfile, testfile)
    P = loadparams(paramsfile)
    vtrain = vocab(P[1][1], trainfile)
    vtest = vocab(P[1][1], testfile)
    vtrain, vtest
end
