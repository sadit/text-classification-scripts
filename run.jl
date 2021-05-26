using TextClassification
using LIBLINEAR, KNearestCenters

include("utils.jl")
include("commands.jl")

function help_and_exit()
    println("""
usage: ARGS julia --project=. run.jl (test|train|predict|scores)

test command:
    Runs a full test procedure (training, predicting and scoring).
    
    arguments:
        params=[configuration file from search models, it will use the best config]
        train=[training dataset]
        test=[test dataset]
        nick=[filename prefix to output *all* files]

train command:
    Creates a model from a configuration and a given configuration
    
    arguments:
        model=[filename to output model]
        params=[configuration file from search models, it will use the best config]
        train=[training dataset]

predict command:
    Applies a model to a set of messages

    arguments:
        model=[filename containing the model]
        test=[test dataset]
        pred=[predictions output filename]

scores command:
    Computes several quality scores for prediction using a gold standard

    arguments:
        pred=[filename with the predictions]
        gold=[filename with the goldstandard]
        scores=[output filename for the scores]
    """)

    exit(0)
end

function predict(model::MicroTC, D, predictedfile, textkey, labelkey; raw=false)
    ypred = predict_corpus(model, get.(D, textkey, nothing))
    open(predictedfile, "w") do f
        if raw
            for y in ypred
                println(f, y)
            end
        else
            for i in eachindex(D)
                d = D[i]
                d[labelkey] = ypred[i]
                println(f, JSON3.write(d))
            end
        end
    end

    ypred
end

function predict(modelfile::AbstractString, testfile::AbstractString, predictedfile, textkey, labelkey; raw=false)
    model = load(modelfile, "model")
    D = [JSON3.read(line, Dict) for line in eachline(testfile)]
    predict(model, D, predictedfile, textkey, labelkey; raw=raw)
end

function create_model(paramsfile::AbstractString, trainfile::AbstractString, modelfile, textkey, labelkey)
    params = load(paramsfile, "population")
    D = loadjson(trainfile; textkey=textkey, labelkey=labelkey)
    run_train(D, params[1][1], modelfile)
end

function run_scores(ygold::AbstractVector, ypred::AbstractVector, scoresfile)
    scores = classification_scores(ypred, ygold)

    open(scoresfile, "w") do f
        s = JSON3.write(scores)
        println(stdout, s)
        println(f, s)
    end

    scores
end

function run_scores(gold::AbstractString, predicted::AbstractString, scoresfile, textkey, labelkey)
    ygold = loadjson(gold; textkey, labelkey)
    ypred = loadjson(predicted; textkey, labelkey)
    run_scores(ygold.labels, ypred.labels, scoresfile)
end

function run_test(paramsfile, trainfile, testfile, nick)
    traindata = loadjson(trainfile; textkey=textkey, labelkey=labelkey)
    testdata = loadjson(testfile; textkey=textkey, labelkey=labelkey)
    params = load(paramsfile, "population")
    modelfile = nick * ".model"
    predictedfile = nick * ".predicted"
    scoresfile = nick * ".scores"
    model = run_train(traindata, params[1][1], modelfile)
    ypred = predict(model, testdata, predictedfile, textkey, labelkey)
    run_scores(testdata.labels, ypred, scoresfile)
end

if !isinteractive()
    command = nothing
    if length(ARGS) > 0
        command = ARGS[1]
    end

    command === nothing && help_and_exit() 

    textkey = get(ENV, "text", "text")
    labelkey = get(ENV, "klass", "klass")

    if command == "test"
        paramsfile = ENV["params"]
        trainfile = ENV["train"]
        testfile = ENV["test"]
        nick = get(ENV, "nick", nothing)
        if nick === nothing
            nick = "__" * replace(basename(paramsfile), ".params" => "")
        end       
        run_test(paramsfile, trainfile, testfile, nick)
    elseif command == "train"
        modelfile = ENV["model"]
        paramsfile = ENV["params"]
        trainfile = ENV["train"]
        create_model(paramsfile, trainfile, modelfile, textkey, labelkey)
    elseif command == "predict"
        modelfile = ENV["model"]
        testfile = ENV["test"]
        predictedfile = ENV["pred"]
        predict(modelfile, testfile, predictedfile, textkey, labelkey; raw=false)
    elseif command == "scores"
        predicted = ENV["pred"]
        gold = ENV["gold"]
        scoresfile = ENV["scores"]
        run_scores(predicted, gold, scoresfile, textkey, labelkey)
    else
        help_and_exit()
    end
end
