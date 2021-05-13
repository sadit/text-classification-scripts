using KCenters, KNearestCenters, TextSearch, TextClassification, SearchModels, UnicodePlots, StatsBase
using LIBLINEAR

function score_by_name(score, y, ypred)
    if score === :macrorecall
        recall_score(y, ypred, weight=:macro)
    elseif score === :macrof1
        f1_score(y, ypred, weight=:macro)
    elseif score === :accuracy
        recall_score(y, ypred, weight=:macro)
    elseif score isa NamedTuple
        if score.name == :classf1
            classification_scores(y, ypred).classf1[score.label]
        elseif score.name == :classrecall
            classification_scores(y, ypred).classrecall[score.label]
        else
            error("Unknown score $score")
        end
    else
        error("Unknown score $score")
    end
end

function search_models_for_task(D, paramsfile, space, search_kwargs; verbose=true)
    cv = pop!(search_kwargs, :cv)

    if cv.kind == :folds
        partitions = stratified_folds(D.labels; folds=cv.folds, shuffle=true)
    elseif cv.kind == :montecarlo
        partitions = [stratified_split(D.labels; p=cv.ratio, shuffle=true) for i in 1:cv.repeat]
    elseif cv.kind == :holdout
        partitions = [stratified_split(D.labels; p=cv.ratio, shuffle=true)]
    else
        error("Unknown $cv")
    end

    score = pop!(search_kwargs, :score, :recall)
    initialpopulation = pop!(search_kwargs, :initialpopulation, 32)
    function error_function(config::MicroTC_Config)
        S = Float64[]
        for (itrain, itest) in partitions
            tc = MicroTC(config, D.corpus[itrain], D.labels[itrain]; verbose=true)
            ypred = predict_corpus(tc, D.corpus[itest])
            push!(S, score_by_name(score, D.labels[itest], ypred))
        end
        s = mean(S)
        verbose && println(stderr, "score: $s, $(typeof(config)), config: $(JSON3.write(config))")
        1.0 - s
    end

    iter = 0
    
    function inspect_population(space, params, population)
        iter += 1
        println(stderr, "===== inspecting population iter: $iter, popsize: $(length(population))")
        filename = paramsfile * ".iter=$iter"
        println(stderr, "saving iteration to $filename")
        saveparams(filename, population)
        println(lineplot(last.(population), title="Error", xlabel="configurations", ylabel="error"))
    end

    search_kwargs[:inspect_population] = inspect_population

    best_list = search_models(space, error_function, initialpopulation; search_kwargs...)

    for (i, b) in enumerate(best_list)
        @info "-- perf best_lists[$i]:", b[1] => b[2]
    end
    
    best_list
end

function run_params(D, paramsfile, space, search_kwargs)
    if !isfile(paramsfile)
        B = search_models_for_task(D, paramsfile, space, search_kwargs)
        saveparams(paramsfile, B)
        B
    else
        loadparams(paramsfile)
    end
end

function run_train(D, config, modelfile; tok=Tokenizer(config.textconfig, invmap=nothing))
    if !isfile(modelfile)
        cls = MicroTC(config, D.corpus, D.labels; tok)
        savemodel(modelfile, cls)
        cls
    else
        loadmodel(modelfile)
    end
end

function run_predict(T, cls, predictedfile; textkey="text", labelkey="klass", levels_=levels(T.labels))
    ypred = predict_corpus(cls, T.corpus)
    open(predictedfile, "w") do f
        for (message, ilabel) in zip(T.corpus, ypred)
            println(f, JSON3.write(Dict(textkey => message, labelkey => levels_[ilabel])))
        end
    end

    ypred
end


