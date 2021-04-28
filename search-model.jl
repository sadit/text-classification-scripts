using KCenters, KNearestCenters, TextSearch, TextClassification, SearchModels, UnicodePlots
using LIBLINEAR
using Distributed

const PROCS = parse(Int, get(ENV, "procs", "1"))

addprocs(PROCS, exeflags="--project=.")

@everywhere begin
    include("utils.jl")
    include("commands.jl")
end

function main_search_model(nick, train, textconfig, textmodel, cls)
    labelkey = get(ENV, "klass", "klass")
    textkey = get(ENV, "text", "text")

    # search hyper-parameters
    search_kwargs = Dict(
        :initialpopulation => 64,
        :cv => (kind=:folds, folds=3),
        # :cv => (kind=:montecarlo, repeat=15, ratio=0.5),
        #:cv => (kind=:montecarlo, repeat=20, ratio=0.7),
        :maxpopulation => 32,
        :bsize => 4,
        :mutbsize => 64,
        :crossbsize => 0,
        :tol => 0.0,
        :maxiters => 20,
        :verbose => true,
        :parallel => (PROCS>1 ? :distributed : :none)
        #:parallel => :threads
    )

    space = MicroTC_ConfigSpace(textconfig=textconfig, textmodel=textmodel, cls=cls)
    outdir = nick
    !isdir(outdir) && mkdir(outdir)
    paramsfile = "$outdir/params"
    D = loadjson(train; textkey=textkey, labelkey=labelkey)
    run_params(D, paramsfile, space, search_kwargs)
end

if !isinteractive()
    textconfig = TextConfigSpace(
        del_diac=[true, false],
        del_dup=[true, false],
        del_punc=[true, false],
        group_num=[true, false],
        group_url=[true, false],
        group_usr=[true, false],
        group_emo=[true, false],
        lc=[true, false],
        qlist=[[3], [3, 5], [3, 7], []],
        nlist=[[1], [1, 2], [2], [1, 2, 3], []],
        slist=[[Skipgram(2,1)], []],
        # nlist_space = [1, 2, 3],
        # qlist_space = [3, 5, 7],
        # slist_space = []
    )

    #llspace = LiblinearConfigSpace()
    #knnspace = KnnClassifierConfigSpace()
    kncspace = KncPerClassConfigSpace{0.5}(
        centerselection=[TextCentroidSelection()],
        kernel=[k_(CosineDistance()) for k_ in [DirectKernel]]
    )
    llspace = LiblinearConfigSpace(scale_C=nothing, scale_eps=nothing)
    knnspace = KnnClassifierConfigSpace(k=[1, 3], scale_k=nothing)
    entspace = EntModelConfigSpace(
        weights=[:balance],
        #local_weighting=[FreqWeighting(), TpWeighting()],
        #local_weighting=[BinaryLocalWeighting()],
        #scale_mindocs=nothing,
        #scale_smooth=nothing
    )
    vecspace = VectorModelConfigSpace()
    # classifiers = [llspace, knnspace]
    classifiers = [llspace]
    # classifiers = [kncspace]
    method = get(ENV, "method", "microtc")
    train = ENV["train"]
    nick = get(ENV, "nick", "_$(method)_" * replace(basename(train), ".json" => ""))

    if method == "entropy"
        main_search_model(nick, train, textconfig, entspace, classifiers)
    elseif method == "no-entropy"
        main_search_model(nick, train, textconfig, vecspace, classifiers)
    elseif method == "microtc"
        main_search_model(nick, train, textconfig, [vecspace, entspace], classifiers)
    else
        error("unknown method $method")
    end
end
