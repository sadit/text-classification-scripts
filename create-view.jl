include("utils.jl")
include("commands.jl")
include("inspect-vectors.jl")
using JLD2

function create_prototypes(trainfile, paramsfile)
    D = loadjson(trainfile; textkey="text", labelkey="klass")
    ncenters_per_class = 30
    kind=:fft
    kncconfig = KncProtoConfig(
        ncenters=-ncenters_per_class,
        maxiters=10,
        initial_clusters=kind,
        kernel=DirectKernel(CosineDistance()),
        centerselection=TextCentroidSelection()
    )
    best = load(paramsfile, "population") |> first |> first
    textconfig = best.textconfig # copy(best.textconfig, nlist=[3])
    config = MicroTC_Config(textconfig, best.textmodel, kncconfig)
    model = MicroTC(config, D.corpus, D.labels; tok=Tokenizer(config.textconfig))
    jldsave("H.model"; model)
    model
    # savemodel("H.model", model)
end

#model = load("H.model", "model")
#model = create_prototypes("pan21/train-es.json_train.json", "_pan21-es-microtc-64+acc+bsize=4/params")
