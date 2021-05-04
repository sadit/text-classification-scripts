include("utils.jl")
include("commands.jl")

D = loadjson("haha2021/haha_train.json"; textkey="text", labelkey="klass")
ncenters_per_class = 10
kind=:fft
kncconfig = KncProtoConfig(
    ncenters=-ncenters_per_class,
    maxiters=10,
    initial_clusters=kind,
    kernel=DirectKernel(CosineDistance()),
    centerselection=TextCentroidSelection()
)
params = loadparams("_haha21/params")
best = params[1][1]
config = MicroTC_Config(best.textconfig, best.textmodel, kncconfig)
run_train(D, config, "H.model", tok=Tokenizer(config.textconfig))