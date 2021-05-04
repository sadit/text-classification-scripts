#!/bin/bash

lang=es
method=entropy
isample=128
nick=_pan20-$lang-$method-$isample


train=pan20/train-$lang.json
test=pan20/test-$lang.json
procs=64

export nick method train test procs isample
srun -c62 -xgeoint0 julia --project=. search-model.jl
params=$nick/params nick=$nick/model julia --project=. run.jl test
