#!/bin/bash
isample=64
method=entropy
lang=en
nick=_pan21-$lang-$method-$isample+acc+bsize=16

train=pan21/train-en.json_train.json
test=pan21/train-en.json_test.json
procs=64

export nick method train test procs isample
srun -c62 -xgeoint0 julia --project=. search-model.jl
params=$nick/params nick=$nick/model julia --project=. run.jl test
