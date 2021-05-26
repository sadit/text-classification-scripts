#!/bin/bash
isample=64
method=entropy
score=accuracy
lang=es
nick=_pan21-$lang-$method-$isample+$score+bsize=4

train=pan21/train-$lang.json
#train=pan21/train-$lang.json_train.json
#test=pan21/train-$lang.json_test.json
procs=64

export nick method train test procs isample score
srun -c62 -xgeoint0 julia --project=. search-model.jl
#params=$nick/params nick=$nick/model julia --project=. run.jl test
