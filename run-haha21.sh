#!/bin/bash
isample=128
method=entropy
score=macrof1
bsize=4
nick=_haha21-$method-$isample+$score+bsize=$bsize

train=haha2021/haha_train.json
test=haha2021/haha_dev.json
procs=64

export nick method train test procs isample
srun -c62 -xgeoint0 julia --project=. search-model.jl
params=$nick/params nick=$nick/model julia --project=. run.jl test
