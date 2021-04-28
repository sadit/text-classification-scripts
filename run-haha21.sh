#!/bin/bash
nick=_haha21
method=entropy

train=haha2021/haha_train.json
test=haha2021/haha_dev.json
procs=64

export nick method train test procs
#srun -c62 -xgeoint0 julia --project=. search-model.jl
params=$nick/params nick=$nick/model julia --project=. run.jl test
