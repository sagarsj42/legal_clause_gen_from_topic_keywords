EXP_NAME='10-kwd-to-clause'

cp ../path/to/topic-nostep-clause-kwds.c100.k200.lemstop.test.json .

cp -r ../path/to/$EXP_NAME .

python ../codes/eval_clause_generation.py

# python ~/clause-generation/codes/clsgen/eval_causal_clause_generation.py

