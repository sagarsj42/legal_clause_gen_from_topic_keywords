cp ../path/to/topic-nostep-clause-kwds.c100.k200.lemstop.*.json .

torchrun --nproc_per_node=1 ../codes/train_kwd_to_clause_bart.py

torchrun --nproc_per_node=1 ../codes/train_kwd_to_clause_causal_generation.py

torchrun --nproc_per_node=1 ../codes/train_prompt_to_clause_causal_generation.py

torchrun --nproc_per_node=1 ../codes/train_topic_to_clause.py

torchrun --nproc_per_node=1 ../codes/train_random_kwd_to_clause.py

torchrun --nproc_per_node=1 ../codes/train_k_kwd_to_clause.py
