# Graph-based Keyword Planning for Legal Clause Generation from Topics

Source code for our paper to be published at the Natural Legal Language Processing Workshop, EMNLP 2022.  
We would update the link to the paper here once available.
You can find the camera ready version [here](https://iiitaphyd-my.sharepoint.com/:b:/g/personal/sagar_joshi_research_iiit_ac_in/EcR2sgf2HL1Kp6QNKRRFu84B1y7IepqZleor7uHxGB85Xg?e=JTjffT).

## Reproducing the results

You'll find the scripts to run inside the ./scripts folder, and code for different components in other folders. Follow this order to prepare the workable subset from the LEDGAR corpus (download the LEDGAR dataset from [here](https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A).) We'll be making use of the cleaned corpus.

### Preparing the dataset and vocabular files
`bash ./scripts/run_create_dataset.sh`

### Create and serialize the adjacency matrix to be used for graph creation from extracted and topics in the dataset
`bash ./scripts/run_create_adjacency_matrix.sh`

### This script points to all the keyword to clause generation experiments in the paper:
```
bash ./scripts/run_train_clause_generation.sh
```

### Evaluate clause generation performance post training
```
bash ./scripts/run_evaluate_clause_generation.sh
```
