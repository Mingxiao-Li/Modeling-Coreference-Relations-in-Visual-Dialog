## PyTorch implementation for the EACL 2021 paper
### [Modeling Coreference Relations in Visual Dialog](https://www.aclweb.org/anthology/2021.eacl-main.290/)
* bert_base folder: implementation of baseline model
* bert_pos folder: implementation of baseline model + pos constraint
* bert_sen folder: implementation of baseline model + nearest preference constraint
* bert_pos_senemd folder: implementation of the best model (baseline model + two constraints)

### Set up dependencies 

`pip install -r dependencies.txt`

### Running code
##### Set up data path and parameters properly 
`run {bert_base/bert_pos/bert_sen/bert_pos_senemd}{train/evaluate}.py`

### Results on the [online leaderboard](https://eval.ai/web/challenges/challenge-page/518/leaderboard/1421)
![Image](https://github.com/Mingxiao-Li/Modeling-Coreference-Relations-in-Visual-Dialog/blob/master/results.png)
