# GCN-BSI

A Graph Convolutional Network to Improve Item Recommendation by Incorporating Bundle-based Side Information with Multi-level Propagations. 


## Bundle Info

### Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.2.0
* numpy == 1.17.4
* scipy == 1.4.1
* temsorboardX == 2.0

### Usage
The hyperparameter search range and optimal settings have been clearly stated in the codes (see the 'CONFIG' dict in config.py).
* Train

```
python main.py 
```

* Futher Train

Replace 'sample' from 'simple' to 'hard' in CONFIG and add model file path obtained by Train to 'conti_train', then run
```
python main.py 
```

* Test

Add model path obtained by Futher Train to 'test' in CONFIG, then run
```
python eval_main.py 
```

Some important hyperparameters:
* `lrs`
  * It indicates the learning rates. 
  * The learning rate is searched in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}.

* `mess_dropouts`
  * It indicates the message dropout ratio, which randomly drops out the outgoing messages. 
  * We search the message dropout within {0, 0.1, 0.3, 0.5}.

* `node_dropouts`
  * It indicates the node dropout ratio, which randomly blocks a particular node and discard all its outgoing messages. 
  * We search the node dropout within {0, 0.1, 0.3, 0.5}.

* `decays`
  * we adopt L2 regularization and use the decays to control the penalty strength.
  * L2 regularization term is tuned in {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}.

* `hard_window`
  * It indicates the difficulty of sampling in the hard-negative sampler.
  * We set it to the top thirty percent.

* `hard_prob`
  * It indicates the probability of using hard-negative samples in the further training stage.
  * We set it to 0.8 (0.4 in the item level and 0.4 in the bundle level), so the probability of simple samples is 0.2.

### Dataset
We provide two processed dataset: Netease, steamGame.

* `user_bundle_train.txt`
  * Train file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.

* `user_item.txt`
  * Train file.
  * Each line is 'userID\t itemID\n'.
  * Every observed interaction means user u once interacted item i. 

* `bundle_item.txt`
  * Train file.
  * Each line is 'bundleID\t itemID\n'.
  * Every entry means bundle b contains item i.

* `Netease_data_size.txt`
  * Assist file.
  * The only line is 'userNum\t bundleNum\t itemNum\n'.
  * The triplet denotes the number of users, bundles and items, respectively.

* `user_bundle_tune.txt`
  * Tune file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.

* `user_bundle_test.txt`
  * Test file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.
  
## Level Info

### Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.8.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

### Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).
* Gowalla dataset
```
python NGCF.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

* Amazon-book dataset
```
python NGCF.py --dataset amazon-book --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 200 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```
Some important arguments:
* `alg_type`
  * It specifies the type of graph convolutional layer.
  * Here we provide three options:
    * `ngcf` (by default), proposed in [Neural Graph Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir19-NGCF.pdf), SIGIR2019. Usage: `--alg_type ngcf`.
    * `gcn`, proposed in [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl), ICLR2018. Usage: `--alg_type gcn`.
    * `gcmc`, propsed in [Graph Convolutional Matrix Completion](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf), KDD2018. Usage: `--alg_type gcmc`.

* `adj_type`
  * It specifies the type of laplacian matrix where each entry defines the decay factor between two connected nodes.
  * Here we provide four options:
    * `ngcf` (by default), where each decay factor between two connected nodes is set as 1(out degree of the node), while each node is also assigned with 1 for self-connections. Usage: `--adj_type ngcf`.
    * `plain`, where each decay factor between two connected nodes is set as 1. No self-connections are considered. Usage: `--adj_type plain`.
    * `norm`, where each decay factor bewteen two connected nodes is set as 1/(out degree of the node + self-conncetion). Usage: `--adj_type norm`.
    * `gcmc`, where each decay factor between two connected nodes is set as 1/(out degree of the node). No self-connections are considered. Usage: `--adj_type gcmc`.

* `node_dropout`
  * It indicates the node dropout ratio, which randomly blocks a particular node and discard all its outgoing messages. Usage: `--node_dropout [0.1] --node_dropout_flag 1`
  * Note that the arguement `node_dropout_flag` also needs to be set as 1, since the node dropout could lead to higher computational cost compared to message dropout.

* `mess_dropout`
  * It indicates the message dropout ratio, which randomly drops out the outgoing messages. Usage `--mess_dropout [0.1,0.1,0.1]`.

### Dataset
We provide two processed datasets: Gowalla and Amazon-book.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.

### Acknowledgement

This research is supported by the National Research Foundation, Singapore under its International Research Centres in Singapore Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

### Citation 

Our code is modified on the basis of the following two papers.

```
@inproceedings{BGCN20,
  author    = {Jianxin Chang and 
               Chen Gao and 
               Xiangnan He and 
               Depeng Jin and 
               Yong Li},
  title     = {Bundle Recommendation with Graph Convolutional Networks},
  booktitle = {Proceedings of the 43nd International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2020, Xi'an,
               China, July 25-30, 2020.},
  year      = {2020},
}
@inproceedings{NGCF19,
  author    = {Xiang Wang and
               Xiangnan He and
               Meng Wang and
               Fuli Feng and
               Tat{-}Seng Chua},
  title     = {Neural Graph Collaborative Filtering},
  booktitle = {Proceedings of the 42nd International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2019, Paris,
               France, July 21-25, 2019.},
  pages     = {165--174},
  year      = {2019},
}
```
