- 目录树


    .
    ├── config
    │   ├── config.py
    │   └── ___init__.py
    ├── data
    │   ├── datasize
    │   ├── entity2id
    │   ├── id2entity
    │   ├── id2property
    │   ├── id2value
    │   ├── num_pos_neg_samples.json
    │   ├── origintext.txt
    │   ├── property2id
    │   ├── test
    │   ├── train
    │   ├── tripletid
    │   ├── triplettext.pkl
    │   ├── tupleid
    │   └── value2id
    ├── dataset.py
    ├── findrules
    │   └── find_rules.py
    ├── main.py
    ├── model.py
    ├── parameters
    │   ├── entity_embedding.weight.npy
    │   ├── fc1.bias.npy
    │   ├── fc1.weight.npy
    │   ├── fc2.bias.npy
    │   ├── fc2.weight.npy
    │   ├── fc3.bias.npy
    │   ├── fc3.weight.npy
    │   ├── fc4.bias.npy
    │   ├── fc4.weight.npy
    │   ├── fc5.bias.npy
    │   ├── fc5.weight.npy
    │   ├── fc6.bias.npy
    │   ├── fc6.weight.npy
    │   ├── model.pth
    │   └── rule_embedding.npy
    ├── readme.md
    └── train.py


- main.py
    - RuleE模型入口：训练时执行python main.py 
- train.py
    - 定义loss, optimizer等，并接入模型和dataloader进行训练
- model.py
    - RuleE模型
- dataset.py
    - 定义datadet类
- parameters
    - 其中各个参数文件在训练过程中会生成
- data
    - 数据文件，data_preprocess模块生成
- findrules
    - 模型训练完成后用于生成规则