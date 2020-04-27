import pickle
import random

BASE_DIR = "../data/"


tupleid = pickle.load(open(BASE_DIR + "tupleid", "rb"))
DATASIZE = len(tupleid)
TRAINDATASIZE = int(DATASIZE * 0.8)
TESTDATASIZE = DATASIZE - TRAINDATASIZE
tripletid = pickle.load(open(BASE_DIR + "tripletid", "rb"))
property2id = pickle.load(open(BASE_DIR + "property2id", "rb"))
value2id = pickle.load(open(BASE_DIR + "value2id", "rb"))
entity2id = pickle.load(open(BASE_DIR + "entity2id", "rb"))
id2property = pickle.load(open(BASE_DIR + "id2property", "rb"))
id2value = pickle.load(open(BASE_DIR + "id2value", "rb"))
id2entity = pickle.load(open(BASE_DIR + "id2entity", "rb"))

"""
we random split the "tupleid" file and we can get two file, 
which is "train" and "test" respectively
"""

# 先随机打乱这个列表
random.seed(2019)
random.shuffle(tupleid)
# 分割
train = tupleid[: TRAINDATASIZE]
test = tupleid[TRAINDATASIZE + 1:]

# 存储文件
pickle.dump(train, open(BASE_DIR + "train", "wb"))
pickle.dump(test, open(BASE_DIR + "test", "wb"))
# 记录 datasize
pickle.dump([DATASIZE, TRAINDATASIZE, TESTDATASIZE], open(BASE_DIR + "datasize", "wb"))
