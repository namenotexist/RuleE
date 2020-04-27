import torch
from model import RuleE
from dataset import TrainDataset, TestDataset
from config import aliconfig
from train import train
import random
import numpy as np
def set_seed(seed = 2018):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# 设置随机种子，保证可复现
seed = aliconfig.seed
set_seed(seed)
# 设置使用的gpu
gpu = aliconfig.gpu
torch.cuda.set_device(gpu)
# 开始训练
model = RuleE(aliconfig).cuda()
print("Loading Train and Test Dataset.........")
traindata = TrainDataset(config=aliconfig)
testdata = TestDataset(config=aliconfig)
print("Loading Train and Test Dataset Done.........")
train(aliconfig, model, traindata, testdata)

