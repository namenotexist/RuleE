import torch
from torch.utils.data import Dataset
import pickle


class TrainDataset(Dataset):
    def __init__(self, config):

        self.X = []
        self.Y = []
        self.len = []
        """
            padmask: [1, 1, 1, 0, 0]
            samemask: [1, 0, 1, 0, 0]
            说明该样本有3个pv对
            其中两个商品第1个和第3个属性对应的属性值是相同的，第二个属性的属性值是不同的
        """
        self.padmask = torch.zeros(config.traindatasize, config.maxpv)
        self.samemask = torch.zeros(config.traindatasize, config.maxpv)

        trainid = pickle.load(open(config.datadir + config.trainfile, "rb"))
        for i, instance in enumerate(trainid):
            """
                instance:
                    [[52632, 139], 
                    [41318, -1], 
                    [2, 142], 
                    [26742, 143], 
                    [4, 144], 
                    [5, -1], 
                    [6, 147],
                    0]
            """
            self.Y.append(torch.tensor(instance[-1], dtype=torch.long))
            x = instance[:-1]
            self.len.append(len(x))
            self.padmask[i][:len(x)] = 1
            x = x[:config.maxpv]
            if len(x) < config.maxpv:
                x = x + [[0, 0] for _ in range(config.maxpv - len(x))]
            x = torch.tensor(x, dtype=torch.long)
            self.samemask[i] = x[:, 1] > 0
            self.X.append(x)
        self.X = torch.stack(self.X, dim=0)
        self.Y = torch.stack(self.Y, dim=0)
        self.len = torch.tensor(self.len, dtype=torch.long)
        self.property = self.X[:, :, 0]
        self.value = self.X[:, :, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.property[i], self.value[i], self.samemask[i], self.padmask[i], self.Y[i]



class TestDataset(Dataset):
    def __init__(self, config):
        self.X = []
        self.Y = []
        self.len = []
        """
            padmask: [1,1,1,0,0]
            samemask: [1,0,1,0,0]
            说明该样本有3个pv对
            其中两个商品第1个和第3个属性对应的属性值是相同的，第二个属性的属性值是不同的
        """
        self.padmask = torch.zeros(config.testdatasize, config.maxpv)
        self.samemask = torch.zeros(config.testdatasize, config.maxpv)
        """
            instance
                [[52632, 139], 
                [41318, -1], 
                [2, 142], 
                [26742, 143], 
                [4, 144], 
                [5, -1], 
                [6, 147],
                0]     
        """
        testid = pickle.load(open(config.datadir + config.testfile, "rb"))
        for i, instance in enumerate(testid):
            self.Y.append(torch.tensor(instance[-1], dtype=torch.long))
            x = instance[:-1]
            self.len.append(len(x))
            self.padmask[i][:len(x)] = 1
            x = x[:config.maxpv]
            if len(x) < config.maxpv:
                x = x + [[0, 0] for _ in range(config.maxpv - len(x))]
            x = torch.tensor(x, dtype=torch.long)
            self.samemask[i] = x[:, 1] > 0
            self.X.append(x)
        self.X = torch.stack(self.X, dim=0)
        self.Y = torch.stack(self.Y, dim=0)
        self.len = torch.tensor(self.len, dtype=torch.long)
        self.P = self.X[:, :, 0]
        self.V = self.X[:, :, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.P[i], self.V[i], self.samemask[i], self.padmask[i], self.Y[i]

