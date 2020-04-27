import pickle

class config:
    def __init__(self):


        # 分段取max时候每一段的长度
        self.piecelen = 10

        # 一个很小的数，以防除0操作
        self.eps = 1e-8

        # 正样本的weight
        self.pos_weight = 2

        # 负样本的weight
        self.neg_weight = 1

        # 训练的epoch数量
        self.epochnum = 20

        # 训练时使用的gpu
        self.gpu = 1

        # 参数初始化的随机种子
        self.seed = 1978

        # 当一个property在一条规则下的得分大于一个阈值的时候，我们认为这个property是重要的
        self.threshold = 0.6

        # 一个instance最多包含的pv对的个数
        self.maxpv = 10

        # 学习率
        self.lr = 0.01

        # batch_size
        self.batch_size = 512

        # 规则的dim大小
        self.rule_embsize = 16



        # entity的dim大小
        self.entity_embsize = 16

        # 数据文件所在的文件夹(用绝对路径确保没有问题)
        self.datadir = "./data/"

        # 训练文件名

        self.trainfile = "train"

        # 测试文件名
        self.testfile = "test"

        # 训练集的大小
        self.traindatasize = len(pickle.load(open(self.datadir + "train", "rb")))

        # 测试集的大小
        self.testdatasize = len(pickle.load(open(self.datadir + "test", "rb")))

        # property + value 的个数
        self.entitynum = len(pickle.load(open(self.datadir + "entity2id", "rb")))

        # property的个数
        self.propertynum = len(pickle.load(open(self.datadir + "property2id", "rb")))

        # 规则的数量
        self.rulesnum = self.propertynum
        # self.rulesnum = 10

        # value 的个数
        self.valuenum = self.entitynum - self.propertynum


# 实例化
aliconfig = config()