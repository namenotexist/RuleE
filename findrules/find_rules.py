import torch
import json
import numpy as np
import pickle
from time import time


import sys
sys.path.append("..")
from config import aliconfig
import torch.nn.functional as F
import os
from collections import defaultdict

# 获得所有property的id
def getpropertyid():
    propertyid = list(range(aliconfig.propertynum))
    return propertyid

# 获得训练集和测试集的大小
def readdatasize():
    traindatasize, testdatasize = aliconfig.traindatasize, aliconfig.testdatasize
    return traindatasize, testdatasize

# 获得正样本和负样本的个数
def read_pos_neg_samples():
    if not os.path.exists(aliconfig.datadir + "num_pos_neg_samples.json"):
        pos_samples, neg_samples = 0, 0
        tupleid = pickle.load(open(aliconfig.datadir + "tupleid", "rb"))
        for sample in tupleid:
            if sample[-1] == 0:
                neg_samples += 1
        pos_samples = len(tupleid) - neg_samples
        pos_and_neg_samples = [pos_samples, neg_samples]
        json.dump(pos_and_neg_samples, open(aliconfig.datadir+"num_pos_neg_samples.json","w"))
    pos_samples, neg_samples = json.load(open(aliconfig.datadir+"num_pos_neg_samples.json","r"))
    return pos_samples, neg_samples


# 获得每个property的所有可能的value
def readcandidate():
    """
        时间复杂度：O(samplesnum)
    """
    if not os.path.exists("candidates.json"):
        tripletid = pickle.load(open(aliconfig.datadir + "tripletid", "rb"))
        candidates = defaultdict(set)
        for instance in tripletid:
            for triplet in instance:
                if triplet in [0, 1]:
                    continue
                candidates[triplet[0]].add(triplet[1])
                candidates[triplet[0]].add(triplet[2])
        for key, val in candidates.items():
            candidates[key] = list(val)
        json.dump(candidates, open("candidates.json", 'w', encoding='utf-8'), ensure_ascii=True, indent=4)
    candidates = json.load(open("candidates.json", 'r'))
    return candidates

# 读取id2entity
def readid2entity(filename = "id2entity"):
    id2entity = pickle.load(open(aliconfig.datadir + filename, "rb"))
    return id2entity

# 读取entity2id
def readentity2id(filename = "entity2id"):
    entity2id = pickle.load(open(aliconfig.datadir + filename, "rb"))
    return entity2id

# 读取tupleid文件
def readtupleid(filename = "tupleid"):
    tupleid = pickle.load(open(aliconfig.datadir + filename, "rb"))
    return tupleid

# 获得rule关注的property
def read_property_importance():
    """
    该函数的目的是计算每条规则关注property具体是哪些
        property importance: [0.3, 0.9, 0.8, 0.7, ..., 1.0]
    """
    if not os.path.exists("../parameters/property_importance.pkl"):
        rules_embedding = torch.from_numpy(np.load("../parameters/rule_embedding.npy"))
        entity_embedding = torch.from_numpy(np.load("../parameters/entity_embedding.weight.npy"))
        propertyid = getpropertyid()
        property_embedding = entity_embedding[torch.LongTensor(propertyid)]
        # 归一化
        rules_embedding = F.normalize(rules_embedding, p=2, dim=-1)
        property_embedding = F.normalize(property_embedding, p=2, dim=-1)

        W1 = torch.from_numpy(np.load("../parameters/fc1.weight.npy")).transpose(1,0)
        b1 = torch.from_numpy(np.load("../parameters/fc1.bias.npy"))
        W4 = torch.from_numpy(np.load("../parameters/fc4.weight.npy")).transpose(1,0)
        b4 = torch.from_numpy(np.load("../parameters/fc4.bias.npy"))

        rules_embedding = rules_embedding.unsqueeze(1).repeat(1, aliconfig.propertynum, 1)
        property_embedding = property_embedding.unsqueeze(0).repeat(aliconfig.rulesnum, 1, 1)
        p_concat_rule = torch.cat(
            (property_embedding, rules_embedding), -1)
        property_importance = torch.matmul(p_concat_rule, W1) + b1
        property_importance = F.relu(property_importance)
        property_importance = torch.matmul(property_importance, W4) + b4
        property_importance = torch.sigmoid(property_importance)
        property_importance = property_importance.squeeze()
        pickle.dump(property_importance.numpy().tolist(), open("../parameters/property_importance.pkl","wb"))
    property_importance = pickle.load(open("../parameters/property_importance.pkl","rb"))
    return property_importance


# 解析规则
def read_rule():
    """
    该函数的目的是抽取出如下的规则
        rule:{
            "品牌": [
                "鼎缘",
                "鼎盛李白",
                "黄龙",
                "黄飞红",
                "黄尾袋鼠",
                "麦高瑞",
                "麦香皇",
                "麦百热",
                "鹤庆乾",
                "鹤兴",
                "鸿云",
            ],
            "系列": "相同"
            }
    """
    print("=====================begin extract rule==========================")
    begin_time = time()
    filedir = "../parameters"
    rules_embedding = torch.from_numpy(np.load(filedir + "/rule_embedding.npy"))
    # 归一化
    rules_embedding = F.normalize(rules_embedding, p=2, dim=-1)

    entity_embedding = torch.from_numpy(np.load(filedir + "/entity_embedding.weight.npy"))
    # 归一化
    entity_embedding = F.normalize(entity_embedding, p=2, dim=-1)

    W1 = torch.from_numpy(np.load(filedir+"/fc1.weight.npy")).transpose(1,0)
    b1 = torch.from_numpy(np.load(filedir+"/fc1.bias.npy"))
    W2 = torch.from_numpy(np.load(filedir+"/fc2.weight.npy")).transpose(1,0)
    b2 = torch.from_numpy(np.load(filedir+"/fc2.bias.npy"))
    W3 = torch.from_numpy(np.load(filedir+"/fc3.weight.npy")).transpose(1,0)
    b3 = torch.from_numpy(np.load(filedir+"/fc3.bias.npy"))
    W4 = torch.from_numpy(np.load(filedir+"/fc4.weight.npy")).transpose(1,0)
    b4 = torch.from_numpy(np.load(filedir+"/fc4.bias.npy"))
    W5 = torch.from_numpy(np.load(filedir + "/fc5.weight.npy")).transpose(1, 0)
    b5 = torch.from_numpy(np.load(filedir + "/fc5.bias.npy"))
    W6 = torch.from_numpy(np.load(filedir + "/fc6.weight.npy")).transpose(1, 0)
    b6 = torch.from_numpy(np.load(filedir + "/fc6.bias.npy"))
    propertyid = getpropertyid()
    property_embedding = entity_embedding[torch.LongTensor(propertyid)]

    candidates = readcandidate()
    avg_property = 0

    """
        时间复杂度O(rulenum * property_value_num)
    """
    textrules = []
    idrules = []
    # for i in range(100):
    for i in range(aliconfig.rulesnum):

        print("extract rule {}".format(i))
        textrule = {}
        idrule = {}
        """
            计算当前规则下重要的property
        """
        # (propertynum, entity_embsize)
        rule_embedding = rules_embedding[i].unsqueeze(0).repeat(aliconfig.propertynum, 1)

        # (propertynum, rule_embsize + entity_embsize)
        p_concat_rule = torch.cat(
            (property_embedding, rule_embedding), -1)
        # (propertynum, )

        s1 = torch.matmul(p_concat_rule, W1) + b1
        s1 = F.relu(s1)
        s1 = torch.sigmoid(torch.matmul(s1, W4) + b4).squeeze()

        avg_property += torch.sum((s1 > aliconfig.threshold))

        # (propertynum, )
        importantproperty = torch.where(s1 > aliconfig.threshold, torch.tensor(propertyid),(torch.ones(aliconfig.propertynum)*(-1)).long())

        importantproperty = importantproperty.numpy().tolist()

        importantproperty = [_ for _ in importantproperty if _ != -1.]

        id2entity = readid2entity()



        for property in importantproperty:
            """
                依次计算重要的property下的两个商品的属性值应该是“相同”还是“某些具体的值”
            """
            if str(property) in candidates.keys():
                vids = candidates[str(property)]
                pemb = property_embedding[property]
                remb = rules_embedding[i]
                p_concat_rule = torch.cat((pemb, remb))
                p = torch.matmul(p_concat_rule, W2) + b2
                p = F.relu(p)
                p = torch.sigmoid(torch.matmul(p, W5) + b5).squeeze()
                """
                    在这个重要的property下，两个商品的属性值应该取“相同”
                """
                if p > 0.9:
                    textrule[id2entity[property]] = "相同"
                    idrule[property] = []
                    continue
                """
                    在这个重要的property下，两个商品的属性值应该取某些具体的值
                """
                ground_value = torch.matmul(p_concat_rule, W3) + b3
                ground_value = F.relu(ground_value)
                ground_value = F.normalize(torch.matmul(ground_value, W6) + b6, p=2, dim=-1)

                idtopkvalue = []
                texttopkvalue = []
                for vid in vids:
                    vemb = entity_embedding[vid]

                    s2 = (0.5 * torch.cosine_similarity(vemb.unsqueeze(0), ground_value.unsqueeze(0)) + 0.5).item()
                    if s2 >= 0.9:
                        idtopkvalue.append(vid)
                        texttopkvalue.append(id2entity[vid])
                if texttopkvalue == []:
                    continue
                textrule[id2entity[property]] = texttopkvalue
                idrule[property] = idtopkvalue
        textrules.append(textrule)
        idrules.append(idrule)
    avg_property = avg_property.item()
    avg_property = [avg_property / aliconfig.rulesnum]
    json.dump(avg_property, open("avg_property.json","w", encoding='utf-8'))
    print("===================={:.2f}s elpased, end extract rule=====================".format(time() - begin_time))
    print("每条规则关注的property的个数: \n", avg_property)
    return textrules, idrules


# 计算规则的support、confidence、headcoverage
def cal_support_confidence_headcoverage():

    """
        计算一条规则能过多少个样本
    """


    def is_sample_get_through_one_rule(sample, rule):
        """
            计算一个样本能不能过一条规则
            sample:
                    [[52632, 139],
                    [41318, -1],
                    [2, 142],
                    [26742, 143],
                    [4, 144],
                    [5, -1],
                    [6, 147],
                    0]
            rule:
                {
                "品牌": [
                    "鼎缘",
                    "鼎盛李白",
                    "黄龙",
                    "黄飞红",
                    "黄尾袋鼠",
                    "麦高瑞",
                    "麦香皇",
                    "麦百热",
                    "鹤庆乾",
                    "鹤兴",
                    "鸿云",
                    ],
                "系列": "相同"
                }

                返回值有两个，第一个是规则的body是否覆盖样本，
                第二个值是规则的body和head是否完全覆盖样本
        """
        itemproperty2value = {}
        for property_value_pair in sample[:-1]:
            itemproperty2value[property_value_pair[0]] = property_value_pair[1]
        for property, values in list(rule.items()):
            if property not in itemproperty2value.keys():
                return 0, 0
        for property, values in list(rule.items()):
            if values != [] and itemproperty2value[property] not in values:
                return 0, 0
        if sample[-1] == 1:
            return 1, 1
        else:
            return 1, 0
    f = open("rules.txt", "w",encoding='utf-8')

    if not os.path.exists("statistics_rules.json"):
        textrules, idrules = read_rule()
        samples = readtupleid()
        pos_samples, neg_samples = read_pos_neg_samples()
        statistics_rules = []
        for i, rule in enumerate(idrules):
            if rule == {}:
                continue
            print("process rule{}".format(i))
            bodys, bodyandheads = 0, 0
            for sample in samples:
                body, bodyandhead = is_sample_get_through_one_rule(sample, rule)
                bodys += body
                bodyandheads += bodyandhead
            support = bodyandheads / (pos_samples + neg_samples + 1)
            confidence = bodyandheads / (bodys + 1)
            head_coverage = bodyandheads / (pos_samples + 1)
            statistics_rules.append([support, confidence, head_coverage, textrules[i]])
            f.write(str([support, confidence, head_coverage, textrules[i]]))
            f.write("\n")
        json.dump(statistics_rules, open("statistics_rules.json", "w"),ensure_ascii=False,indent=4)
    f.close()
    statistics_rules = json.load(open("statistics_rules.json", "r"))
    return statistics_rules



if __name__ == "__main__":
    cal_support_confidence_headcoverage()


