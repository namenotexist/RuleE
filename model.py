import torch
import torch.nn as nn
import torch.nn.functional as F

class RuleE(nn.Module):
    def __init__(self, config):
        super(RuleE, self).__init__()
        self.rule_embsize = config.rule_embsize
        self.entity_embsize = config.entity_embsize
        self.rulenum = config.rulesnum
        self.piecelen = config.piecelen
        self.entitynum = config.entitynum
        self.propertynum = config.propertynum
        self.valuenum = config.valuenum
        self.maxpv = config.maxpv
        self.threshold = config.threshold
        self.rule_embedding = nn.Parameter(torch.FloatTensor(self.rulenum, self.rule_embsize), requires_grad=True)
        self.propertyids = torch.LongTensor(list(range(self.propertynum))).cuda()
        self.entity_embedding = nn.Embedding(self.entitynum, self.entity_embsize)
        self.fc1 = nn.Linear(self.rule_embsize + self.entity_embsize, self.entity_embsize)
        self.fc2 = nn.Linear(self.rule_embsize + self.entity_embsize, self.entity_embsize)
        self.fc3 = nn.Linear(self.rule_embsize + self.entity_embsize, self.entity_embsize)
        self.fc4 = nn.Linear(self.entity_embsize, 1)
        self.fc5 = nn.Linear(self.entity_embsize, 1)
        self.fc6 = nn.Linear(self.entity_embsize, self.entity_embsize)
        self.weight_init()

    def weight_init(self):
        """
            parameters initialization
        """
        # 正太分布
        torch.nn.init.normal_(self.rule_embedding, mean=0, std=1)
        torch.nn.init.normal_(self.entity_embedding.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.fc3.weight, mean=0, std=1)

    def forward(self, property, value, samemask, padmask):
        """
            参数
                property: (batch_size, maxpv)
                value: (batch_size, maxpv)
                samemask: (batch_size, maxpv)
                padmask: (batch_size, maxpv)
            返回值：
                total_score: (batch_size,)
                reg_rule_cos: (1, ) rule之间应该尽可能的不相似
                reg_property_importance: (1, ) 每条规则关注的property个数应该尽可能小
                rule_avg_property: (1,)
        """
        # 由于最后一个batch的样本数量可能不足预设的batch_size的大小，所以需要以这种方式获得当前batch的batch_size
        batch_size = property.size(0)
        # (batch_size, maxpv, entity_embsize)
        p_embedding = self.entity_embedding(property)
        v_embedding = self.entity_embedding(value)
        """
            embedding归一化
        """
        # (batch_size, maxpv, entity_embsize)
        p_embedding = F.normalize(p_embedding, p=2, dim=-1)
        v_embedding = F.normalize(v_embedding, p=2, dim=-1)
        rule_embedding = F.normalize(self.rule_embedding, p=2, dim=-1)
        # (batch_size, rulenum, maxpv)
        samemask = samemask.unsqueeze(1).repeat(1, self.rulenum, 1)
        # (batch_size, rulenum, maxpv, entity_embsize)
        p_embedding = p_embedding.unsqueeze(1).repeat(1, self.rulenum, 1, 1)
        v_embedding = v_embedding.unsqueeze(1).repeat(1, self.rulenum, 1, 1)
        """
            拼接property 和 rule的embedding获得各个模块的输入
        """
        # (batch_size, rulenum, maxpv, rule_embsize + entity_embsize)
        p_concat_rule = torch.cat(
            (p_embedding, rule_embedding.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self.maxpv, 1)), -1)
        """
            计算s_1
        """
        # (batch_size, rulenum, maxpv, entity_embsize)
        s1 = self.fc1(p_concat_rule)
        # (batch_size, rulenum, maxpv, entity_embsize)
        s1 = F.relu(s1)
        # (batch_size, rulenum, maxpv, 1)
        s1 = self.fc4(s1)
        # (batch_size, rulenum, maxpv, 1)
        s1 = torch.sigmoid(s1)
        # (batch_size, rulenum, maxpv)
        s1 = s1.squeeze()
        """
            计算p
        """
        # (batch_size, rulenum, maxpv)
        p = self.fc2(p_concat_rule)
        # (batch_size, rulenum, maxpv)
        p = F.relu(p)
        # (batch_size, rulenum, 1)
        p = self.fc5(p)
        # (batch_size, rulenum)
        p = torch.sigmoid(p).squeeze()
        """
            计算s_2
        """
        # (batch_size, rulenum, maxpv, entity_embsize)
        ground_v_embedding = self.fc3(p_concat_rule)
        # (batch_size, rulenum, maxpv, entity_embsize)
        ground_v_embedding = F.relu(ground_v_embedding)
        # (batch_size, rulenum, maxpv, entity_embsize)
        ground_v_embedding =  self.fc6(ground_v_embedding)
        # (batch_size, rulenum, maxpv, entity_embsize)
        ground_v_embedding =  F.normalize(ground_v_embedding, p=2, dim=-1)
        # (batch_size, rulenum, maxpv)
        s2 = 0.5 * torch.cosine_similarity(v_embedding,
                                           ground_v_embedding, dim=-1) + 0.5
        """
            计算score_ij
        """
        # (batch_size, rulenum, maxpv)
        score = (p + (1 - p) * s2)

        # 如果两个商品在当前属性下的属性值相同，那么该pv对的分数为s1 * score，否则为 1 - s1
        # (batch_size, rulenum, maxpv)
        score = torch.where(samemask == 1, s1 * score, 1.0 - s1)
        """
            计算K
        """
        # (propertynum, entity_embsize)
        property_embedding = self.entity_embedding(self.propertyids)
        # (propertynum, entity_embsize)
        property_embedding = F.normalize(property_embedding, p = 2, dim=-1)
        # (rulenum, propertynum, entity_embsize)
        property_embedding = property_embedding.unsqueeze(0).repeat(self.rulenum, 1, 1)
        # (rulenum, propertynum, rule_embsize + entity_embsize)
        property_importance = torch.cat((property_embedding, rule_embedding.unsqueeze(1).repeat(1, self.propertynum, 1)), dim=-1)
        # (rulenum, propertynum,entity_embsize)
        property_importance = self.fc1(property_importance)
        # (rulenum, propertynum,entity_embsize)
        property_importance = F.relu(property_importance)
        # (rulenum, propertynum,1)
        property_importance = self.fc4(property_importance)
        # (rulenum, propertynum)
        property_importance = torch.sigmoid(property_importance).squeeze()
        # (rulenum, propertynum)
        rule_mask = property_importance > self.threshold
        """
            计算score_i
        """
        # (rulenum, propertynum)
        rule_mask = property_importance * rule_mask
        # (batch_size, rulenum, propertynum)
        rule_mask = rule_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        # (batch_size, rulenum)
        rule_mask = torch.sum(rule_mask, dim=-1)  + 1

        # padding的property-value pair 的score_ij设置为0
        # (batch_size, rulenum, propertynum)
        padmask = padmask.unsqueeze(1).repeat(1, self.rulenum, 1)
        # (batch_size, rulenum, propertynum)
        score = score * padmask

        # 如果该属性是不重要的属性，那么score_ij直接设置为0
        score = score * (s1 > self.threshold)
        # (batch_size, rulenum)
        score = score.sum(-1)
        # (batch_size, rulenum)
        score = score / rule_mask
        """
            计算score(一个商品对最终的分数)
        """
        # # (batch_size, )
        pooledscore = score.max(1)[0]
        """
            对property importance矩阵的每个元素做二分类: 对角线label为1, 非对角线元素为0
            使得每条rule尽量只关注一个property
        """
        # (rulenum, rulenum)
        diag = torch.eye(self.rulenum).cuda()

        # (rulenum, rulenum)
        reg_logits =  property_importance * diag + (1 - property_importance) * (1 - diag)

        # (rulenum, rulenum)
        reg_logits = - torch.log(reg_logits + 1e-8)

        #增加正例的权重
        # (rulenum, rulenum)
        reg_logits = reg_logits * ((1 - diag) + diag * self.rulenum)
        # (1)
        reg_logits = reg_logits.mean()

        return pooledscore, reg_logits