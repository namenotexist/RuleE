import torch
from torch.utils.data import DataLoader
import numpy as np
from time import time

def train(config, model, traindata, testdata):
    batch_size = config.batch_size
    epochnum = config.epochnum
    lr = config.lr
    eps = config.eps
    pos_weight = config.pos_weight
    neg_weight = config.neg_weight

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)
    # 定义loss

    NLLLossFunc = torch.nn.NLLLoss(weight=(torch.FloatTensor([neg_weight, pos_weight])).cuda())
    # 定义优化器

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    start_time = time()
    print("========================== start training =========================")
    for epoch in range(epochnum):
        avg_loss = 0
        train_avg_acc = 0
        epoch_start_time = time()

        train_tp = 0.0
        train_tn = 0.0
        train_fp = 0.0
        train_fn = 0.0

        test_tp = 0.0
        test_tn = 0.0
        test_fp = 0.0
        test_fn = 0.0

        rule_avg_properties = 0.0

        for i, (property, value, samemask, padmask, y) in enumerate(trainloader):

            # 模型前向传播
            model.train()
            score, reg_logits\
                = model(property.cuda(), value.cuda(),samemask.cuda(), padmask.cuda())

            # 计算准确率
            y = y.cuda()
            acc = ((score < 0.5) == (y == 0)).sum().item()
            train_avg_acc += acc

            # 计算TP, FP, TN, FN
            train_tp += ((score > 0.5) * (y == 1)).sum().item()
            train_fp += ((score > 0.5) * (y == 0)).sum().item()
            train_tn += ((score < 0.5) * (y == 0)).sum().item()
            train_fn += ((score < 0.5) * (y == 1)).sum().item()

            # 计算loss
            score = torch.stack((1 - score + eps, score + eps), dim=-1)
            score = torch.log(score)
            loss = NLLLossFunc(score, y) + 0.1 * reg_logits
            avg_loss += loss.item()
            # 梯度回传
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
        # 分别计算一个epoch的loss, 准确率以及每条规则关注的property个数
        avg_loss = avg_loss / len(trainloader)
        train_avg_acc = train_avg_acc / len(traindata)
        rule_avg_properties = rule_avg_properties / len(trainloader)

        # 在测试集上测试模型性能
        test_avg_acc = 0
        model.eval()
        with torch.no_grad():
            for (property, value, samemask, padmask, y) in testloader:
                score, _,= model(property.cuda(), value.cuda(), samemask.cuda(), padmask.cuda())

                # 计算准确率
                y = y.cuda()
                test_avg_acc += ((score < 0.5) == (y == 0)).sum().item()

                # 计算TP, FP, TN, FN
                test_tp += ((score > 0.5) * (y == 1)).sum().item()
                test_fp += ((score > 0.5) * (y == 0)).sum().item()
                test_tn += ((score < 0.5) * (y == 0)).sum().item()
                test_fn += ((score < 0.5) * (y == 1)).sum().item()
            test_avg_acc = test_avg_acc / len(testdata)
        # 输出训练结果
        print("==============================epoch: {}==========================================".format(epoch))
        print("==> elpased_time: {:.2f}s, loss: {:.2f},\n "
              "train_avg_acc: {:.2f},train_precision: {:.2f}, train_recall: {:.2f} \n "
              "test_avg_acc: {:.2f},test_precision: {:.2f}, test_recall: {:.2f},\n"
              " rule_avg_properties: {:.2f}".
              format(time() - epoch_start_time, avg_loss,
                     train_avg_acc, train_tp / (train_tp + train_fp + 1),train_tp / (train_tp + train_fn + 1),
                     test_avg_acc, test_tp / (test_tp + test_fp + 1), test_tp / (test_tp + test_fn + 1),
                     rule_avg_properties))
        print("==============================epoch: {}==========================================".format(epoch))
    # 保存模型
    torch.save(model.state_dict(), "./parameters/model.pth")
    params = {}
    for k, v in model.named_parameters():
        params[k] = v.detach().cpu().numpy()
        np.save("./parameters/{}.npy".format(k), params[k])
    print("total train time: {:.2f}s".format(time() - start_time))


