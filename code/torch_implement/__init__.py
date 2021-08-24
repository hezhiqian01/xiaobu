import yaml
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


with open("./torch_implement/config.yml") as f:
    config = yaml.load(f)


def find_best_threshold(all_predictions, all_labels):
    """寻找最佳的分类边界, 在0到1之间"""
    # 展平所有的预测结果和对应的标记
    # all_predictions为0到1之间的实数
    all_predictions = np.ravel(all_predictions)
    all_labels = np.ravel(all_labels)
    # 从0到1以0.01为间隔定义99个备选阈值, 分别是从0.01-0.99之间
    thresholds = [i / 100 for i in range(100)]
    all_f1s = []
    for threshold in thresholds:
        # 计算当前阈值的f1 score
        preds = (all_predictions >= threshold).astype("int")
        f1 = f1_score(y_true=all_labels, y_pred=preds)
        all_f1s.append(f1)
    # 找出可以使f1 socre最大的阈值
    best_threshold = thresholds[int(np.argmax(np.array(all_f1s)))]
    print("best threshold is {}".format(str(best_threshold)))
    print(all_f1s)
    return best_threshold


def lr_warm_up_and_decay(init_lr, curr_steps, warm_up_steps):
    if curr_steps < warm_up_steps:
        warm_up_percent = curr_steps/warm_up_steps
        lr = init_lr * warm_up_percent
    else:
        lr = init_lr/(1 + config['pretrain']['decay_rate'] * curr_steps)
    if curr_steps % 100 == 0:
        print("current steps:{}, used lr:{}".format(curr_steps, lr))
    return lr


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

