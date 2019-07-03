#! /usr/bin/env python

import os
import datetime
import math
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "./prepro_train_50.txt"
valid_file = "./prepro_valid_50.txt"
test_file = "./prepro_test_50.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # window width (6h)
up_time = 560632.0  # min  #最大时间
lw_time = 0.
up_dist = 457.335   # km   # 最大距离
lw_dist = 0.
reg_lambda = 0.1

# Training Parameters
batch_size = 2
num_epochs = 30
learning_rate = 0.001
momentum = 0.9            # 用到了moment算法
evaluate_every = 1
h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)

# torch.randn(dim, 1) 返回一个张量，包含了从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数。 Torch size 为 31*1
# 在训练时如果想要固定网络的底层，那么可以令这部分网络对应子图的参数requires_grad为False。这样，在反向过程中就不会计算这些参数对应的梯度
# 要问下那个计算图是什么用？
# b = a + z #a ,z 中，有一个 requires_grad 的标记为True，那么输出的变量的 requires_grad为True
# b.requires_grad  True

#如果使用data.type(torch.FloatTensor)则强制转换为torch.FloatTensor类型张量。   print ( h_0.shape)   torch.Size([13, 1])
#相当于最后h_0是一个 13*1  的 cuda.FloatTensor                                print(h_0.type())    torch.cuda.FloatTensor



try:
    xrange
except NameError:
    xrange = range
# name = object
# 目标是indentifier(name)
# 将object关联到当前namespace，
# 如果当前namespace没有这个name就创建
# 创建一个range class的变量 xrange

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")

# return train_user, train_td, train_ld, train_loc, train_dst
# 一一对应的赋值

train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = data_loader.treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = data_loader.treat_prepro(test_file, step=3)

# 创建几个变量，然后使用了 data_loader.treat_prepro.
#最后返回的是三个数据集，每个数据集包含了train_user, train_td, train_ld, train_loc, train_dst这几个元素


user_cnt = 32899 #50 #107092#0
loc_cnt = 1115406 #50 #1280969#0
#user_cnt = 42242 #30
#loc_cnt = 1164559 #30

#这两个参数直接给的，应该是数据库里总共有32899个用户，然后已知的标号有1115406个
print("User/Location: {:d}/{:d}".format(user_cnt, loc_cnt))
print("==================================================================================")


class STRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super(STRNNCell, self).__init__()

        # # Model Hyperparameters
        # dim = 13  # dimensionality
        # ww = 360  # window width (6h)
        # up_time = 560632.0  # min  #最大时间
        # lw_time = 0.
        # up_dist = 457.335  # km   # 最大距离
        # lw_dist = 0.
        # reg_lambda = 0.1

        self.hidden_size = hidden_size
        # 后面初始化的时候会用这个语句 strnn_model = STRNNCell(dim).cuda() 来用dim赋值给hidden_size，这里
        # print(hidden_size) 13

        # 可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # 当塔使用parameter时会成为model的一个属性，然后被加到parameter的一个list里。就可以通过“parameters”来进入
        # 可以通过list(net.parameters())来访问

        # 应该是先创建一个13*13的张量，然后把他们转成STRNNCell这个model的weight，训练的时候就是训练这5个weight

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # C

        # Temporal 有关
        self.weight_th_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T
        self.weight_th_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # T

        # Spatial有关
        self.weight_sh_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S
        self.weight_sh_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # S

        self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        self.permanet_weight = nn.Embedding(user_cnt, hidden_size)
        # 原始数据已经很用数字ID表示，这里是将
        # user_cnt = 32899  嵌入到13维
        # loc_cnt = 1115406 嵌入到13维
        # 输入会是一个N *32899的张量，然后输出一个N* 13的张量


        self.sigmoid = nn.Sigmoid()
        # 激活函数是sigmoid

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        # stdv = 1除以 13的平方根
        # 标准差 Standard Deviation

        # 参数初始化，不是很理解为什么要这么做？？？？？
        # 把weight变成一个STDV分布之间的一个随机数
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    # parameters including "weight_ih" "weight_th_upper" "weight_th_lower" "weight_sh_upper and lower"

    # rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)
    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):
        loc_len = len(loc)
        # 去过地点的个数

        # Ttd和Sld是一个比值，有一定的物理意义
        # 是由 weight_th_upper这个权值和第i个地点的td_upper相乘，再加上weight_th_upper和第i个地点的下限td_lower。
        # 然后用这个和除以 刚才run函数里面的算出来的两个权值和第i个地点的上限和下限的和，算出来的一个东西
        Ttd = [((self.weight_th_upper*td_upper[i] + self.weight_th_lower*td_lower[i])\
                /(td_upper[i]+td_lower[i])) for i in xrange(loc_len)]
        Sld = [((self.weight_sh_upper*ld_upper[i] + self.weight_sh_lower*ld_lower[i])\
                /(ld_upper[i]+ld_lower[i])) for i in xrange(loc_len)]
        #
        # 初始化RNN时候随机生成的
        # self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # C
        # self.weight_th_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # T
        # self.weight_th_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # T
        # self.weight_sh_upper = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # S
        # self.weight_sh_lower = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # S

        # Run 里面计算的
        # td_upper = Variable(torch.from_numpy(np.asarray(up_time - td[idx]))).type(ftype)
        # td_lower = Variable(torch.from_numpy(np.asarray(td[idx] - lw_time))).type(ftype)
        # ld_upper = Variable(torch.from_numpy(np.asarray(up_dist - ld[idx]))).type(ftype)
        # ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx] - lw_dist))).type(ftype)
        # location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(ltype)



        loc = self.location_weight(loc).view(-1,self.hidden_size,1)
        # self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        # self.permanet_weight = nn.Embedding(user_cnt, hidden_size)

        # 把loc这个 n维的矩阵喂给地点嵌入层,出来的是一个 n*13的张量
        # 然后把这个张量重组为一个 （X，13，1）维的张量，X为未知值，实际为n
        # 最后  loc是一个（n，13，1）的张量

        # -1 means no sure about the size of one row
        # View() method can regroup the tensor into different size , but does not change content.
        # e.g. a = torch.arange(1, 17)  # a's shape is (16,)
        # a.view(4, 4) # output below
        #   1   2   3   4
        #   5   6   7   8
        #   9  10  11  12
        #  13  14  15  16
        # [torch.FloatTensor of size 4x4]


        # for i in xrange(loc_len) :
        #      a= torch.mm(Ttd[i], loc[i])
        #      a= torch.mm(Sld[i],a)
        #      print("bbb")
        #      a=torch.cat([a.view(1,self.hidden_size,1)],dim=0)
        #

        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i]))\
                                      .view(1, self.hidden_size, 1) for i in xrange(loc_len)], dim=0), dim=0)
        # torch.mm 是两个矩阵的乘积。
        # torch.mm(a, b) 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3) 的矩阵

        # print(Ttd[i].size(), loc[i].size())  torch.Size([13, 13]) torch.Size([13, 1])
        # print(Sld[i].size()) torch.Size([13, 13])

        # 对于所有location，计算每个里面先是Ttd【i】和loc【i】相乘，得到一个13*1的张量。 然这个乘积和Sld【i】相乘。得到的乘积是13*1的张量
        # 然后这个13*1的张量变成（1，13，1）的张量对于一个地点来说
        # 然后torch.cat把loc_len()这个多个张量以行为单位拼在一起
        # a = torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i])).view(1, self.hidden_size, 1) for i in xrange(loc_len)], dim=0)
        # print( a.size())   torch.Size([1, 13, 1])
        # 拼完之后还是（1，13，1）张量   ？？？？？？？？？？？？？？？？

        # print(loc_vec.size())            torch.Size([13, 1])
        # 然后torch.sum(x,dim=0) 对于这个张量，按列求和
        # 最后输出的是一个13*1的张量，这个应该是代表所有地点的一个东西

        usr_vec = torch.mm(self.weight_ih, hx)
        # self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # C
        #  13*13 的torch

        # hx 来源于 # rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output) 里的rnn_output
        # run里面说 rnn_output = h_0
        # h_0是一个 13*1  的 cuda.FloatTensor，初始值是随机的
        # 乘出来的是13*1的张量

        # 两个13*1的张量相乘，然后用激活函数做输出
        hx = loc_vec + usr_vec # hidden_hx.size()size x 1
        # print(self.sigmoid(hx).size())     torch.Size([13, 1])
        return self.sigmoid(hx)


    # J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location, destination,
    #                      rnn_output)  # , neg_lati, neg_longi, neg_loc, step)
    def loss(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):

        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        # h_tq 是一个13*1

        p_u = self.permanet_weight(user)
        # print(p_u.size()) torch.Size([1, 13])
        # self.permanet_weight = nn.Embedding(user_cnt, hidden_size)
        # 用户ID嵌入层

        q_v = self.location_weight(dst)
        # print(q_v.size())torch.Size([1, 13])
        output = torch.mm(q_v, (h_tq + torch.t(p_u)))
        # print(output.size())  torch.Size([1, 1])

        # print (torch.log(1 + torch.exp(torch.neg(output))).size()) torch.size[1,1]
        return torch.log(1+torch.exp(torch.neg(output)))

    # torch.neg(input, out=None) → Tensor
    #     # 返回一个新张量，包含输入input张量按元素取负。 即，out =−1∗input
    # torch.exp 计算每个元素的指数
    # torch.log(input)  # y_i=log_e(x_i)


    # 如果是测试集合的话
    # return strnn_model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[-1], rnn_output), dst[-1]

    def validation(self, user, td_upper, td_lower, ld_upper, ld_lower, loc, dst, hx):
        # error exist in distance (ld_upper, ld_lower)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        # 逻辑不是很清楚，只知道forward里是怎么写的，出来的是一个经过激活函数的值

        p_u = self.permanet_weight(user)
        # print(p_u.size()) torch.Size([1, 13])
        # self.permanet_weight = nn.Embedding(user_cnt, hidden_size)

        # p_u的转置和算出来的状态  torch.Size(13*1)
        user_vector = h_tq + torch.t(p_u)

        # user_vector和
        #print(self.location_weight.weight.size()) #torch.Size([1115406, 13])
        #        1115406是记录地点的总数
        ret = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        # print (torch.mm(self.location_weight.weight, user_vector).size())    torch.Size([1115406, 1])
        # self.location_weight = nn.Embedding(loc_cnt, hidden_size)
        return np.argsort(np.squeeze(-1*ret))
    #  ret 所有元素乘 -1 然后删除所有单维条目，从（1115406*1 ） 变成了 [1115406]，然后返回数组从小到大的索引
    #  相当于从大到小排列  ret的元素，然后返回索引

    # numpy.squeeze(a, axis=None)
    # 可选。若axis为空，则删除所有单维度的条目

    # 从中可以看出argsort函数返回的是数组值从小到大的索引值
    # >> > x = np.array([3, 1, 2])
    # >> > np.argsort(x)
    # array([1, 2, 0])
#
# print (np.argsort(np.squeeze(-1*ret)))
# validation:   1%|          | 200/32660 [01:25<3:47:24,  2.38it/s][845061 163814 818899 ... 434373 644797 782981]
# validation:   1%|          | 201/32660 [01:26<3:47:14,  2.38it/s][845061 807503 818899 ... 434373 334915 431000]
# validation:   1%|          | 202/32660 [01:26<3:49:28,  2.36it/s][562384 975191 845061 ... 334915 782981  47535]
# validation:   1%|          | 203/32660 [01:27<3:39:07,  2.47it/s][845061 807503 163814 ... 431000 434373 782981]
# validation:   1%|          | 204/32660 [01:27<3:40:05,  2.46it/s][845061 818899 807503 ... 521520 651036 782981]


###############################################################################################
def parameters():
    params = []
    for model in [strnn_model]:
        params += list(model.parameters())

    return params
# 将parameter加入到参数组里

# valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))，step=2
def print_score(batches, step):
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    iter_cnt = 0

    # 一样的进度条
    for batch in tqdm.tqdm(batches, desc="validation"):
        batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
        # 解压的一条batch由一个用户ID的所有信息

        #区别轻度用户
        if len(batch_loc) < 3:
            continue

         #循环计数器
        iter_cnt += 1

        batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)
        # return strnn_model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[-1], rnn_output), dst[-1]
        # validation函数的输出为  return np.argsort(np.squeeze(-1 * ret))

        # ？？？？？？？？？？？？
        # 可能是先出去多少个推荐地点，然后看预测的target在不在推荐地点的前几个里，然后用这个正确个数取除跑的循坏的个数
        # ？？？？？？？？？？？？

        recall1 += target in batch_o[:1]
        recall5 += target in batch_o[:5]
        recall10 += target in batch_o[:10]
        recall100 += target in batch_o[:100]
        recall1000 += target in batch_o[:1000]
        recall10000 += target in batch_o[:10000]

    print("recall@1: ", recall1/iter_cnt)
    print("recall@5: ", recall5/iter_cnt)
    print("recall@10: ", recall10/iter_cnt)
    print("recall@100: ", recall100/iter_cnt)
    print("recall@1000: ", recall1000/iter_cnt)
    print("recall@10000: ", recall10000/iter_cnt)

###############################################################################################

# total_loss += run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
def run(user, td, ld, loc, dst, step):

    optimizer.zero_grad()
    #开始训练前要清0，上次没有听清楚，还是再问一下 ？？？？？？？？？？？？？？？？？？

    seqlen = len(td)   #相当于知道这个人去过多少的地方的个数。 因为每个location都有一个精度和维度对应
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)
    # Numpy桥，将numpy.ndarray
    # 转换为pytorch的Tensor。 返回的张量tensor和numpy的ndarray共享同一内存空间。修改一个会导致另外一个也被修改。返回的张量不能改变大小。

    # 先把 user  这个list转成array，然后转成一个cuda.longTensor类型的张量，然后建立前进树

    #neg_loc = Variable(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long()).type(ltype)
    #(neg_lati, neg_longi) = poi2pos.get(neg_loc.data.cpu().numpy()[0])
    rnn_output = h_0
    # h_0是一个 13*1  的 cuda.FloatTensor，初始值是随机的

    for idx in xrange(seqlen-1):
        #应该是用遍历所有的地点，但是为什么要减1不清楚？？？？？？？？？？？？？

        # up_time = 560632.0  # min  #最大时间
        # lw_time = 0.
        # up_dist = 457.335  # km   # 最大距离
        # lw_dist = 0.
        td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[idx]))).type(ftype)
        td_lower = Variable(torch.from_numpy(np.asarray(td[idx]-lw_time))).type(ftype)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[idx]))).type(ftype)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx]-lw_dist))).type(ftype)
        location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(ltype)
        # 这个的实际物理意义是什么不太清楚，就知道会有5个值求出来

        # 循环的把这个人去过的所有地方计算出来的东西
        # 把这5个隐state 喂给 def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)#, neg_lati, neg_longi, neg_loc, step)

        # print(self.sigmoid(hx).size())     torch.Size([13, 1]) 返回的还是13*1的torch
        #return self.sigmoid(hx)

    # 计算这个人去的最新的地方，算个什么东西出来
    td_upper = Variable(torch.from_numpy(np.asarray(up_time-td[-1]))).type(ftype)
    td_lower = Variable(torch.from_numpy(np.asarray(td[-1]-lw_time))).type(ftype)
    ld_upper = Variable(torch.from_numpy(np.asarray(up_dist-ld[-1]))).type(ftype)
    ld_lower = Variable(torch.from_numpy(np.asarray(ld[-1]-lw_dist))).type(ftype)
    location = Variable(torch.from_numpy(np.asarray(loc[-1]))).type(ltype)

    # 如果是测试集或者验证集
    if step > 1:
        return strnn_model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[-1], rnn_output), dst[-1]

     # 如果是训练集

    destination = Variable(torch.from_numpy(np.asarray([dst[-1]]))).type(ltype)
    # print(destination.size()) torch.Size([1])
    # 把最新的destination保存

    J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location, destination, rnn_output)#, neg_lati, neg_longi, neg_loc, step)
    # J  为【1，1】 torch
    print(J.size())

    J.backward()    # 反向
    optimizer.step() # 更新所有信息

    return J.data.cpu().numpy()    # 如果跑的是训练集，就返回loss

###############################################################################################
strnn_model = STRNNCell(dim).cuda()
# 初始化model，注释在上面
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum, weight_decay=reg_lambda)
# params(iterable) – 待优化参数的iterable或者是定义了参数组的dict
# lr(float) – 学习率
# momentum(float, 可选) – 动量因子（默认：0）
# weight_decay(float, 可选) – 权重衰减（L2惩罚）（默认：0）
# 牵扯到范数，这个要之后再看。    现在知道这个是防止过拟合




# 30个epochs
for i in xrange(num_epochs):
    # Training
    total_loss = 0.

    # return train_user, train_td, train_ld, train_loc, train_dst
    #train_user, train_td, train_ld, train_loc, train_dst = data_loader.treat_prepro(train_file, step=1)

    train_batches = list(zip(train_user, train_td, train_ld, train_loc, train_dst))
    # print("len(train_user)",len(train_user))   1   包含了三个文件之中的一个文件里所有的user_ID信息）
    # print("len(train_td)",len(train_td))       1
    # print("len(train_loc)", len(train_loc))    1
    # print("len(train_dst)",len(train_dst))     1
    # print ("len(train_batches)",len(train_batches))  1
    # 最后train_batchs就是当前处理文件（e.g training file里所有的信息）

    # >> > a = [1, 2, 3]
    # >> > b = [4, 5, 6]
    # >> > zipped = zip(a, b)  # 打包为元组的列表
    # [(1, 4), (2, 5), (3, 6)]

    # tqdm 是创建进度条用的。比如说读了100个batch，那么这里就会显示正在处理“N/100”个包
    # desc 是在进度条前面加前缀
    for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
        #总共有32899个人，可是每个文件只会包含其中的一部分人。
        #如果人的ID连续不断的话,j = 0就是ID为1的user，j = 1就是ID为2的人，以此类推
        #与其对应的train_batches[j]就是这个ID的user的所有的数据

        batch_user, batch_td, batch_ld, batch_loc, batch_dst = train_batch
        # 对于一个一个的人的信息

        # 如果这个用户去的地方少于3个，就不算他了
        # 轻度用户鉴别
        if len(batch_loc) < 3:
            continue

        total_loss += run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
        # 如果训练集，就返回loss

        #if (j+1) % 2000 == 0:
        #    print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()
    # Evaluation
    # evaluate_every = 1
    if (i+1) % evaluate_every == 0:  # 每隔几个包执行下面的语句
        print("==================================================================================")
        # print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j, datetime.datetime.now()
        valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))
        # 和上面train_user一样，valid_batches 也是包含了验证集的所有用户的所有信息

        print_score(valid_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = list(zip(test_user, test_td, test_ld, test_loc, test_dst))
print_score(test_batches, step=3)
