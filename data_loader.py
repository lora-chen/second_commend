import numpy as np
from datetime import datetime
import pandas as pd


# train_file = "./prepro_train_50.txt"     step = 1
# valid_file = "./prepro_valid_50.txt"     step = 2
# test_file = "./prepro_test_50.txt"       step = 3

def treat_prepro(train, step):
    train_f = open(train, 'r')
    # Need to change depending on threshold
    if step==1:
        lines = train_f.readlines()#[:86445] #659 #[:309931]

        # print("This is train date ")
        # print(len(lines))            2203975
        # 不是很明白他的备注是什么意思
    elif step==2:
        lines = train_f.readlines()#[:13505]#[:309931]
        # print("This is valid date ")
        # print(len(lines))            319853
    elif step==3:
        lines = train_f.readlines()#[:30622]#[:309931]
        # print("This is test date ")
        # print(len(lines))              708026

    train_user = []
    train_td = []
    train_ld = []
    train_loc = []
    train_dst = []

    user = 1
    user_td = []
    user_ld = []
    user_loc = []
    user_dst = []





    # 枚举的作用
    # >> > seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    # >> > list(enumerate(seasons))
    # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

    for i, line in enumerate(lines):
     # if step == 1:
        tokens = line.strip().split('\t')
        # 原始的lines是
        # ['1\n', '82.25\t0.331054\t133\t132\n', '1297.62\t0.109035\t132\t131\n',
        #  '382.5,321.375,32.5625\t0.0926226,0.0924566,0.00229809\t131,130,129\t128\n', '3783.62\t9.29145\t128\t6\n',
        #  '1942.94\t0.00119504\t6\t1\n', '3338.06\t25.7679\t1\t127\n', '2272.69\t25.7676\t127\t35']
        #变成了
        # ['1']
        # ['82.25', '0.331054', '133', '132']
        # ['1297.62', '0.109035', '132', '131']
        # ['382.5,321.375,32.5625', '0.0926226,0.0924566,0.00229809', '131,130,129', '128']
        # ['3783.62', '9.29145', '128', '6']
        # ['1942.94', '0.00119504', '6', '1']
        # ['3338.06', '25.7679', '1', '127']
        # # ['2272.69', '25.7676', '127', '35']

        # \t 横向制表符，一个TAB
        #  比如说 82.25	0.331054	133	132 变成 82.25\t	0.331054\t	133\t	132\t
        # 这里对每一行，先按照\t分割，然后去掉了空白字符（包括'\n', '\r',  '\t',  ' ')

        if len(tokens) < 3:
            #数据集的结构是一个ID代表人，下面是若干个打卡的数据，打卡数据每一列都是4个数据来表示
            #所以 len（tokens）《3 意味着这一行是ID

            if user_td:
            # 当user_td不为空时执行下面

                # 把user
                train_user.append(user)   # user其实是user的ID
                train_td.append(user_td)  # user latitude 纬度    后面看起来应该是时间
                train_ld.append(user_ld)  # user longitude经度    这个应该是距离        后面看来是精度
                train_loc.append(user_loc)  # User的location,应该是打卡的地方的编号     后面看来是维度
                train_dst.append(user_dst)  # User的distance移动距离                   后面看来是location

            user = int(tokens[0])   #这个tokens[0]里面装的就是用户的识别ID
            user_td = []            # 相当于初始化用户的档案
            user_ld = []
            user_loc = []
            user_dst = []
            continue
            # 这个continue加的挺奇怪的，感觉不加也没有影响？？？？？？？？？

        #这一行开始就是读的token不是ID行，而是数据行
        #创建 array 把tokens的数据度进去，从左到右分别是
        #  latitude 纬度    user longitude经度  location 打卡位置  distance 物理距离
        # 因为可能有多个数据在一行，可能是再短时间内去了好几个，所以按照“，”分开
        td = np.array([float(t) for t in tokens[0].split(',')])
        ld = np.array([float(t) for t in tokens[1].split(',')])
        loc = np.array([int(t) for t in tokens[2].split(',')])
        dst = int(tokens[3])


        user_td.append(td)
        user_ld.append(ld)
        user_loc.append(loc)
        user_dst.append(dst)
        #将读入的关于一个ID的所有数据读入到这个ID的档案里

    #将所有user的档案到train_user和其他的参数里
    if user_td: 
        train_user.append(user)
        train_td.append(user_td)
        train_ld.append(user_ld)
        train_loc.append(user_loc)
        train_dst.append(user_dst)

    #最后返回的是三个数据集 train valid，test文件所有user的所有信息
    #总共要返回三次
    return train_user, train_td, train_ld, train_loc, train_dst

def load_data(train):
    user2id = {}
    poi2id = {}

    train_user = []
    train_time = []
    train_lati = []
    train_longi = []
    train_loc = []
    valid_user = []
    valid_time = []
    valid_lati = []
    valid_longi = []
    valid_loc = []
    test_user = []
    test_time = []
    test_lati = []
    test_longi = []
    test_loc = []

    train_f = open(train, 'r')
    lines = train_f.readlines()

    user_time = []
    user_lati = []
    user_longi = []
    user_loc = []
    visit_thr = 30

    prev_user = int(lines[0].split('\t')[0])
    visit_cnt = 0
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user==prev_user:
            visit_cnt += 1
        else:
            if visit_cnt >= visit_thr:
                user2id[prev_user] = len(user2id)
            prev_user = user
            visit_cnt = 1

    train_f = open(train, 'r')
    lines = train_f.readlines()

    prev_user = int(lines[0].split('\t')[0])
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user2id.get(user) is None:
            continue
        user = user2id.get(user)

        time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")\
                -datetime(2009,1,1)).total_seconds()/60  # minutes
        lati = float(tokens[2])
        longi = float(tokens[3])
        location = int(tokens[4])
        if poi2id.get(location) is None:
            poi2id[location] = len(poi2id)
        location = poi2id.get(location)

        if user == prev_user:
            user_time.insert(0, time)
            user_lati.insert(0, lati)
            user_longi.insert(0, longi)
            user_loc.insert(0, location)
        else:
            train_thr = int(len(user_time) * 0.7)
            valid_thr = int(len(user_time) * 0.8)
            train_user.append(user)
            train_time.append(user_time[:train_thr])
            train_lati.append(user_lati[:train_thr])
            train_longi.append(user_longi[:train_thr])
            train_loc.append(user_loc[:train_thr])
            valid_user.append(user)
            valid_time.append(user_time[train_thr:valid_thr])
            valid_lati.append(user_lati[train_thr:valid_thr])
            valid_longi.append(user_longi[train_thr:valid_thr])
            valid_loc.append(user_loc[train_thr:valid_thr])
            test_user.append(user)
            test_time.append(user_time[valid_thr:])
            test_lati.append(user_lati[valid_thr:])
            test_longi.append(user_longi[valid_thr:])
            test_loc.append(user_loc[valid_thr:])

            prev_user = user
            user_time = [time]
            user_lati = [lati]
            user_longi = [longi]
            user_loc = [location]

    if user2id.get(user) is not None:
        train_thr = int(len(user_time) * 0.7)
        valid_thr = int(len(user_time) * 0.8)
        train_user.append(user)
        train_time.append(user_time[:train_thr])
        train_lati.append(user_lati[:train_thr])
        train_longi.append(user_longi[:train_thr])
        train_loc.append(user_loc[:train_thr])
        valid_user.append(user)
        valid_time.append(user_time[train_thr:valid_thr])
        valid_lati.append(user_lati[train_thr:valid_thr])
        valid_longi.append(user_longi[train_thr:valid_thr])
        valid_loc.append(user_loc[train_thr:valid_thr])
        test_user.append(user)
        test_time.append(user_time[valid_thr:])
        test_lati.append(user_lati[valid_thr:])
        test_loc.append(user_loc[valid_thr:])

    return len(user2id), poi2id, train_user, train_time, train_lati, train_longi, train_loc, valid_user, valid_time, valid_lati, valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc

def inner_iter(data, batch_size):
    data_size = len(data)
    num_batches = int(len(data)/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
