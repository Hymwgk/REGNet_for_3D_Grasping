import os
import glob
import pickle
import torch
import math
import torch.utils.data
import torch.nn as nn
import numpy as np
import random

class ScoreDataset(torch.utils.data.Dataset):
    '''数据集类
    '''
    def __init__(self, all_points_num, path, tag, data_seed, data_width):
        self.all_points_num = all_points_num#一阵图像的点数
        self.base_path = path#文件夹地址
        self.tag = tag#自定义标签
        self.width = np.array(data_width, dtype=np.float32)

        np.random.seed(data_seed)
        if 'eval_data' in self.base_path:#如果是作者给定默认文件夹
            p_path = os.listdir(self.base_path)
            p_path.sort()
            p_path = np.array(p_path)
            #p_path = p_path[:3]

            index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)#随机选择80%的文件索引
            if self.tag != "train":
                ori = np.arange(len(p_path)) 
                index = np.array(list( set(list(ori)) - set(list(index)) ))

            self.data_name = p_path[index]
        else:#如果是自定义文件夹路径
            if self.tag == "test":
                base_dir_name = "training_data_test"
            else:
                base_dir_name = "training_data"

            self.base_path = os.path.join(self.base_path, base_dir_name) 
            p_path = os.listdir(self.base_path)
            p_path.sort()
            p_path = np.array(p_path)

            if self.tag == "test":
                self.data_name = p_path
            else:#训练模式
                index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)#随机抽选80%数据由于训练
                if self.tag == "validate":#验证模式，就反选剩下的20%数据
                    ori = np.arange(len(p_path)) 
                    index = np.array(list( set(list(ori)) - set(list(index)) ))
                
                self.data_name = p_path[index]

    def _noise_color(self, color, label):
        table_color_time = np.random.rand(3)
        obj_color_time = 1-np.random.rand(3) / 5
        for i in range(3):
            color[label==0, i] *= table_color_time[i]
            color[label!=0, i] *= obj_color_time[i]
        return color

    def __getitem__(self, index):
        '''按照index 把一个样本读出来
        '''
        data_path = os.path.join(self.base_path, self.data_name[index])#单个样本的地址
        data = np.load(data_path, allow_pickle=True)#读取样本数据
        view = data['view_cloud'].astype(np.float32)#点云三通道坐标数据
        view_color = data['view_cloud_color'].astype(np.float32)#点云三通道颜色数据
        view_score = data['view_cloud_score'].astype(np.float32)#点云每个点的0/1分数，构成一个分割mask
        view_label = data['view_cloud_label'].astype(np.float32)#这个lable没搞懂

        select_point_index = None
        if len(view) >= self.all_points_num:#如果一帧点云的点数大于指定点数，就对点云进行随机抽选剪切
            select_point_index = np.random.choice(len(view), self.all_points_num, replace=False)
        elif len(view) < self.all_points_num:#如果小于指定点数，就随机补点
            select_point_index = np.random.choice(len(view), self.all_points_num, replace=True)
        
        view, view_color, view_label, view_score = view[select_point_index], view_color[select_point_index], \
                                                view_label[select_point_index], view_score[select_point_index]

        view_color = self._noise_color(view_color, view_label)
        view_mean = np.mean(view, axis=0)
        view = np.c_[view, view_color]#在axis=1方向拼接

        view_score = np.tanh(view_score)#
        return view, view_score, view_label, data_path, self.width

    def __len__(self):
        return len(self.data_name)
        # return int(len(self.data_name) / 400)