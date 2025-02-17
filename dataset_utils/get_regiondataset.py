import os
import glob
import pickle
import torch
import math
import torch.utils.data
import torch.nn as nn
import numpy as np
import random
import open3d #0.7.0
from multi_model.utils.pn2_utils import function as _F

def get_grasp_allobj(pc, predict_score, params, data_paths, use_theta=True):
    '''
      randomly sample grasp center in positive points set (all obj), 
      and get grasps centered at these centers.
      按照分数，从原始点云中，通过PFS算法（感觉实际上没有用这个）抽取出k个抓取中心

      Input:
        pc             :[B,N,6]  6维度的原始点云
        predict_score  :[B,N]  每个点的预测分数
        params         :list [center_num(int), score_thre(float), group_num(int), r_time_group(float), group_num_more(int), \
                              r_time_group_more(float), width(float), height(float), depth(float)]
        data_paths     :list
      Output:
        center_pc           :[B, center_num, 6] 各抓取中心点的坐标&颜色
        center_pc_index     :[B, center_num] index of selected center in sampled points 各抓取中心点在原始点云中的索引
        pc_group            :[B, center_num, group_num, 6]  以各抓取中心点构造包围球，返回包围球内部所有点的坐标
        pc_group_index      :[B, center_num, group_num]  包围球内部所有点在原始点云中的索引
        pc_group_more       :[B, center_num, group_num_more, 6]  更多一点的抓取的
        pc_group_more_index :[B, center_num, group_num_more] 
        grasp_labels        :[B, center_num, 8] the labels of grasps (center[3], axis_y[3], grasp_angle[1], score[1])
    '''
    [center_num, score_thre, group_num, r_time_group, group_num_more, r_time_group_more, \
                                                            width, height, depth] = params
    #从原始点云中，按照分数和阈值，抓取中心的点云子集；此时的center_pc是通过FPS采样得来的
    center_pc, center_pc_index = _select_score_center(pc, predict_score, center_num, score_thre)
    #根据抓取中心，获取他的包围球的内部点，以及索引
    pc_group_index, pc_group = _get_group_pc(pc, center_pc, center_pc_index, group_num, width, height, depth, r_time_group)
    #另一个尺寸的包围球聚类
    pc_group_more_index, pc_group_more = _get_group_pc(pc, center_pc, center_pc_index, group_num_more, width, height, depth, r_time_group_more)

    grasp_labels = None
    if len(data_paths) > 0:
        #抽出了k1个点，查阅数据集，
        grasp_labels = _get_center_grasp(center_pc_index, center_pc, data_paths, depth, use_theta)
    return center_pc, center_pc_index, pc_group_index, pc_group, pc_group_more_index, pc_group_more, grasp_labels


def _get_center_grasp(center_pc_index, center_pc, data_paths, depth, use_theta=True):
    '''查阅样本数据集，为指定点分配groundtruth抓取以及对应的抓取分数
    GRN之前，通过FPS抽出k1个点，该函数读取这些点的索引和位置，
      Input:
        center_pc_index: [B, center_num] 输入抓取中心点的索引
        center_pc:   [B, center_num, 6] x,y,z,r,g,b  抓取中心点的坐标，
        data_paths:  list 当前batch的样本在数据集中的路径
        depth: 夹爪的深度
        use_theta:是否使用了
      Output:
        输出的这个是啥？
        grasp_trans: [B, center_num, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
        or
        grasp_trans: [B, center_num, 10] (axis_x[3], axis_y[3], axis_z[3], center[3], score[1])
    '''
    #创建
    grasp_index = torch.full(center_pc_index.shape, -1)# [B, center_num] 填充-1
    grasp_label = torch.full((center_pc_index.shape[0], center_pc_index.shape[1], 3, 4), -1.0)#[B, center_num,3,4] 填充-1.0 
    grasp_trans = torch.full((center_pc_index.shape[0], center_pc_index.shape[1], 13), -1.0)#[B, center_num,13] 填充-1.0 
    grasp_score_label           = torch.full((center_pc_index.shape[0], center_pc_index.shape[1]), -1.0)#[B, center_num] 填充-1.0 
    grasp_antipodal_score_label = torch.full((center_pc_index.shape[0], center_pc_index.shape[1]), -1.0)#[B, center_num] 填充-1.0 
    grasp_center_score_label    = torch.full((center_pc_index.shape[0], center_pc_index.shape[1]), -1.0)#[B, center_num] 填充-1.0 
    
    if center_pc_index.is_cuda:
        grasp_index, grasp_label, grasp_score_label = grasp_index.cuda(), grasp_label.cuda(), grasp_score_label.cuda()
    for i in range(len(data_paths)):
        data = np.load(data_paths[i], allow_pickle=True)
        if 'frame' in data.keys():
            grasp = torch.Tensor(data['frame'])
            grasp_score =  torch.Tensor(data['antipodal_score'])#读取antipodal分数作为抓取分数
            grasp_antipodal_score = None
            if center_pc_index.is_cuda:
                grasp, grasp_score = grasp.cuda(), grasp_score.cuda()

        else:
            #读取当前帧样本点云中的所有候选抓取以及其他数据
            grasp  = torch.Tensor(data['select_frame'])#当前样本点云中的所有候选抓取坐标系的4*4标准变换矩阵（每帧大约有1000+）
            grasp_score  = torch.Tensor(data['select_score']) / 3 if type(data['select_score']) is np.ndarray else data['select_score'] / 3
            grasp_antipodal_score = torch.Tensor(data['select_antipodal_score']) if type(data['select_antipodal_score']) is np.ndarray else data['select_antipodal_score']
            grasp_center_score    = torch.Tensor(data['select_center_score']) if type(data['select_center_score']) is np.ndarray else data['select_center_score']
            grasp_vertical_score  = torch.Tensor(data['select_vertical_score']) if type(data['select_vertical_score']) is np.ndarray else data['select_vertical_score']
            grasp_class_label     = torch.Tensor(data['select_frame_label']) if type(data['select_frame_label']) is np.ndarray else data['select_frame_label']
            
            if center_pc_index.is_cuda:
                grasp, grasp_score = grasp.cuda(), grasp_score.cuda()
                grasp_antipodal_score, grasp_center_score, grasp_vertical_score, grasp_label = \
                    grasp_antipodal_score.cuda(), grasp_center_score.cuda(), grasp_vertical_score.cuda(), grasp_label.cuda()
        
        #grasp = grasp.float()
        #grasp_select_inedx = (grasp_score > 0.4)
        #grasp_score = grasp_score[grasp_select_inedx]
        #grasp = grasp[grasp_select_inedx]

        grasp_center = grasp[:,:3,3]#当前点云所有典范抓取坐标系的bottom_center的xyz坐标[B*center_num,3]
        grasp_x, grasp_y, grasp_z = grasp[:,:3,0], grasp[:,:3,1], grasp[:,:3,2]#当前点云所有候选抓取的三个坐标轴
        grasp_center = (grasp_center + grasp_x * depth).float()#当前点云所有典范抓取坐标系的grasp_center的xyz坐标
        in_grasp_center = (grasp_center - grasp_x * depth).float()#?

        #找到样本点云中与选出的k1个点距离最近的那些点
        distance = _compute_distance(center_pc[i], in_grasp_center)#先计算每个点与数据集的距离
        distance_min = torch.min(distance, dim=1)#
        distance_min_values, distance_min_index = distance_min[0], distance_min[1]#[center_num,]
        #distance_min_index[distance_min_values > 0.015] = -1

        grasp_index[i] = distance_min_index#k1个抓取点中每个抓取点与样本集中最近的抓取的索引，一共也是k1个
        grasp_label[i] = grasp[distance_min_index,:3,:4]#k1个最近抓取的三个抓取轴以及中心点坐标
        grasp_score_label[i]           = grasp_score[distance_min_index]#k1个最近groundtruth抓取的分数
        if grasp_antipodal_score is not None:
            grasp_antipodal_score_label[i] = grasp_antipodal_score[distance_min_index]#也记录下k1个最近ground truth的antipodal分数
            grasp_center_score_label[i]    = grasp_center_score[distance_min_index]
        
        no_grasp_mask = (distance_min_values > 0.005)#如果在样本中与某点的最近抓取距离还大于5mm，
        grasp_index[i][no_grasp_mask] = -1#将该点对应的最近抓取的索引记为-1
        grasp_label[i][no_grasp_mask] = -1#最近抓取的三个抓取轴以及中心点坐标也设置为-1
        grasp_score_label[i][no_grasp_mask]           = -1#分数也都是-1
        if grasp_antipodal_score is not None:
            grasp_antipodal_score_label[i][no_grasp_mask] = -1
            grasp_center_score_label[i][no_grasp_mask]    = -1
    
    if use_theta:#如果使用了theta值
        grasp_trans = _transform_grasp(grasp_label, grasp_score_label, grasp_antipodal_score_label, grasp_center_score_label)
        # test whether the grasp transformed right
        ##grasp_label, grasp_score_label = _inv_transform_grasp(grasp_trans)
    else:
        grasp_label = grasp_label.view(-1,3,4)
        inv_mask = (grasp_label[:,0,1] < 0)
        grasp_label[inv_mask, :, 1:2] = -grasp_label[inv_mask, :, 1:2]
        grasp_trans[:,:,:12] = grasp_label.transpose(2,1).contiguous().view(center_pc_index.shape[0], center_pc_index.shape[1], 12)
        grasp_trans[:,:,12:13] = grasp_score_label.view(center_pc_index.shape[0], center_pc_index.shape[1], 1)
    
    if grasp_label.is_cuda:
        grasp_trans = grasp_trans.cuda()
    return grasp_trans

def _transform_grasp(grasp_ori, grasp_score_ori, antipodal_score_ori, center_score_ori):
    '''计算拼接groundtruth抓取向量并返回
      Input:
        grasp_ori: [B, center_num, 3, 4] 抓取的位置姿态
                   [[x1, y1, z1, c1],
                    [x2, y2, z2, c2],
                    [x3, y3, z3, c3]]
        grasp_score_ori: [B, center_num]#分数
      Output:
        grasp_trans:[B, center_num, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
    '''
    B, CN = grasp_score_ori.shape
    #如果数据集中没有 antipodal_score
    if torch.sum(antipodal_score_ori == -1) == antipodal_score_ori.shape[0] * antipodal_score_ori.shape[1]:
        grasp_trans = torch.full((B, CN, 8), -1.0)#就只记录基于score= cos_alpha1*cos_alpha2的分数
    else:
        grasp_trans = torch.full((B, CN, 10), -1.0)#否则都记录下来

    #ground truth抓取三坐标轴
    axis_x = grasp_ori[:,:,:3,0].view(B*CN, 3)
    axis_y = grasp_ori[:,:,:3,1].view(B*CN, 3)
    axis_z = grasp_ori[:,:,:3,2].view(B*CN, 3)


    no_grasp_mask = ((axis_x[:,0]==-1) & (axis_x[:,1]==-1) & (axis_x[:,2]==-1))
    #计算ground truth的theta值
    grasp_angle = torch.atan2(axis_x[:,2], axis_z[:,2])  ## torch.atan(torch.div(axis_x[:,2], axis_z[:,2])) is not OK!!!


    grasp_angle[axis_y[:,0] < 0] = np.pi-grasp_angle[axis_y[:,0] < 0]
    axis_y[axis_y[:,0] < 0] = -axis_y[axis_y[:,0] < 0]

    grasp_angle[grasp_angle >= 2*np.pi] = grasp_angle[grasp_angle >= 2*np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -2*np.pi] = grasp_angle[grasp_angle <= -2*np.pi] + 2*np.pi
    grasp_angle[grasp_angle > np.pi] = grasp_angle[grasp_angle > np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -np.pi] = grasp_angle[grasp_angle <= -np.pi] + 2*np.pi
    '''
    cond1 = (grasp_angle > 0.5*np.pi) & (grasp_angle <= 1.5*np.pi)
    grasp_angle[cond1] = -(grasp_angle[cond1] - np.pi)
    axis_z[cond1] = -axis_z[cond1]
    axis_x[cond1] = -axis_x[cond1]
    axis_y[cond1] = -axis_y[cond1]
    cond2 = (grasp_angle > 1.5*np.pi) & (grasp_angle <= 2*np.pi)
    grasp_angle[cond2] = grasp_angle[cond2] - 2*np.pi

    cond3 = (grasp_angle < -0.5*np.pi) & (grasp_angle >= -1.5*np.pi)
    grasp_angle[cond3] = -(grasp_angle[cond3] + np.pi)
    axis_z[cond3] = -axis_z[cond3]
    axis_x[cond3] = -axis_x[cond3]
    axis_y[cond3] = -axis_y[cond3]
    cond4 = (grasp_angle < -1.5*np.pi) & (grasp_angle >= -2*np.pi)
    grasp_angle[cond4] = grasp_angle[cond4] + 2*np.pi    
    '''
    grasp_angle[no_grasp_mask] = -1

    axis_x, axis_y, axis_z, grasp_angle = axis_x.view(B, CN, 3), axis_y.view(B, CN, 3), axis_z.view(B, CN, 3), grasp_angle.view(B, CN)

    grasp_trans[:,:,:3]  = grasp_ori[:,:,:3,3].view(B, CN, 3)
    grasp_trans[:,:,3:6] = axis_y
    grasp_trans[:,:,6] = grasp_angle
    grasp_trans[:,:,7] = grasp_score_ori
    if grasp_trans.shape[2] > 8:
        grasp_trans[:,:,8] = antipodal_score_ori
        grasp_trans[:,:,9] = center_score_ori

    return grasp_trans

def _inv_transform_grasp(grasp_trans):
    '''
      Input:
        grasp_trans:[B, center_num, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
      Output:
        matrix: [B, center_num, 3, 4] 
                   [[x1, y1, z1, c1],
                    [x2, y2, z2, c2],
                    [x3, y3, z3, c3]]
        grasp_score_ori: [B, center_num]
    '''
    no_grasp_mask = (grasp_trans.view(-1,8)[:,-1] == -1)

    center = grasp_trans.view(-1,8)[:,:3]
    axis_y = grasp_trans.view(-1,8)[:,3:6]
    angle = grasp_trans.view(-1,8)[:,6]
    cos_t, sin_t = torch.cos(angle), torch.sin(angle)

    B = len(grasp_trans.view(-1,8))
    R1 = torch.zeros((B, 3, 3))
    for i in range(B):
        r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
        R1[i,:,:] = r

    norm_y = torch.norm(axis_y, dim=1)
    axis_y = torch.div(axis_y, norm_y.view(-1,1))
    zero = torch.zeros((B, 1), dtype=torch.float32)
    if axis_y.is_cuda:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).cuda()
        zero = zero.cuda()
        R1 = R1.cuda()
    else:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float)
    axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
    norm_x = torch.norm(axis_x, dim=1)
    axis_x = torch.div(axis_x, norm_x.view(-1,1))
    if axis_y.is_cuda:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    axis_z = torch.cross(axis_x, axis_y, dim=1)
    norm_z = torch.norm(axis_z, dim=1)
    axis_z = torch.div(axis_z, norm_z.view(-1,1))
    if axis_z.is_cuda:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).cuda()
    else:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float)
    matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
    matrix = torch.bmm(matrix, R1)
    approach = matrix[:,:,0]
    norm_x = torch.norm(approach, dim=1)
    approach = torch.div(approach, norm_x.view(-1,1))
    if approach.is_cuda:
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    minor_normal = torch.cross(approach, axis_y, dim=1)
    matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1), center.view(-1,3,1)), dim=2)#.permute(0,2,1)
    matrix[no_grasp_mask] = -1
    matrix = matrix.view(len(grasp_trans), -1, 3, 4)

    grasp_score_ori = grasp_trans[:,:,7]
    grasp_score_ori = grasp_score_ori.view(-1)
    grasp_score_ori[no_grasp_mask] = -1
    grasp_score_ori = grasp_score_ori.view(len(grasp_trans), -1)

    return matrix, grasp_score_ori

def _compute_distance(points1, points2):
    '''计算点
    '''
    #distance [len(points1), len(points2)]
    distance = -2 * points1[:, :3].mm(points2.transpose(1,0))
    distance += torch.sum(points2.mul(points2), 1).view(1,-1).repeat(points1.size()[0],1)
    distance += torch.sum(points1[:, :3].mul(points1[:, :3]), 1).view(-1,1).repeat(1, points2.size()[0])
    
    return distance.double()

def _get_local_points_batch(all_points, center, width, height, depth, r_time):
    '''
      Get the points around **one** scored center
        Input:
          all_points: [N, C]
          center:     [center_num, C]
        Return:
          group_points_index: [center_num, N]
    '''
    all_points_repeat = all_points[:, :3].repeat(len(center),1).view(-1,3)
    center_repeat = center[:, :3].repeat(1,len(all_points)).view(-1,3)
    dist = all_points_repeat - center_repeat
    max_radius = max(width, height, depth)*r_time
    # distance:  [center_num, N]
    distance = torch.sqrt(torch.mul(dist[:,0], dist[:,0]) + torch.mul(dist[:,1], dist[:,1]) + torch.mul(dist[:,2], dist[:,2])).view(len(center),-1)
    group_points_index = (distance <= max_radius)
    return group_points_index

def _get_local_points(all_points, center, width, height, depth, r_time):
    '''
      Get the points around **one** scored center
        Input:
          all_points: [N, C]
          center:     [C]
        Return:
          group_points_index: [uncertain number]
    '''
    max_radius = max(width, height, depth)*r_time
    group_points_index = (torch.abs(all_points[:,0]-center[0]) < max_radius) & (torch.abs(all_points[:,1]-center[1]) < max_radius) & (torch.abs(all_points[:,2]-center[2]) < max_radius) #& (torch.sum(all_points[:,:3] != center[:3], dim=1)/3).byte()
    group_points_index = torch.nonzero(group_points_index).view(-1)
    return group_points_index

def _get_group_pc(pc, center_pc, center_pc_index, group_num, width, height, depth, r_time):
    '''
     Input:
        pc              :[B,N,6]
        center_pc       :[B,center_num,6]
        center_pc_index :[B,center_num]
        group_num       :int
     Return:
        pc_group_index  :[B,center_num,group_num] index of grouped points of selected center in sampled points
        pc_group        :[B,center_num,group_num,6]
    '''
    B,A,C = pc.shape
    #获取
    center_num = center_pc.shape[1]
    pc_group = torch.full((B, center_num, group_num, C), -1.0)
    pc_group_index = torch.full((B, center_num, group_num), -1)
    if pc.is_cuda:
        pc_group, pc_group_index = pc_group.cuda(), pc_group_index.cuda()
        
    # Get the points around one scored center    
    for i in range(B):
        group_points_index = _get_local_points_batch(pc[i], center_pc[i], width, height, depth, r_time)
        for j in range(center_num):
            group_points_index_one = torch.nonzero(group_points_index[j]).view(-1)
            if len(group_points_index_one) >= group_num:
                group_points_index_one = group_points_index_one[np.random.choice(len(group_points_index_one), group_num, replace=False)]
            elif len(group_points_index_one) > 0:
                group_points_index_one = group_points_index_one[np.random.choice(len(group_points_index_one), group_num, replace=True)]
                
            if len(group_points_index_one) > 0:
                pc_group_index[i,j] = group_points_index_one
                pc_group[i,j] = pc[i,group_points_index_one]

    ##if pc.is_cuda:
    ##    pc = pc.cpu()
    ##for i in range(B):
    ##    cur_pc_o3d = open3d.geometry.PointCloud()
    ##    cur_pc_o3d.points = open3d.utility.Vector3dVector(pc[i,:,:3]) 
    ##    cur_pc_tree = open3d.geometry.KDTreeFlann(cur_pc_o3d)
    ##    for j in range(center_num):
    ##        [k, idx, _] = cur_pc_tree.search_radius_vector_3d(cur_pc_o3d.points[center_pc_index[i,j]], max(width, height, depth)*r_time)

    return pc_group_index, pc_group

def _select_score_center(pc, pre_score, center_num, score_thre):
    '''
     Get the points where their scores are positive as regression centers of grasps
     1.先根据阈值抽取出原始点云中的正向点集
     2.再根据FPS算法（不过，这里好像并没有这样算啊），计算出k个抓取中心点
     Input:
        pc              :[B,N,6]
        pre_score       :[B,N], belongs to [0,1]
        score_thre      :float, score threshold, belongs to (0,1)
        center_num      :int
     Return:
        center_pc       :[B, center_num, 6]
        center_pc_index :[B, center_num] index of selected center in sampled points
    '''
    B,A,C = pc.shape
    #将数据转换到CPU中
    pre_score = pre_score.cpu()
    #如果Batchsize只有1
    if B == 1:
        positive_pc_mask = (pre_score.view(-1) > score_thre)
        positive_pc_mask = (pre_score.view(-1) > score_thre)
        positive_pc_mask = positive_pc_mask.cpu().numpy()
        #
        map_index = torch.Tensor(np.nonzero(positive_pc_mask)[0]).view(-1).long()

        center_pc = torch.full((center_num, C), -1.0)
        center_pc_index = torch.full((center_num,), -1)

        pc = pc.view(-1,C)
        cur_pc = pc[map_index,:]
        if len(cur_pc) > center_num:
            center_pc_index = _F.farthest_point_sample(cur_pc[:,:3].view(1,-1,3).transpose(2,1), center_num).view(-1)

            center_pc_index = map_index[center_pc_index.long()]
            center_pc = pc[center_pc_index.long()]
            
        elif len(cur_pc) > 0:
            center_pc_index[:len(cur_pc)] = torch.arange(0, len(cur_pc))
            center_pc_index[len(cur_pc):] = torch.Tensor(np.random.choice(cur_pc.shape[0], center_num-len(cur_pc), replace=True))
            center_pc_index = map_index[center_pc_index.long()]
            center_pc = pc[center_pc_index.long()]
            
        else:
            center_pc_index = torch.Tensor(np.random.choice(pc.shape[0], center_num, replace=False))
            center_pc = pc[center_pc_index.long()]
    
        center_pc = center_pc.view(1,-1,C)
        center_pc_index = center_pc_index.view(1,-1)
        if pc.is_cuda:
            center_pc = center_pc.cuda()
            center_pc_index = center_pc_index.cuda()
        return center_pc, center_pc_index

    # ---------------------- for train -------------------
    positive_pc_mask = (pre_score > score_thre)

    center_pc = torch.full((B, center_num, C), -1.0)
    center_pc_index = torch.full((B, center_num), -1)
    for i in range(B):
        cur_pc = pc[i,positive_pc_mask[i],:]
        if len(cur_pc) > center_num:
            #center_pc_index[i] = torch.Tensor(np.random.choice(cur_pc.shape[0], center_num, replace=False))
            #center_pc_index[i] = _farthest_point_sample(cur_pc, center_num)
            center_pc_index[i] = _F.farthest_point_sample(cur_pc[:,:3].view(1,-1,3).transpose(2,1), center_num).view(-1)

            map_index = torch.nonzero(positive_pc_mask[i]).view(-1)
            center_pc_index[i] = map_index[center_pc_index[i].long()]
            center_pc[i] = pc[i, center_pc_index[i].long()]
            
        elif len(cur_pc) > 0:
            center_pc_index[i,:len(cur_pc)] = torch.arange(0, len(cur_pc))
            center_pc_index[i,len(cur_pc):] = torch.Tensor(np.random.choice(cur_pc.shape[0], center_num-len(cur_pc), replace=True))
            #center_pc[i] = cur_pc[center_pc_index[i].long()]

            map_index = torch.nonzero(positive_pc_mask[i]).view(-1)
            center_pc_index[i] = map_index[center_pc_index[i].long()]
            center_pc[i] = pc[i, center_pc_index[i].long()]
            
        else:
            center_pc_index[i] = torch.Tensor(np.random.choice(pc.shape[1], center_num, replace=False))
            center_pc[i] = pc[i, center_pc_index[i].long()]
    
    if pc.is_cuda:
       center_pc = center_pc.cuda()
       center_pc_index = center_pc_index.cuda()
    return center_pc, center_pc_index

def _farthest_point_sample(xyz, npoint):
    """
      Input:
        xyz: pointcloud data, [N, C]
        npoint: number of samples
      Return:
        centroids: sampled pointcloud index, [npoint]
    """
    cuda = xyz.is_cuda
    N, C = xyz.shape

    #import time
    #s1 = time.time()
    centroids = torch.zeros((npoint), dtype=torch.long)
    distance = torch.ones((N)) * 1e10
    farthest = torch.tensor([np.random.randint(0, N)], dtype=torch.long)
    if cuda:
        centroids, distance, farthest = centroids.cuda(), distance.cuda(), farthest.cuda()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :3].view(1, 3)
        dist = torch.sum((xyz[:, :3] - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        torch.argmin(distance)
        farthest = int(torch.argmax(distance))
    #e1 = time.time()
    #print(e1-s1)
    '''
    s2 = time.time()
    if cuda:
       xyz = xyz.cpu()
    xyz = np.array(xyz)
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :3].reshape(1, 3)
        dist = np.sum((xyz[:, :3] - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    centroids = torch.from_numpy(centroids)
    if cuda:
        centroids = centroids.cuda()
    e2 = time.time()
    print(e2-s2)
    '''
    return centroids
