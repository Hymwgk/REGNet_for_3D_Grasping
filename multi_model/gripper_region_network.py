import numpy as np
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
#from multi_model.utils.pointnet_test import PointNetPartFeature
from multi_model.utils.pointnet2 import PointNet2TwoStage, PointNet2Refine

class GripperRegionNetwork(nn.Module):
    """
    关于GraspRegion的网络部分，通过标志位可以选择性地实现对RefineNet的集成
    """
    def __init__(self, training, group_num, gripper_num, grasp_score_threshold, radius, reg_channel):
        '''
        training：bool型，选择是否加载RN网络
        group_num：Grasp Region包围球中的点数
        gripper_num：
        grasp_score_threshold：
        radius：
        reg_channel：单个anchor回归出的对应bias等参数(作者设置的通道是8； x,y,z,rx,ry,rz)
        '''
        super(GripperRegionNetwork, self).__init__()
        #是否
        self.group_number = group_num
        
        self.templates = _enumerate_templates()#枚举不同角度的anchors模板，这个只是把多种角度显式地表示出来，还没有和抓取中心点结合起来
        
        self.anchor_number = self.templates.shape[1]*self.templates.shape[2]#计算一下所有anchors的数量
        #
        self.gripper_number = gripper_num
        self.grasp_score_thre = grasp_score_threshold
        self.is_training_refine = training
        self.radius = radius
        self.reg_channel = reg_channel #每个anchor抓取要回归出的值一共8通道，其中3通道位置残差

        #利用包围球内点特征，回归抓取位置姿态残差与分数
        self.extrat_feature_region = PointNet2TwoStage(
            num_points=group_num, #包围球内点数
            input_chann=6, #点云通道数xyzrgb
            k_cls=self.anchor_number,#每个中心点对应M个anchor
            k_reg=self.reg_channel*self.anchor_number,#同时回归出M个anchor抓取的每个通道的res残差
            k_reg_theta=self.anchor_number)  #回归出theta的残差

        #构建RN网络（不一定用上去，要看模式的选择）
        self.extrat_feature_refine = PointNet2Refine(num_points=gripper_num, input_chann=6, k_cls=2, k_reg=self.reg_channel)

        self.criterion_cos = nn.CosineEmbeddingLoss(reduction='mean')
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

    def _enumerate_anchors(self, centers):
        '''根据给定的锚点位置坐标，把预设的姿态模板和位置坐标连接在一起，构成预设的系列anchors位姿向量
        每个抓取中心，将预设M个anchors
          Enumerate anchors.
          Input:
            centers: [B*num of centers, 3] -> x, y, z  输入是Batch的所有中心点的xyz坐标
            self.templates :[1,M,1,4] -> M:r_x r_y r_z_num 1:theta_num 4:r_x r_y r_z theta
          Return:
            t_anchors: [B*num of centers, M, 7] -> the number of anchors is M
                                                     7 means (x, y, z, rx, ry, rz, theta)
                                                     每个中心点将固定有M个模板，每个模板都是7维的
        '''
        if centers.cuda:
            self.templates = self.templates.cuda() #获取模板
        t_center = centers.view(centers.shape[0],1,1,-1).repeat(1,self.templates.shape[1],self.templates.shape[2],1)
        t_anchors = torch.cat( [t_center, self.templates.float().repeat(centers.shape[0],1,1,1)], dim=-1).view(-1, self.templates.shape[1]*self.templates.shape[2], 7)
        return t_anchors

    def compute_loss(self, first_grasp, anchors, first_cls, ground):
        '''在训练时，计算回归出的抓取和ground truth抓取之间的差别loss
        Input:  
            first_grasp : [B*center_num, num_anchor, 8]  回归出的残差+score
            anchors     : [B*center_num, num_anchor, 7]    anchors
            first_cls   : [B*center_num, num_anchor]           对B*M个anchors的分类结果
            ground      : [B, center_num, 8]  ground truth. 只在训练的时候有用，而在测试时候是没有的
                                    8 means the (p,r,theta,score) of a grasp. 说明groudtruth，在每个中心点处，只有一个真实抓取
        Return:
            next_grasp  : [len(select_center_index), 7]预测的完整抓取
            loss_tuple, correct_tuple
            next_gt : [len(select_center_index), 7]
        Loss一共有两部分：
        1. B*M个anchor的分类误差
        2. res的回归损失
        问题：groundtruth是如何确定是哪个编号的anchor离自己最近的呢？在哪里计算的？
        '''
        #### gmask: [len(gmask)]  true mask index of centers which has a corresponding grasp 
        BmulN_C = len(first_grasp) #一个batch中有多少个抓取中心
        # B, N_C = ground.shape[0], ground.shape[1]
        if ground is not None:#训练时
            '''在数据集中查到带有正抓取的center的index列表
            例如gmask=[0,1,3,5,6,7] 表明在第0，1，3，5，7号抓取中心点处可以在数据集中查找到合理的抓取
            ground.view(-1,ground.shape[2])     [B*center_num, 8] 注意这里实际上是10维度的
            gmask     len(gmask)=带有正抓取的抓取中心点的数量
            '''
            gmask = torch.nonzero(ground.view(-1,ground.shape[2])[:,-1] != -1).view(-1) #gmask[len(gmask) ]
            print(BmulN_C, "centers has", len(gmask), "grasps" )              
        else:#测试时
            gmask = torch.arange(0, BmulN_C)#
        if first_grasp.cuda:
            gmask = gmask.cuda()

        
        anchors = anchors[gmask, :, :]#利用gmask筛选出具有正抓取的预设anchor,  [len(gmask) ,num_anchor,7]
        tt = anchors.clone().detach().transpose(1,0).contiguous().view(-1,7) #拷贝并变形为[len(gmask)*num_anchor, 7]
        first_grasp, first_cls = first_grasp[gmask], first_cls[gmask]#提取出与groundtruth正抓取中心点对应的回归预测grasp，以及预测的类别

        
        first_grasp = first_grasp.transpose(1,0).contiguous().view(-1, first_grasp.shape[2])# 变形为[len(gmask)*num_anchor, 8] 

        num_anchor = self.templates.shape[1]*self.templates.shape[2]
        #计算预测的一个batch的anchor索引
        _, predict_8 = torch.max(first_cls.transpose(1,0).contiguous(), dim=0)#[len(gmask),]找到预测分类值为1的anchor索引
        final_mask = predict_8.clone().view(-1)#找到预测分类值为1的anchor索引
        for i in range(len(final_mask)):#
            final_mask[i] = final_mask[i] * len(final_mask) + i

        #筛选出来一个batch的预测抓取残差，以及预测anchor姿态
        first_grasp_pre, tt_pre = first_grasp[final_mask], tt[final_mask]

        #利用预测残差与anchor姿态，还原出完整抓取位姿
        sum_r_pre = torch.sqrt(torch.sum(torch.mul(first_grasp_pre[:,3:6]+tt_pre[:,3:6],
                 first_grasp_pre[:,3:6]+tt_pre[:,3:6]), dim=1).add_(1e-12) ).view(-1,1)#求原始binormal轴模长

        first_grasp_center_pre = first_grasp_pre[:, :3]*self.radius + tt_pre[:,:3]#还原抓取中心[len(gmask),3]
        first_grasp_r_pre = torch.div(first_grasp_pre[:,3:6]+tt_pre[:,3:6], sum_r_pre)#还原binormal轴（单位化）[len(gmask),3]
        first_grasp_angle_pre = np.pi * (first_grasp_pre[:,6:7]+tt_pre[:,6:7])#还原抓取角度[len(gmask),1]
        first_grasp_score_pre = first_grasp_pre[:,7:]#分数[len(gmask),1]
       
        #构成完整的预测位姿+score的tensor
        next_grasp = torch.cat((first_grasp_center_pre, first_grasp_r_pre, \
                                    first_grasp_angle_pre, first_grasp_score_pre), dim=-1)#[]
        
        loss_tuple = (None, None)
        correct_tuple, next_gt, tt_gt = (None, None, None, None), None, None

        if ground is not None:
            '''计算GRN阶段的Loss
            两部分Loss：1.
            '''
            #[B,center_num,10]->[B*center_num,7]->[len(gmask),7]->[len(gmask)*anchor_num,7]
            repeat_ground = ground[:,:,:7].contiguous().view(-1, 7)[gmask, :].\
                repeat(self.templates.shape[1]*self.templates.shape[2],1)
            repeat_ground_truth = ground[:,:,7:].contiguous().view(-1, ground.shape[2]-7)[gmask, :].\
                repeat(self.templates.shape[1]*self.templates.shape[2],1)
            ## r_sim: [num_anchor, len(gmask)]，计算ground truth的binormal轴与每个预设的anchor之间的cos值
            r_sim = compute_cos_sim(tt[:,3:6], repeat_ground[:,3:6]).view(-1).view(num_anchor, -1)

            ## ground_8: [len(gmask)]
            sim = r_sim.clone().transpose(1,0)
            #[len(gmask),anchor_num] 找到ground truth的binormal轴与哪个预设的anchor之间最近，得到index
            sort_cls, sort_index = torch.sort(sim, dim=1, descending=False)
            ground_8 = sort_index[:,0].view(-1)#[len(gmask),] 每个ground truth最相似预设anchor的index
            
            #print(ground_8)
            iou_nonzero = ground_8.clone()
            for i in range(len(iou_nonzero)):
                iou_nonzero[i] = iou_nonzero[i] * len(iou_nonzero) + i
            
            len_ground_anchor, num_0, num_t = np.zeros([num_anchor]), 0, 0
            #一共M种anchor，计算每种anchor与之最近的ground truth的数量
            for i in range(num_anchor):
                len_ground_anchor[i] = (ground_8==i).sum()
                if len_ground_anchor[i] == 0:
                    num_0 += 1
            for i in range(num_anchor):#计算每种anchor与之最近的预测抓取的数量
                print(i, "num:", (predict_8==i).sum()) 


            min_len_ground_anchor = len_ground_anchor.min()#找到最少的数量
            if min_len_ground_anchor == 0:
                min_len_ground_anchor = 1
            ground_anchor_index = torch.zeros([num_anchor-num_0, int(min_len_ground_anchor)])
            for i in range(num_anchor):
                cur_anchor_index = torch.nonzero(ground_8==i).view(-1)
                if len(cur_anchor_index) == 0:
                    continue
                ground_anchor_index[num_t] = cur_anchor_index[np.random.choice(len(cur_anchor_index), \
                                                          int(min_len_ground_anchor), replace=False)]
                num_t += 1
                
            ground_anchor_index = ground_anchor_index.view(-1).long()
            if ground.is_cuda:
                ground_anchor_index = ground_anchor_index.cuda()
            #交叉熵计算分类Loss，
            loss_class = self.criterion_cls(first_cls, ground_8.long())
            #loss_class = self.criterion_cls(first_cls[ground_anchor_index], ground_8[ground_anchor_index].long())
            print("regression stage 1 class loss:", loss_class)
            
            Tcls = ( ground_8 == predict_8).sum().float()#anchor分类正确的个数
            Fcls = (ground_8 != predict_8).sum().float()#anchor分类失败的个数
            correct_tuple = (Tcls, Fcls)
            acc = Tcls / (Tcls + Fcls)
            print(Tcls, Fcls, "acc1:", acc)
            

            first_grasp_gt, tt_gt  = first_grasp[iou_nonzero], tt[iou_nonzero]
            sum_r_gt               = torch.sqrt(torch.sum(torch.mul(first_grasp_gt[:,3:6]+tt_gt[:,3:6], \
                                                first_grasp_gt[:,3:6]+tt_gt[:,3:6]), dim=1).add_(1e-12) ).view(-1,1)
            first_grasp_center_gt  = first_grasp_gt[:,:3]*self.radius + tt_gt[:,:3]
            first_grasp_r_gt       = torch.div(first_grasp_gt[:,3:6]+tt_gt[:,3:6], sum_r_gt)
            first_grasp_delta_r_gt = torch.mul(first_grasp_gt[:,3:6], sum_r_gt)
            first_grasp_angle_gt   = np.pi * (first_grasp_gt[:,6:7]+tt_gt[:,6:7])
            first_grasp_score_gt   = first_grasp_gt[:,7:]
            # (sinx, cosx)
            #first_grasp_angle_gt = torch.atan2(first_grasp_gt[:,-1].view(-1,1), first_grasp_gt[:,-2].view(-1,1)).view(-1,1)

            ground_gt = repeat_ground[iou_nonzero]                 # same as repeat_ground[final_mask]
            ground_score_gt = repeat_ground_truth[iou_nonzero]  # same as repeat_ground_truth[final_mask]
            #计算回归姿态+score差的Loss，对比对象为res残差
            loss_first1_gt  = F.smooth_l1_loss(first_grasp_gt[:,:3],  (ground_gt[:,:3]-tt_gt[:,:3]) / self.radius, reduction='mean')#res_xyz的Loss
            loss_first2_gt  = F.smooth_l1_loss(first_grasp_delta_r_gt, ground_gt[:,3:6]-tt_gt[:,3:6], reduction='mean')#res_rxyz的Loss
            loss_first3_gt  = F.smooth_l1_loss(first_grasp_gt[:,6:7], (ground_gt[:,6:7]-tt_gt[:,6:7]) / np.pi, reduction='mean')#res_theta的Loss
            loss_first4_gt  = F.smooth_l1_loss(first_grasp_gt[:,7:],   ground_score_gt, reduction='mean')#score的Loss
            print("regress loss of stage2", loss_first1_gt.data, loss_first2_gt.data, loss_first3_gt.data, loss_first4_gt.data)
            #同样是计算回归姿态+score差的Loss，对比对象为完整姿态
            tensor_y_gt = torch.ones(len(iou_nonzero), 1)
            loss_center_gt  = F.smooth_l1_loss  (first_grasp_center_gt, ground_gt[:,:3], reduction='mean').data
            loss_cos_r_gt   = self.criterion_cos(first_grasp_r_gt, ground_gt[:,3:6], tensor_y_gt.cuda()).data
            loss_theta_gt   = F.smooth_l1_loss  (first_grasp_angle_gt, ground_gt[:,6:7], reduction='mean').data
            loss_score_gt   = loss_first4_gt.data
            print("under gt class loss", loss_center_gt, loss_cos_r_gt, loss_theta_gt, loss_score_gt)

            tensor_y_pre = torch.ones(len(final_mask), 1)
            loss_center_pre = F.smooth_l1_loss  ( first_grasp_center_pre, ground_gt[:,:3], reduction='mean').data
            loss_cos_r_pre  = self.criterion_cos( first_grasp_r_pre, ground_gt[:,3:6], tensor_y_pre.cuda()).data
            loss_theta_pre  = F.smooth_l1_loss  ( first_grasp_angle_pre, ground_gt[:,6:7], reduction='mean').data
            loss_score_pre  = F.smooth_l1_loss  ( first_grasp_score_pre, ground_score_gt, reduction='mean').data
            print("under pre class loss", loss_center_pre, loss_cos_r_pre, loss_theta_pre, loss_score_pre)
            
            #完整的ground truth抓取向量(pose+score)
            next_gt = torch.cat((ground_gt, ground_score_gt), dim=1)# [len(gmask), 10]

            loss = loss_first1_gt*10 + loss_first2_gt*5 + loss_first3_gt + loss_first4_gt + loss_class
            loss_tuple = (loss, loss_class.data, loss_first1_gt.data, loss_first2_gt.data, loss_first3_gt.data, \
                            loss_first4_gt.data, loss_center_pre, loss_cos_r_pre, loss_theta_pre, loss_score_pre, )

        return next_grasp, loss_tuple, correct_tuple, next_gt, tt_gt, gmask

    def compute_loss_refine(self, next_grasp, next_x_cls, next_x_reg, next_gt):
        '''计算RN子网络的Loss
          Input:
            next_grasp      :[len(gripper_mask),8] regressed grasp from the stage1
            next_x_cls      :[len(gripper_mask),2]
            next_x_reg      :[len(gripper_mask),8] delta grasp from the stage2 (loss)
            next_gt         :[len(gripper_mask),8]  ground truth 并非残差gt，而是完整的抓取gt
          Return:
            final_grasp_select       : [len(class_select), 8] 
            select_grasp_class_stage2: [len(class_select), 8]
            class_select             : [len(class_select)]
            loss_stage2              : tuple
            correct_stage2_tuple     : tuple
        '''
        print("Refine Module init number:", next_grasp.shape[0])
        final_grasp = next_grasp.clone()
        final_grasp[:,:3] = final_grasp[:,:3] + next_x_reg[:,:3] * self.radius#最终预测的抓取
        final_grasp[:,3:] = final_grasp[:,3:] + next_x_reg[:,3:] #最终预测的分数
        
        # next_x_cls[:,1] += 1
        predit_formal = torch.max(next_x_cls, dim=-1)[1]
        class_select  = torch.nonzero(predit_formal==1).view(-1)#预测的正抓取的index
        score_mask    = (predit_formal==1) & (final_grasp[:,7] > self.grasp_score_thre)#选择回归出的抓取
        score_select  = torch.nonzero(score_mask).view(-1)#预测的正且高分数的抓取index
        print("########################################")
        print("predict class 0:", torch.sum((predit_formal==0)).data, "; predict class 1:", torch.sum((predit_formal==1)).data)

        select_grasp_class  = final_grasp[class_select].data#筛选出预测出的正完整抓取姿态
        select_grasp_score  = final_grasp[score_select].data#筛选出预测的正且高分数的完整抓取姿态
        select_grasp_class_stage2 = next_grasp[class_select].data#

        print("final grasp: {}".format(len(select_grasp_class)))
        print("final >{} score grasp: {}".format(self.grasp_score_thre, len(score_select)))
        print("########################################")
        loss_refine_tuple, correct_refine_tuple = (None, None), (None, None, None, None)

        if next_gt is not None:
            #计算gt的分类
            gt_class = torch.zeros((len(next_gt)))
            if next_grasp.is_cuda:
                gt_class = gt_class.cuda()
            #计算GRN预测位置与gt位置之间的距离
            center_dist = (next_grasp[:,:3] - next_gt[:,:3]) 
            #预测与ground truth之间的距离之差在2.5mm以内的mask
            center_dist_mask = (torch.sqrt(torch.mul(center_dist[:,0],center_dist[:,0])+torch.mul(center_dist[:,1],center_dist[:,1])\
                                                +torch.mul(center_dist[:,2],center_dist[:,2])) < 0.025).view(-1) 
            # 找到GRN pre_r与gt_r之间的夹角小于60度的mask
            r_sim = compute_cos_sim(next_grasp[:,3:6], next_gt[:,3:6]).view(-1)
            r_sim_mask = (r_sim < 0.5).view(-1) # cos60 = 0.5 #0.234
            #GRN pre_theta与gt_theta之间的角度小于60度的mask
            theta_sim = torch.abs(next_grasp[:,6] - next_gt[:,6]) 
            theta_sim_mask = (theta_sim < 1.047).view(-1) # 1.047 = 60/180*np.pi

            #同时考虑位置姿态的mask
            class_mask = (center_dist_mask & r_sim_mask & theta_sim_mask)
            gt_class[class_mask] = 1
            gt_class_1 = torch.nonzero(gt_class == 1).view(-1)#预测的合法grasp index
            gt_class_0 = torch.nonzero(gt_class == 0).view(-1)#预测的非法grasp index

            num_0, num_1 = len(gt_class_0), len(gt_class_1)
            print("class 0:", num_0, "; class 1:", num_1)
            num = min(num_0, num_1)
            loss = torch.tensor((0), dtype=torch.float)
            loss_class, loss_grasp_center, loss_grasp_r, loss_grasp_theta, loss_grasp_score = loss.clone(), loss.clone(), loss.clone(), loss.clone(), loss.clone()

            loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2 = loss.clone(), loss.clone(), loss.clone(), loss.clone()
            loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre = loss.clone(), loss.clone(), loss.clone(), loss.clone()
            loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score = loss.clone(), loss.clone(), loss.clone(), loss.clone()

            if next_x_cls.is_cuda:
                loss, loss_class = loss.cuda(), loss_class.cuda()
                loss_grasp_center, loss_grasp_r, loss_grasp_theta, loss_grasp_score = loss_grasp_center.cuda(), loss_grasp_r.cuda(), loss_grasp_theta.cuda(), loss_grasp_score.cuda()
                
                loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2 = loss_center_pre_stage2.cuda(), loss_r_cos_pre_stage2.cuda(), loss_theta_pre_stage2.cuda(), loss_score_pre_stage2.cuda()
                loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre = loss_center_pre.cuda(), loss_r_cos_pre.cuda(), loss_theta_pre.cuda(), loss_score_pre.cuda()
                loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score = loss_center_pre_score.cuda(), loss_r_cos_pre_score.cuda(), loss_theta_pre_score.cuda(), loss_score_pre_score.cuda()
                
            if num > 0:
                index_0 = gt_class_0[np.random.choice(num_0, num, replace=False)].view(-1)
                index_1 = gt_class_1[np.random.choice(num_1, num, replace=False)].view(-1)
                index = torch.cat((index_0, index_1), dim=-1)
                #分类Loss
                loss_class = self.criterion_cls(next_x_cls.view(-1,2)[index], gt_class.view(-1)[index].long())

                #使用gt_cls，筛选并直接计算pre_res与gt_res之间的Loss
                loss_grasp_center = F.smooth_l1_loss(next_x_reg[gt_class_1,:3], (next_gt[gt_class_1,:3]-next_grasp[gt_class_1,:3]) / self.radius, reduction='mean')
                loss_grasp_r      = F.smooth_l1_loss(next_x_reg[gt_class_1,3:6], (next_gt[gt_class_1,3:6]-next_grasp[gt_class_1,3:6]) , reduction='mean')
                loss_grasp_theta  = F.smooth_l1_loss(next_x_reg[gt_class_1,6], (next_gt[gt_class_1,6]-next_grasp[gt_class_1,6]) , reduction='mean')
                loss_grasp_score  = F.smooth_l1_loss(next_x_reg[gt_class_1,7:], (next_gt[gt_class_1,7:]-next_grasp[gt_class_1,7:]) , reduction='mean')
                loss              = loss_class + loss_grasp_center + loss_grasp_r + loss_grasp_theta + loss_grasp_score

            if len(class_select) > 0:
                tensor_y = torch.ones(len(class_select), 1)
                if next_x_cls.is_cuda:
                    tensor_y = tensor_y.cuda()
                #不带score差异分类(仅位姿差异) pre_cls_no_score,筛选出完整pre_grasp与完整gt_grasp之间的Loss 
                loss_center_pre        = F.smooth_l1_loss(select_grasp_class[:,:3], next_gt[class_select,:3], reduction='mean').data
                loss_r_cos_pre        = self.criterion_cos(select_grasp_class[:,3:6], next_gt[class_select,3:6],tensor_y).data
                loss_theta_pre        = F.smooth_l1_loss(select_grasp_class[:,6], next_gt[class_select,6], reduction='mean').data
                loss_score_pre        = F.smooth_l1_loss(select_grasp_class[:,7:], next_gt[class_select,7:], reduction='mean').data
                
                #不带score差异分类(仅位姿差异) pre_cls_no_score,筛选出残差pre_res与完整gt_grasp之间的Loss; 
                loss_center_pre_stage2 = F.smooth_l1_loss(select_grasp_class_stage2[:,:3], next_gt[class_select,:3], reduction='mean').data
                loss_r_cos_pre_stage2 = self.criterion_cos(select_grasp_class_stage2[:,3:6], next_gt[class_select,3:6],tensor_y).data
                loss_theta_pre_stage2 = F.smooth_l1_loss(select_grasp_class_stage2[:,6], next_gt[class_select,6], reduction='mean').data
                loss_score_pre_stage2 = F.smooth_l1_loss(select_grasp_class_stage2[:,7:], next_gt[class_select,7:], reduction='mean').data
                
                #带有score差异分类pre_cls_with_score,筛选出完整pre_grasp与完整gt_grasp;  
                loss_center_pre_score  = F.smooth_l1_loss(select_grasp_score[:,:3], next_gt[score_select,:3], reduction='mean').data
                loss_r_cos_pre_score  = self.criterion_cos(select_grasp_score[:,3:6], next_gt[score_select,3:6],tensor_y).data
                loss_theta_pre_score  = F.smooth_l1_loss(select_grasp_score[:,6], next_gt[score_select,6], reduction='mean').data
                loss_score_pre_score  = F.smooth_l1_loss(select_grasp_score[:,7:], next_gt[score_select,7:], reduction='mean').data

                print("loss stage 2 - class: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2))
                print("loss stage 3 - class: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre) )
                print("loss stage 3 - score: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score))
            
            #
            TP = ((gt_class.view(-1) == 1 ) & (predit_formal.view(-1) == 1)).sum().float()#真阳性
            TN = ((gt_class.view(-1) == 0 ) & (predit_formal.view(-1) == 0)).sum().float()#真阴性
            FP = ((gt_class.view(-1) == 0 ) & (predit_formal.view(-1) == 1)).sum().float()#假阳性
            FN = ((gt_class.view(-1) == 1 ) & (predit_formal.view(-1) == 0)).sum().float()#假阴性
            print(TP,TN,FN,FP)
            acc = (TP + TN) / (TP + TN + FP + FN)#计算准确率
            correct_refine_tuple = (TP, TN, FP, FN)
            print("stage2 acc:", acc)

            loss_refine_tuple = (loss, loss_class.data, loss_grasp_center.data, loss_grasp_r.data, loss_grasp_theta.data, loss_grasp_score, \
                                                loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2,
                                                loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre,
                                                loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score)
                                    
        return select_grasp_class, select_grasp_score, select_grasp_class_stage2, class_select, score_select, loss_refine_tuple, correct_refine_tuple

    def refine_forward(self, pc_group_more_xyz, pc_group_more_index, gmask, all_feature, \
                    group_feature_mp, next_grasp, gripper_params, next_gt=None):
        '''refine net前向计算
          pc_group_more_xyz   :[B*center_num, group_num_more, 6]
          pc_group_more_index :[B, center_num, group_num_more]
          gmask           :[len(gmask)]
          all_feature         :[B, A, Feature]
          group_feature_mp    :[B*N_C, 128, 1] 每个抓取包围球经过特征提取之后的128维度特征
          next_grasp          :[len(gmask), 10]
          gripper_params      :List [torch.tensor(),float,float] widths, height, depth
          next_gt             :[len(gmask), 10]
        '''
        B, feature_len = all_feature.shape[0], all_feature.shape[2]
        N_C, N_G_M = pc_group_more_index.shape[1], pc_group_more_index.shape[2]
        cuda = pc_group_more_xyz.is_cuda
        
        #将包围球内部区域
        gripper_pc, gripper_pc_index, gripper_pc_index_inall, gripper_mask = get_gripper_region_transform(pc_group_more_xyz[gmask], 
                        pc_group_more_index.view(-1,N_G_M)[gmask], next_grasp, self.gripper_number, gripper_params)
        select_grasp_class, select_grasp_score, select_grasp_class_stage2 = None, None, None
        final_mask, final_mask_sthre, loss_refine_tuple, correct_refine_tuple = None, None, (None, None), (None, None)


        if len(gripper_mask) >= 2:#内部点数满足条件的抓取数量大于2
            all_feature_new = all_feature.contiguous().view(-1, feature_len)#[B*A,feature]
            add = torch.arange(B).view(-1,1).repeat(1, N_C).view(-1)[gmask].view(-1,1).repeat(1, self.gripper_number)
            if pc_group_more_index.cuda:
                add = add.cuda()
            #### gripper_pc_index_inall: [len(gmask), region_num]
            gripper_pc_index_inall_new = (gripper_pc_index_inall.long() + add * all_feature.shape[1]).view(-1)
            gripper_feature = all_feature_new[gripper_pc_index_inall_new].view(-1, self.gripper_number, feature_len)[gripper_mask]#.detach()
            #### gripper_feature: [len(gripper_mask), self.gripper_number, feature_len]
            
            group_feature_mp = group_feature_mp.view(-1,128)[gripper_mask].contiguous()
            
            # next_x_cls: [len(gripper_mask), 2], next_x_reg: [len(gripper_mask), 8]
            #利用夹爪内部点云的feature和包围球整体feature结合回归出回归res残差
            next_x_cls, next_x_reg = self.extrat_feature_refine(gripper_feature.permute(0,2,1), group_feature_mp)
            if next_gt is not None:
                next_gt = next_gt[gripper_mask]
            #计算RN  Loss
            select_grasp_class, select_grasp_score, select_grasp_class_stage2, class_select, score_select, loss_refine_tuple, \
                                correct_refine_tuple = self.compute_loss_refine(next_grasp[gripper_mask], next_x_cls, next_x_reg, next_gt)

            if next_gt is not None:
                next_gt = next_gt[class_select]#挑出最终的预测抓取对应的gt抓取

            final_mask = gmask.clone()[gripper_mask][class_select] 
            final_mask_sthre = gmask.clone()[gripper_mask][score_select] 

        return select_grasp_class, select_grasp_score, select_grasp_class_stage2, final_mask, \
                            final_mask_sthre, loss_refine_tuple, correct_refine_tuple, next_gt
        
    def forward(self, pc_group, pc_group_more, pc_group_index, pc_group_more_index, center_pc, \
                    center_pc_index, pc, all_feature, gripper_params, ground_grasp=None, data_path=None):
        '''GRN网络的前向传播
        pc_group                            :[B, center_num, group_num, 6]          k1 个包围球内点的xyzrgb值
        pc_group_more               :[B, center_num, group_num_more, 6]   
        pc_group_index              :[B, center_num, group_num]   k1个包围球内部点在pc中的索引
        pc_group_more_index :[B, center_num, group_num_more]
        center_pc                            :[B, center_num, 6]   FPS返回的k1个抓取中心点的xyzrgb
        center_pc_index              :[B, center_num]        k1个抓取中心点在原始pc中的索引
        pc                                           :[B, A, 6]  原始（剪切后）点云
        all_feature                          :[B, A, Feature]  所有点云点的点特征
        gripper_params               :List [float,float,float] width, height, depth 夹爪参数
        ground_grasp:                  :[B,center_num,8] the labels of grasps (ground truth + score) ground truth抓取
        '''
        B,N_C,N_G,C = pc_group.shape
        _,_,N_G_M,_ = pc_group_more.shape
        
        cuda = pc.is_cuda
        final_grasp, final_grasp_stage1 = torch.Tensor(), torch.Tensor()
        
        loss_tuple, loss_tuple_stage2 = (None, None), (None, None)#设置两个loss

        #在这里，获取到每个锚点的多个anchors（位置+姿态）
        anchors = self._enumerate_anchors(center_pc[:,:,:3].view(-1,3).float())# [B*center_num, M, 7]
        
        #anchor_number = anchors.shape[1]#
        #pc_group_xyz = pc_group[:,:,:,:6].clone().view(B*N_C,N_G,-1)

        pc_group_more_xyz = pc_group_more[:,:,:,:6].clone().view(B*N_C,-1,6)
        
        feature_len = all_feature.shape[2]#获得每个点的特征长度
        #变形
        all_feature_new = all_feature.contiguous().view(-1, feature_len)#[B,A,FL] -> [B*A,FL]
        
        add = torch.arange(B).view(-1,1).repeat(1, N_C*N_G)
        if pc_group_index.is_cuda:
            add = add.cuda()

        #[B,N_C,N_G]->[B,N_C*N_G]->[B*N_C*N_G]  因此需要加上长度为点云数A的步长
        pc_group_index_new = (pc_group_index.long().view(B, N_C*N_G) + add * all_feature.shape[1]).view(-1)

        #根据索引抽取每个包围球中各个点的点特征
        pc_group_features = all_feature_new[pc_group_index_new].view(B, N_C, N_G, feature_len)
        #变形[B,N_C,N_G,FL]->[B*N_C,N_G,FL]
        pc_group_features = pc_group_features.view(-1, N_G, feature_len)#[gmask]#.detach()
        
        '''先把center_feature变换顺序[B*N_C,N_G,FL] -> [B*N_C,FL,N_G]再
        输入网络，去抽取每个包围球中的特征，每个包围球都代表了一个center，需要回归出num_anchor个grasp bias      
        pc_group_features:[B*N_C, N_G, feature_len]每个包围球中的点的特征
        x_cls:                             [B*N_C, num_anchor]        对B*N_C个包围球中的每个anchor进行分类的结果
        x_reg:                            [B*N_C, num_anchor, 8]    对B*N_C个包围球中的每个anchor进行位姿res+score回归的结果
        mp_center_feature:[B*N_C, FL(128),1]                每个group的maxpool之后的特征向量，表征该group的全局特征
        '''
        x_cls, x_reg, mp_center_feature = self.extrat_feature_region(pc_group_features.permute(0,2,1), None)
        
        '''将残差与anchor结合，对比ground truth，计算GRN网络的Loss
        next_grasp: [len(gmask), 8]预测的完整抓取向量(pose+score)
        next_gt: [len(gmask), 8]真实的完整抓取向量(pose+score)
        loss_tuple[] Loss组
        correct_tuple :预测的准确率
        gmask: gmask   [len(gmask),]      对比[B*center_num,]
        '''
        next_grasp, loss_tuple, correct_tuple, next_gt, tt_pre, gmask = self.compute_loss(x_reg, anchors, x_cls, ground_grasp)
        
        # print("gmask",gmask)
        keep_grasp_num_stage2 = [(torch.sum((gmask<(i+1)*N_C) & (gmask>=i*N_C))) for i in range(B)] 
        
        select_grasp_class, select_grasp_score, select_grasp_class_stage2, final_mask, final_mask_sthre, keep_grasp_num_stage3, \
            keep_grasp_num_stage3_score, loss_refine_tuple, correct_refine_tuple, gt = None, None, None, None, None, None, None, None, None, None
        #如果使用了refine网络的话
        if self.is_training_refine:
            select_grasp_class, select_grasp_score, select_grasp_class_stage2, final_mask, final_mask_sthre, loss_refine_tuple, \
                            correct_refine_tuple, gt = self.refine_forward(pc_group_more_xyz, pc_group_more_index, gmask, \
                            all_feature, mp_center_feature, next_grasp.detach(), gripper_params, next_gt)

            if final_mask is not None:
                keep_grasp_num_stage3       = [(torch.sum((final_mask<(i+1)*N_C) & (final_mask>=i*N_C))) for i in range(B)] 
                keep_grasp_num_stage3_score = [(torch.sum((final_mask_sthre<(i+1)*N_C) & (final_mask_sthre>=i*N_C))) for i in range(B)] 
            else:
                keep_grasp_num_stage3       = [0 for i in range(B)] 
                keep_grasp_num_stage3_score = [0 for i in range(B)] 

        # print("!!!!!!!!!!!!!!!!!!!!",B, N_C)
        # print(keep_grasp_num_stage2)
        return next_grasp.detach(), keep_grasp_num_stage2, gmask, loss_tuple, correct_tuple, next_gt, \
                select_grasp_class, select_grasp_score, select_grasp_class_stage2, keep_grasp_num_stage3, \
                keep_grasp_num_stage3_score, final_mask, final_mask_sthre, loss_refine_tuple, correct_refine_tuple, gt

def get_gripper_region_transform(group_points, group_index, grasp, region_num, gripper_params):
    '''查看预测抓取姿态下，夹爪内部点云的数量，并返回满足条件的抓取的index
      Return the transformed local points in the closing area of gripper.
      Input: group_points: [B*center_num,group_num_more,6] 包围球内部点xyzrgb
             group_index : [len(gmask),group_num_more] 包围球内部点在完整pc中的索引
             grasp:        [len(gmask),7] 预测抓取向量
             region_num:   the number of saved points in the closing area 夹爪内部点的最少数量
      Return:    
            gripper_pc : [len(gmask),region_num,6]  len(gmask)个抓取，夹爪闭合区域点的xyzrgb
            gripper_pc_index: [len(gmask),region_num]  len(gmask)个抓取，夹爪闭合区域点相对于包围球内部点的索引
            gripper_pc_index_inall:[len(gmask),region_num]  len(gmask)个抓取，夹爪闭合区域点相对于完整pc的索引
            gripper_mask_index: [len(gripper_mask_index),] 内部点数大于指定数量的预测抓取的索引，len(gripper_mask_index)<=len(gmask)
    '''
    widths, height, depths = gripper_params
    B, _ = grasp.shape #len(gmask)
    center = grasp[:, 0:3].float()#预测中心点坐标 [len(gmask),3]
    axis_y = grasp[:, 3:6].float()#预测r轴向量 [len(gmask),3]
    angle = grasp[:, 6].float()#预测theta   [len(gmask),1]
    cuda = center.is_cuda

    cos_t, sin_t = torch.cos(angle), torch.sin(angle)
    # R1 = torch.zeros((B, 3, 3))
    # for i in range(B):
    #     r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
    #     R1[i,:,:] = r
    one, zero = torch.ones((B, 1), dtype=torch.float32), torch.zeros((B, 1), dtype=torch.float32)
    if cuda:
        one, zero = one.cuda(), zero.cuda()
    R1 = torch.cat( (cos_t.view(B,1), zero, -sin_t.view(B,1), zero, one, zero, sin_t.view(B,1), 
                        zero, cos_t.view(B,1)), dim=1).view(B,3,3)
    if cuda:
        R1=R1.cuda()

    #预测binormal单位化
    norm_y = torch.norm(axis_y, dim=1).add_(1e-12)
    axis_y = torch.div(axis_y, norm_y.view(-1,1))

    if cuda:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).cuda()
    else:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float)
    #找到在W:X-O-Y平面内的一个与W-Y轴垂直的向量，作为临时axis_x
    axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
    #预测approach单位化
    norm_x = torch.norm(axis_x, dim=1).add_(1e-12)
    axis_x = torch.div(axis_x, norm_x.view(-1,1))
    if cuda:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)
    #叉乘得到预测minor_normal,并单位化
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    norm_z = torch.norm(axis_z, dim=1)
    axis_z = torch.div(axis_z, norm_z.view(-1,1))
    #
    if cuda:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).cuda()
    else:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float)
    #构造临时预测抓取姿态矩阵[len(gmask),3,3]
    matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
    if cuda:
        matrix = matrix.cuda()
    #经过旋转得到预测的抓取姿态矩阵[len(gmask),3,3]
    matrix = torch.bmm(matrix, R1)
    
    approach = matrix[:,:,0]
    norm_x = torch.norm(approach, dim=1).add_(1e-12)
    approach = torch.div(approach, norm_x.view(-1,1))

    if cuda:
        axis_y = axis_y.cuda()
        group_points = group_points.cuda()
        center = center.cuda()
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    minor_normal = torch.cross(approach, axis_y, dim=1)

    #求逆矩阵
    matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1)), dim=2).permute(0,2,1)
    ## pcs_t: [B,group_num_more,3] 求得包围球内部点云坐标Gp 
    pcs_t = torch.bmm(matrix, (group_points[:,:,:3].float() - \
                        center.view(-1,1,3).repeat(1, group_points.shape[1], 1).float()).permute(0,2,1)).permute(0,2,1)

    # torch.tensor [B,1]  or  float
    #获取夹爪限制包围盒尺寸
    x_limit = depths.float().view(-1,1)/2 if type(depths) is torch.Tensor else depths / 2 
    z_limit = height/2 # float
    # torch.tensor [B,1]  or  float
    y_limit = widths.float().view(-1,1)/2 if type(widths) is torch.Tensor else widths / 2

    #
    gripper_pc = torch.full((B,region_num,group_points.shape[2]),-1)
    #gripper_pc_formal = torch.full((B,region_num,group_points.shape[2]),-1)
    gripper_pc_index       = torch.full((B,region_num),-1)
    gripper_pc_index_inall = torch.full((B,region_num),-1)
    gripper_mask = torch.zeros((B))
    
    x1 = pcs_t[:,:,0] > 0#在夹爪内部
    x2 = pcs_t[:,:,0] < x_limit#小于夹爪深度
    y1 = pcs_t[:,:,1] > -y_limit#
    y2 = pcs_t[:,:,1] < y_limit
    z1 = pcs_t[:,:,2] > -z_limit
    z2 = pcs_t[:,:,2] < z_limit
    
    #
    a = torch.cat((x1.view(B,-1,1), x2.view(B,-1,1), y1.view(B,-1,1), \
                    y2.view(B,-1,1), z1.view(B,-1,1), z2.view(B,-1,1)), dim=-1)
    for i in range(B):
        index = torch.nonzero(torch.sum(a[i], dim=-1) == 6).view(-1)#满足所有夹爪尺寸限制的点索引
        if len(index) > region_num:#大于指定数量就裁切
            #print(len(index))
            index = index[np.random.choice(len(index),region_num,replace=False)]
        elif len(index) > 5:#点数小于指定数量，大于5，就扩充一下
            index = index[np.random.choice(len(index),region_num,replace=True)]

        #从len(gmask)个抓取中，继续筛选出点数足够多的抓取，len(gripper_mask)<=len(gmask)
        if len(index) > 5:##这里他们仅仅设置内部的点数大于5，说明回归出的抓取，内部的点数太少了
            gripper_pc[i] = torch.cat((pcs_t[i,index], group_points[i,index,3:]),-1)
            #gripper_pc_formal[i] = group_points[i,index,:]
            gripper_pc_index[i]       = index
            gripper_pc_index_inall[i] = group_index[i][index]
            gripper_mask[i] = 1
    
    if cuda:
        gripper_pc, gripper_mask, gripper_pc_index, gripper_pc_index_inall = gripper_pc.cuda(), gripper_mask.cuda(), \
                                                        gripper_pc_index.cuda(), gripper_pc_index_inall.cuda()#, gripper_pc_formal.cuda()
    gripper_mask_index = torch.nonzero(gripper_mask==1).view(-1)
    '''gripper_pc : [len(gmask),region_num,6]  len(gmask)个抓取，夹爪闭合区域点的xyzrgb
    gripper_pc_index: [len(gmask),region_num]  len(gmask)个抓取，夹爪闭合区域点相对于包围球内部点的索引
    gripper_pc_index_inall:[len(gmask),region_num]  len(gmask)个抓取，夹爪闭合区域点相对于完整pc的索引
    gripper_mask_index: [len(gripper_mask_index),] 内部点数大于指定数量的预测抓取的索引
    '''
    return gripper_pc, gripper_pc_index, gripper_pc_index_inall, gripper_mask_index#, gripper_pc_formal

def _enumerate_templates():
    '''枚举anchors的姿态，每个抓取点对应M个锚姿态
      (仅仅是姿态，没有位置)
      Enumerate all grasp anchors:
      For one score center, we generate M anchors.

      grasp configuration:(p, r, theta)
      r -> (1,0,0),                   (sqrt(2)/2, 0, sqrt(2)/2),           (sqrt(2)/2, 0, -sqrt(2)/2),           \
           (sqrt(2)/2,sqrt(2)/2,0),   (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),   (sqrt(3)/3, sqrt(3)/3, -sqrt(3)/3),   \
           (0,1,0),                   (0, sqrt(2)/2, sqrt(2)/2),           (0, sqrt(2)/2, -sqrt(2)/2),           \
           (-sqrt(2)/2,sqrt(2)/2,0),  (-sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),  (-sqrt(3)/3, sqrt(3)/3, -sqrt(3)/3),  \
           (-1,0,0),                  (-sqrt(2)/2, 0, sqrt(2)/2),          (-sqrt(2)/2, 0, -sqrt(2)/2),          \
           (-sqrt(2)/2,-sqrt(2)/2,0), (-sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3), (-sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3), \
           (0,-1,0),                  (0, -sqrt(2)/2, sqrt(2)/2),          (0, -sqrt(2)/2, -sqrt(2)/2),          \
           (sqrt(2)/2,-sqrt(2)/2,0),  (sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3),  (sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3)
      theta -> {-pi/2, -pi/4, 0, pi/4, pi/2}
    '''
    sqrt2 = math.sqrt(2)/2
    sqrt3 = math.sqrt(3)/3

    t_r = torch.FloatTensor([[sqrt3, sqrt3, sqrt3],[sqrt3, sqrt3, -sqrt3],
                                                    [sqrt3, -sqrt3, -sqrt3], [sqrt3, -sqrt3, sqrt3]]).view(1,4,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)
    #t_r = torch.FloatTensor([
    #                    [sqrt3, sqrt3, sqrt3], [sqrt3, sqrt3, -sqrt3], \
    #                    [-sqrt3, sqrt3, -sqrt3], [-sqrt3, sqrt3, sqrt3], \
    #                    [-sqrt3, -sqrt3, sqrt3], [-sqrt3,-sqrt3, -sqrt3], \
    #                    [sqrt3, -sqrt3, -sqrt3], [sqrt3,-sqrt3, sqrt3]\
    #                    ]).view(1,8,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)

    #t_r = torch.FloatTensor([#[1.0,0,0], [-1.0,0,0], [0,1.0,0], [0,-1.0,0],
    #                    [sqrt2, sqrt2, 0], [sqrt2, -sqrt2, 0]
    #                    ]).view(1,2,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)
    #t_theta = torch.FloatTensor([-math.pi/4, 0, math.pi/4]).view(1,1,3,1).repeat(1,8,1,1)
    t_theta = torch.FloatTensor([0]).view(1,1,1,1).repeat(1,4,1,1)#角度的anchors全都设置为0
    tem = torch.cat([t_r, t_theta], dim=3).half()
    return tem

def compute_cos_sim(a, b):
    '''计算向量a，b夹角的cos值
      input:
         a :[N, 3]
         b :[N, 3]
      output:
         sim :[N, 1]
    '''
    a_b = torch.sum(torch.mul(a, b), dim=1)#
    epsilon = 1e-12
    a2 = torch.add(torch.sum(torch.mul(a, a), dim=1), (epsilon))
    b2 = torch.add(torch.sum(torch.mul(b, b), dim=1), (epsilon))
    div_ab = torch.sqrt(torch.mul(a2, b2))
    sim = torch.div(a_b, div_ab).mul_(-1).add_(1).view(-1,1)
    '''
    b_copy = -b
    a_b_copy = torch.sum(torch.mul(a, b_copy), dim=1)
    sim_copy = torch.div(a_b_copy, div_ab).mul_(-1).add_(1).view(-1,1)

    sim = torch.min(sim, sim_copy)
    '''
    return sim
    

if __name__ == '__main__':
    pass