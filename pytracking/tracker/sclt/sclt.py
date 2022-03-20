from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation


import sys
import pandas as pd
from pytracking.evaluation.bbox import xywh2xyxy, xyxy2xywh, cxywh2xyxy, cxywh2xywh, xywh2xyxy, xyxy2cxywh,xywh2cxywh
from pytracking.features.resnet import resnet34
import os
import cv2
import pytracking.global_tracking._init_paths
import neuron.data as data
from global_tracker import *
import pytracking.utils._init_paths
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *
from PIL import Image
from math import sqrt
import torch
import torch.nn.functional as F
import numpy as np
from pytracking.evaluation.environment import env_settings

def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class SCLT(BaseTracker):
    multiobj_mode = 'parallel'
    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def process_regions(self,regions):
        regions = regions / 255.0
        regions[:, :, :, 0] = (regions[:, :, :, 0] - 0.485) / 0.229
        regions[:, :, :, 1] = (regions[:, :, :, 1] - 0.456) / 0.224
        regions[:, :, :, 2] = (regions[:, :, :, 2] - 0.406) / 0.225
        regions = np.transpose(regions, (0, 3, 1, 2))
        return regions
    def metric_init(self, im, init_box):
        self.seq_similar_array=[]
        self.bad_similar_mean=10
        self.similar_ver_num=0
        self.metric_model = ft_net(class_num=1120)
        metric_path=r'{}/{}'.format(env_settings().network_path,self.params.metric_path)
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(metric_path))
        temp_s = np.random.rand(1, 3, self.params.metric_size, self.params.metric_size)
        temp_s = (Variable(torch.Tensor(temp_s))).type(torch.FloatTensor).cuda()
        self.metric_model(temp_s)
        init_box = init_box.reshape((1, 4))
        anchor_region = me_extract_regions(im, init_box)
        anchor_region = self.process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        self.anchor_feature, _ = self.metric_model(anchor_region)

    def metric_eval(self, im, boxes, anchor_feature):
        box_regions = me_extract_regions(np.array(im), boxes)
        box_regions = self.process_regions(box_regions)
        box_regions = torch.Tensor(box_regions)
        box_regions = (Variable(box_regions)).type(torch.FloatTensor).cuda()
        box_features, class_result = self.metric_model(box_regions)
        ap_dist = torch.norm(anchor_feature - box_features, 2, dim=1).view(-1)
        return ap_dist
    def calc_similar(self,state):
        local_state = np.array(state).reshape((1, 4))
        try:
            ap_dis = self.metric_eval(self.current_img, local_state, self.anchor_feature)
        except:
            ap_dis = torch.Tensor([10] * self.anchor_feature.size()[0]).float()
        return ap_dis.data.cpu().numpy()[0]

    def calc_last_good_bbox(self,state):
        temp_bbox1=xywh2cxywh(state)
        last_good_bbox=xywh2cxywh(self.good_last_bbox)
        temp_dist=sqrt(abs(temp_bbox1[0]-last_good_bbox[0])*abs(temp_bbox1[1]-last_good_bbox[1]))/sqrt(last_good_bbox[2]*last_good_bbox[3])
        return temp_dist

    def transforms_response_map(self,response_map):
        score_in = cv2.resize(response_map, (self.params.score_map_input_size, self.params.score_map_input_size))
        score_in = cv2.cvtColor(np.asarray(score_in), cv2.COLOR_RGB2BGR)
        score_in -= score_in.min()
        score_in /= score_in.max()
        score_in = (score_in * 255).astype(np.uint8)
        score_in = cv2.applyColorMap(score_in, 2)
        score_img_in = Image.fromarray(cv2.cvtColor(score_in, cv2.COLOR_BGR2RGB))
        score_img_in = self.params.data_transform(score_img_in)
        score_img_in = torch.unsqueeze(score_img_in, dim=0)
        return score_img_in
    def global_search_init(self, image, init_box):
        self.global_num=self.params.global_search_num
        self.explore_num=self.params.global_search_top_num
        self.global_count=0
        self.global_bound_count=0
        self.global_other_count=0
        self.img_w=image.shape[1]
        self.img_h=image.shape[0]
        self.bound_global_flag=False
        self.good_last_bbox=[None,None,None,None]
        init_box = [init_box[0], init_box[1], init_box[0]+init_box[2], init_box[1]+init_box[3]]
        cfg_file = r'{}/{}'.format(env_settings().sclt_path,'global_tracking/configs/qg_rcnn_r50_fpn.py')
        transforms = data.BasicPairTransforms(train=False)
        self.Global_Tracker = GlobalTrack(
            cfg_file, r'{}/{}'.format(env_settings().network_path,self.params.global_track_pth_dir), transforms,
            name_suffix='qg_rcnn_r50_fpn')
        self.Global_Tracker.init(image, init_box)
    def calc_pre_dist(self):
        pre_box=xywh2cxywh(self.last_positive_bbox)
        current_box=xywh2cxywh(self.all_pre_box[-1])
        pre_box=np.array(pre_box)
        current_box=np.array(current_box)
        return math.sqrt(sum(pre_box[:2]-current_box[:2])**2)
    def Global_Track_eval(self, image, num):
        results = self.Global_Tracker.update(image)
        index = np.argsort(results[:, -1])[::-1]
        max_index = index[:num]
        can_boxes = results[max_index][:, :4]
        can_boxes = np.array([can_boxes[:, 0], can_boxes[:, 1], can_boxes[:, 2]-can_boxes[:, 0], can_boxes[:, 3]-can_boxes[:, 1]]).transpose()
        return can_boxes
    def verification_lost(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.tracking_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==False)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==False)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False

    def verification_lost2(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.tracking_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==False)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==False)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False
    def verification_lost_global(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.all_global_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==False)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==False)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False
    def verification_lost_global2(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.tracking_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==False)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==False)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False
    def ver_update_tracker(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.tracking_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==True)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==True)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False
    def ver_model_update_tracker(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.model_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==True)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==True)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False
    def ver_model_lost_tracker(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.model_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==False)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==False)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False

    def ver_update_tracker2(self,num,update_evaluate_rate=0.9):
        temp_update=np.array(self.tracking_update_array,dtype=bool)
        num=int(num)
        if len(temp_update)>num:
            if sum(temp_update[-num:]==True)>num*update_evaluate_rate:
                return True
            else:
                return False
        else:
            if sum(temp_update[-num:]==True)>len(temp_update)*update_evaluate_rate:
                return True
            else:
                return False
    def find_similar_update(self,num,update_evaluate_rate=0.9,frame_num=-1):
        temp_update=np.array(self.sclt_update_flag[frame_num-num:frame_num],dtype=bool)
        num=int(num)
        if sum(temp_update[-num:]==True)>=num*update_evaluate_rate:
            return True
        else:
            return False
    def calc_all_param(self):
        temp_all_similar_array = np.array(self.seq_similar_array)
        temp_all_update_sclt_flag = np.array(self.sclt_update_flag, dtype=bool)
        temp_all_pre_bbox_w = np.array(self.sclt_pre_w_array[-self.params.state_global_step_num:])
        temp_all_pre_bbox_h = np.array(self.sclt_pre_h_array[-self.params.state_global_step_num:])
        self.pos_similarity_mean = temp_all_similar_array.mean()
        self.pos_similarity_mean_5 = np.array(
            temp_all_similar_array[np.argwhere(temp_all_update_sclt_flag == True)]).mean()
        self.pos_similarity_mean_2 = (self.pos_similarity_mean + self.pos_similarity_mean_5) / 2
        self.temp_w_mean = temp_all_pre_bbox_w.mean()
        self.temp_w_bound_mean = temp_all_pre_bbox_w.max()
        self.temp_h_mean = temp_all_pre_bbox_h.mean()
        self.temp_h_bound_mean = temp_all_pre_bbox_h.max()
    def target_previous_close_bounding(self,bbox):
        if self.temp_w_mean!=0 and self.temp_h_mean!=0 and self.img_w!=0 and self.img_h!=0:
            cqw = self.temp_w_mean / self.params.frame_size_thre_rate / self.img_w
            cqh = self.temp_h_mean / self.params.frame_size_thre_rate / self.img_h
        flag = False
        temp_box = np.array(bbox)
        if temp_box[2] < 5 or temp_box[3] < 5:
            flag=True
        elif any(pd.isna(temp_box)):
            flag=True
        else:
            temp_box_0=temp_box
            temp_box = xywh2xyxy(temp_box)
            ctemp_box = xywh2cxywh(temp_box_0)
            ctemp_x1 = ctemp_box[0] / self.img_w
            ctemp_y1 = ctemp_box[1] / self.img_h
            if (ctemp_x1<cqw or ctemp_x1>1-cqw or ctemp_y1<cqh or ctemp_y1>1-cqh):
                flag = True
        return flag
    def target_close_bounding(self,bbox):
        if self.temp_w_mean!=0 and self.temp_h_mean!=0 and self.img_w!=0 and self.img_h!=0:
            if self.img_w<self.params.frame_size_thre or self.img_h<self.params.frame_size_thre:
                q = (sqrt(self.temp_w_mean * self.temp_h_mean) / self.params.frame_size_thre_rate_out_view) / sqrt(self.img_w * self.img_h)
                cqw = self.temp_w_mean / self.params.frame_size_thre_rate_out_view / self.img_w
                cqh = self.temp_h_mean / self.params.frame_size_thre_rate_out_view / self.img_h
            else:
                q=(sqrt(self.temp_w_mean*self.temp_h_mean)/ (self.params.frame_size_thre_rate_out_view/2))/sqrt(self.img_w*self.img_h)
                cqw=self.temp_w_mean/ (self.params.frame_size_thre_rate_out_view/2)/self.img_w
                cqh=self.temp_h_mean/ (self.params.frame_size_thre_rate_out_view/2)/self.img_h
        else:
            q = self.params.target_to_frame_bounding_rate
            cqw = self.params.target_to_frame_bounding_rate
            cqh = self.params.target_to_frame_bounding_rate
        flag = False
        temp_box = np.array(bbox)
        if temp_box[2] < 5 or temp_box[3] < 5:
            flag=True
        elif any(pd.isna(temp_box)):
            flag=True
        else:
            temp_box_0=temp_box
            temp_box = xywh2xyxy(temp_box)
            ctemp_box = xywh2cxywh(temp_box_0)
            temp_x1 = temp_box[0] / self.img_w
            temp_y1 = temp_box[1] / self.img_h
            temp_x2 = temp_box[2] / self.img_w
            temp_y2 = temp_box[3] / self.img_h
            ctemp_x1 = ctemp_box[0] / self.img_w
            ctemp_y1 = ctemp_box[1] / self.img_h
            new_temp = [temp_x1, temp_y1, temp_x2, temp_y2]
            new_temp = np.array(new_temp)
            if (any(new_temp > (1 - q)) or any(new_temp < q)) and (ctemp_x1<cqw or ctemp_x1>1-cqw or ctemp_y1<cqh or ctemp_y1>1-cqh):
                flag = True
        return flag

    def target_close_bounding_agin(self,bbox):
        q_w = self.temp_w_bound_mean*1.2 / self.img_w
        q_h = self.temp_h_bound_mean*1.2 / self.img_h
        flag = False
        temp_box = bbox
        temp_box = xywh2cxywh(temp_box)
        temp_x1 = temp_box[0] / self.img_w
        temp_y1 = temp_box[1] / self.img_h
        new_temp = [temp_x1, temp_y1]
        new_temp = np.array(new_temp)
        if new_temp[0] > (1 - q_w) or new_temp[0] < q_w or new_temp[1] > (1 - q_h) or new_temp[1] < q_h:
            flag = True
        return flag
    def init_remodel(self):
        self.init_num=0
        self.all_response_map = []
        self.all_global_flag=[]
        self.sclt_update_flag=[]
        self.sclt_update_score=[]
        self.sclt_pre_w_array=[]
        self.sclt_pre_h_array=[]
        self.all_pre_box=[]
        self.stre_result_array=[]
        self.superdimp_update_array=[]
        self.stre_result_array.append(True)
        self.superdimp_update_array.append(True)
        self.use_post_flag=False
        self.similar_flag_array=[]
        self.similar_flag3_array=[]
        self.similar_flag4_array=[]
        self.update_grade=1
        self.sclt_update_grade=2
        self.model_update_similar_array=[]
        self.model_update_similar_array.append(1)
        self.global_track_flag=False
        self.global_track_flag_array=[]
        self.bound_global_track_flag=False
        self.bound_global_track_flag_array=[]
        self.global_track_flag_array.append(self.global_track_flag)
        self.bound_global_track_flag_array.append(self.bound_global_track_flag)
        self.tracking_update_array=[]
        self.model_update_array=[]
        self.model_update_array.append(True)
        self.similar_flag=True
        self.similar_flag3=True
        self.similar_flag4=True
        self.similar_flag3_array.append(self.similar_flag3)
        self.similar_flag4_array.append(self.similar_flag4)
        self.superdimp_update=True
        self.tracking_update_array.append(True)
        self.all_update_array=[]
        self.all_global_update_array=[]
        self.all_pre_dist=[]
        self.menory_dist_array=[]
        self.seq_stable_similar_array=[]
        self.seq_stable_similar_array.append(1)
        self.stable_response_map = []
        self.stable_response_map = np.array([1])
        self.pos_similarity_mean=self.params.similarity_super_thre
        self.re_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.re_model = resnet34(num_classes=self.params.stre_type_classes_num)
        self.re_model.load_state_dict(torch.load(r'{}/{}'.format(env_settings().network_path,self.params.stre_model), map_location='cuda:0'))
        self.re_model.to(self.re_device)
        self.re_model.eval()
    def advance_score(self,scores):
        if ((self.down_similar_array(self.params.advance_len,self.params.advance_rate)  and self.ver_model_update_tracker(self.params.advance_len,self.params.advance_update_model_rate) )or (self.ver_model_lost_tracker(self.params.advance_len,self.params.advance_update_model_rate))
            or (self.ver_lost_array(self.superdimp_update_array,self.params.advance_len/2,self.params.advance_rate) and self.ver_lost_array(self.stre_result_array,self.params.advance_len/2,self.params.advance_rate))) and self.update_grade>1:
            self.update_grade-=1
        out_flag=False
        temp_response_map_score = 0
        temp_scores_max=scores.max()
        if temp_scores_max<=0:
            temp_scores_max=0.0001
        if self.frame_num > self.params.stre_begin_frame_num2:
            temp_response_map_score = np.mean(self.stable_response_map) / temp_scores_max
        if self.update_grade==1:
            if temp_response_map_score <= self.params.update_grade1:
                if temp_response_map_score <= self.params.update_grade1 - self.params.menory_reduce:
                    self.stable_response_map = np.append(self.stable_response_map, temp_scores_max)
                out_flag=True
            elif self.similar_flag3 and self.ver_model_update_tracker(self.params.advance_len/2,self.params.advance_rate):
                self.update_grade=2
        if self.update_grade==2 :
            if temp_response_map_score <= self.params.update_grade2:
                if temp_response_map_score <= self.params.update_grade2 - self.params.menory_reduce:
                    self.stable_response_map = np.append(self.stable_response_map, temp_scores_max)
                out_flag=True
            elif self.similar_flag3 and self.ver_model_update_tracker(self.params.advance_len/2,self.params.advance_rate):
                self.update_grade=3
        if self.update_grade==3:
            if temp_response_map_score <= self.params.update_grade3:
                if temp_response_map_score <= self.params.update_grade3 - self.params.menory_reduce:
                    self.stable_response_map = np.append(self.stable_response_map, temp_scores_max)
                out_flag=True
        if self.stable_response_map.size > self.params.memory_size:
            self.stable_response_map = np.delete(self.stable_response_map, 0)
        return out_flag
    def down_similar_array(self,num,rate=0.9):
        num=int(num)
        temp_seq_stable_similar_array=np.array(self.seq_stable_similar_array)
        if len(temp_seq_stable_similar_array)>num:
            if sum(temp_seq_stable_similar_array[-num:]<self.seq_stable_similar_array_value3)>=num*rate:
                return True
            else:
                return False
        else:
            if sum(temp_seq_stable_similar_array[-num:] < self.seq_stable_similar_array_value3) >= len(temp_seq_stable_similar_array) * rate:
                return True
            else:
                return False

    def advance_score2(self,scores):
        if ((self.down_similar_array(self.params.advance_len,
                                     self.params.advance_rate) and self.ver_model_update_tracker(
                self.params.advance_len, self.params.advance_update_model_rate)) or (
            self.ver_model_lost_tracker(self.params.advance_len, self.params.advance_update_model_rate))
            or (self.ver_lost_array(self.superdimp_update_array, self.params.advance_len / 2,
                                    self.params.advance_rate) and self.ver_lost_array(self.stre_result_array,
                                                                                      self.params.advance_len / 2,
                                                                                      self.params.advance_rate))) and self.sclt_update_grade > 1:
            self.sclt_update_grade-=1
        out_flag = False
        temp_response_map_score = 0
        temp_scores_max = scores.max()
        if temp_scores_max <= 0:
            temp_scores_max = 0.0001
        if self.frame_num > self.params.stre_begin_frame_num2:
            temp_response_map_score = np.mean(self.stable_response_map) / temp_scores_max
        if self.sclt_update_grade == 1:
            if temp_response_map_score <= self.params.sclt_update_grade1:
                if temp_response_map_score <= self.params.sclt_update_grade1 - self.params.menory_reduce:
                    self.stable_response_map = np.append(self.stable_response_map, temp_scores_max)
                out_flag = True
            elif self.similar_flag3 and self.ver_model_update_tracker(self.params.advance_len / 2,self.params.advance_rate):
                self.sclt_update_grade = 2
        if self.sclt_update_grade == 2:
            if temp_response_map_score <= self.params.sclt_update_grade2:
                if temp_response_map_score <= self.params.sclt_update_grade2 - self.params.menory_reduce:
                    self.stable_response_map = np.append(self.stable_response_map, temp_scores_max)
                out_flag = True
            elif self.similar_flag3 and self.ver_model_update_tracker(self.params.advance_len / 2,self.params.advance_rate):
                self.sclt_update_grade = 3
        if self.sclt_update_grade == 3:
            if temp_response_map_score <= self.params.sclt_update_grade3:
                if temp_response_map_score <= self.params.sclt_update_grade3 - self.params.menory_reduce:
                    self.stable_response_map = np.append(self.stable_response_map, temp_scores_max)
                out_flag = True

        if self.stable_response_map.size > self.params.memory_size:
            self.stable_response_map = np.delete(self.stable_response_map, 0)
        return out_flag
    def menory_add_dist(self):
        miss_frame = 1
        if len(self.all_pre_dist) < self.params.menory_smallest_len:
            current_bbox = self.all_pre_box[-1]
            pos_bbox = self.all_pre_box[-2]
            current_bbox = xywh2cxywh(current_bbox)
            pos_bbox = xywh2cxywh(pos_bbox)
            self.menory_dist_array.append(abs((current_bbox[0] - pos_bbox[0]) * (current_bbox[1] - pos_bbox[1])))
            return True, miss_frame
        else:
            mean_dist = np.mean(self.menory_dist_array)
            current_dist = self.all_pre_dist[-1]
            dist_flag = current_dist / mean_dist < self.params.dist_menory_num
            if dist_flag:
                self.menory_dist_array.append(self.all_pre_dist[-1])
                if len(self.menory_dist_array) > self.params.advance_len:
                    np.delete(self.menory_dist_array, 0)
            else:
                miss_frame = int(current_dist / mean_dist)
            return dist_flag, miss_frame

    def calc_pre_dist(self):
        pre_box = xywh2cxywh(self.all_pre_box[-2])
        current_box = xywh2cxywh(self.all_pre_box[-1])
        return abs((pre_box[0] - current_box[0]) * (pre_box[1] - current_box[1]))
    def ver_update_array(self,array,num,rate=0.9):
        array=np.array(array)
        num = int(num)
        if len(array)>num:
            if sum(array[-num:]==True)>=num*rate:
                return True
            else:
                return False
        else:
            if sum(array[-num:] == True) >= len(array) * rate:
                return True
            else:
                return False

    def ver_lost_array(self,array,num,rate=0.9):
        array=np.array(array)
        num=int(num)
        if len(array)>num:
            if sum(array[-num:]==False)>=num*rate:
                return True
            else:
                return False
        else:
            if sum(array[-num:] == False) >= len(array) * rate:
                return True
            else:
                return False

    def stre_module(self,response_map):
        score_map=response_map
        current_max_score = max(score_map.flatten())
        self.max_score_map_confidence_array.append(current_max_score)
        self.all_response_map.append(score_map)
        self.all_max_map_array.append(score_map.max())
        bbox = self.all_pre_box[-1]
        state = bbox
        superdimp_update_flag = self.flag not in ['not_found', 'uncertain']
        self.superdimp_update=superdimp_update_flag
        temp_similar_value = self.calc_similar(state)
        self.seq_similar_array.append(temp_similar_value)
        self.seq_stable_similar_array_value = np.mean(self.seq_stable_similar_array)
        self.seq_stable_max_mean_similar_array_value =( np.mean(self.seq_stable_similar_array) + np.max(self.seq_stable_similar_array))/2
        self.seq_stable_similar_array_value2 = np.mean(self.seq_stable_similar_array)
        # if self.model_update:
        self.seq_stable_similar_array_value3 = self.seq_stable_max_mean_similar_array_value
        self.seq_stable_similar_array_value4 = self.seq_stable_max_mean_similar_array_value
        self.seq_stable_similar_array_value4 = (self.seq_stable_similar_array_value4+self.seq_stable_similar_array_value2)/2
        self.seq_stable_similar_array_value4 = (self.seq_stable_max_mean_similar_array_value+self.seq_stable_similar_array_value4)/2
        self.seq_stable_similar_array_value3 = (np.mean(
            self.seq_stable_similar_array) + self.seq_stable_similar_array_value3) / 2
        self.seq_stable_similar_array_value3 = (np.mean(
            self.seq_stable_similar_array) + self.seq_stable_similar_array_value3) / 2
        self.similar_flag = self.seq_similar_array[-1] < self.seq_stable_similar_array_value2
        self.similar_flag3 = self.seq_similar_array[-1] < self.seq_stable_similar_array_value3
        self.similar_flag4 = self.seq_similar_array[-1] < self.seq_stable_similar_array_value4
        self.similar_flag_array.append(self.similar_flag)
        self.similar_flag3_array.append(self.similar_flag3)
        self.similar_flag4_array.append(self.similar_flag4)
        self.stre_result=True
        if self.frame_num > self.params.stre_begin_frame_num:
            score_img_in=self.transforms_response_map(score_map)
            with torch.no_grad():  # predict class
                output_map = self.re_model(score_img_in.to(self.re_device))
                predict_y = torch.max(output_map, dim=1)[1]
                y_label = predict_y.data.cpu().numpy()[0]
                if y_label == 1:
                    self.stre_result = True
                else:
                    self.stre_result = False
                if (self.stre_result or self.superdimp_update) and self.ver_model_update_tracker(self.params.model_update_num,self.params.model_update_rate)==False:
                    update = self.advance_score2(score_map)
                else:
                    update = self.advance_score(score_map)
                self.all_update_array.append(update)
        else:
            update = superdimp_update_flag
            self.stre_result = update
        self.stre_result_array.append(self.stre_result)
        self.superdimp_update_array.append(superdimp_update_flag)
        self.tracking_update_array.append(update)
        self.update=update
        self.sclt_update_flag.append(update)
        self.all_curren_flag_array.append(update)
        if self.ver_update_tracker(self.params.state_step_num, self.params.state_step_rate):
            self.good_last_bbox = np.array(state)
        if self.ver_update_tracker(self.params.global_step_nm, self.params.global_step_rate):
            self.global_count = 0
        if (self.update == False or self.model_update == False) :
            if self.similar_ver_num % self.params.global_step_num2 == 0:
                if self.verification_lost_global2(self.params.global_search_step_num,
                                                  self.params.global_ver_update_tracker_rate):
                    self.tracking_update_array.append(temp_similar_value < self.seq_stable_similar_array_value4)
                else:
                    self.tracking_update_array.append(temp_similar_value < self.seq_stable_similar_array_value3)
                self.all_global_update_array.append(temp_similar_value < self.seq_stable_similar_array_value3)
            self.similar_ver_num += 1
        temp_all_dist=np.array(self.all_pre_dist)
        if sum(temp_all_dist[-self.params.global_step_num2:]<0.1)==self.params.global_step_num2 and update==False:
            self.all_global_update_array.append(False)
        if self.update:
            self.similar_ver_num=0
        if self.model_update:
            self.similar_ver_num=0

        if self.frame_num > self.params.stre_begin_frame_num+1:
            temp_bound_update_flag2 = True
            if self.target_close_bounding(state):
                temp_bound_update_flag2 = False
            if self.ver_model_lost_tracker(self.params.stable_len,self.params.ver_update_tracker_rate) and \
                    self.ver_lost_array(self.similar_flag3_array,self.params.model_update_num,self.params.ver_update_tracker_rate) or \
                    self.ver_update_array(self.global_track_flag_array,self.params.model_update_num,self.params.ver_update_tracker_rate) or \
                    self.ver_update_array(self.bound_global_track_flag_array,self.params.model_update_num,self.params.ver_update_tracker_rate):
                update_type=2
                temp_update = self.ver_update_tracker(self.temp_state_num,
                                                      # temp_update = self.ver_update_tracker2(self.temp_state_num,
                                                      self.params.ver_update_tracker_rate) and temp_bound_update_flag2
            else:
                update_type = 1
                temp_update = update
        else:
            update_type=1
            temp_update = superdimp_update_flag
        self.model_update_array.append(temp_update)
        if update:
            self.seq_stable_similar_array.append(temp_similar_value)
        if len(self.seq_stable_similar_array)>self.params.menory_similar_len:
            np.delete(self.seq_stable_similar_array,0)
        self.all_global_update_array.append(temp_update)
        self.init_num+=1
        global_track_flag = False
        bound_global_track_flag = False
        if self.frame_num > self.params.stre_begin_frame_num:
            if self.verification_lost_global(self.params.global_search_step_num, self.params.global_ver_update_tracker_rate):
                if self.global_track_flag  or self.bound_global_track_flag:
                    if self.verification_lost_global2(self.params.global_search_step_num,
                                                      self.params.global_ver_update_tracker_rate):
                        if self.target_previous_close_bounding(state):
                            bound_global_track_flag = True
                        else:
                            global_track_flag = True
                else:
                    if self.target_previous_close_bounding(state):
                        bound_global_track_flag = True
                    else:
                        global_track_flag = True
        return temp_update,global_track_flag,bound_global_track_flag
    def sppp_module(self,global_track_flag,bound_global_track_flag):
        candidate_bboxes = None
        bound_temp_flag = False
        center_temp_flag = False
        center_in_num=self.params.search_step_num
        if self.verification_lost_global(self.params.lost_global_num,self.params.lost_global_rate):
            center_in_num=self.params.search_step_num2
        if bound_global_track_flag and self.global_bound_count % self.params.global_step_num2 == 0:
            temp_current_img=self.current_img.copy()
            candidate_bboxes = self.Global_Track_eval(temp_current_img, self.global_num)
            self.out['global_track']=candidate_bboxes
            temp_cand_num = [i for i in range(self.global_num)]
            temp_cand_num = np.array(temp_cand_num)
            self.global_bound_count = 0
            bound_temp_flag = True
        if global_track_flag and self.global_other_count % center_in_num == 0 and (not bound_global_track_flag):
            candidate_bboxes = self.Global_Track_eval(self.current_img, self.global_num)
            self.out['global_track'] = candidate_bboxes
            temp_cand_num = [i for i in range(self.global_num)]
            temp_cand_num = np.array(temp_cand_num)
            center_temp_flag = True
            self.global_other_count = 0
        if bound_global_track_flag:
            self.global_bound_count += 1
        if global_track_flag and (not bound_global_track_flag):
            self.global_other_count += 1
        if bound_temp_flag:
            self.bound_global_flag = True
            self.global_count = 0
            bound_result = []
            self.seq_travel_num_array = []
            self.seq_travel_array = []
            for temp_i in range(len(candidate_bboxes)):
                bound_result.append(self.calc_similar(candidate_bboxes[temp_cand_num[temp_i]]))
            for t_i in range(self.global_num):
                bound_result = np.array(bound_result)
                temp_id = np.argmin(bound_result)
                current_temp_similar = bound_result[temp_id]
                temp_bound = False
                if self.frame_num > self.params.stre_begin_frame_num:
                    if self.target_close_bounding_agin(candidate_bboxes[temp_cand_num[temp_id]]) and (
                            current_temp_similar < self.seq_stable_similar_array_value4):
                        temp_bound = True
                    else:
                        self.global_count = 0
                if temp_bound:
                    redet_bboxes = candidate_bboxes[temp_cand_num[temp_id]]
                    self.last_tar_pos = np.array(
                        [redet_bboxes[1], redet_bboxes[0], redet_bboxes[1] + redet_bboxes[3],
                         redet_bboxes[2] + redet_bboxes[0]])
                    self.pos = torch.FloatTensor(
                        [(self.last_tar_pos[0] + self.last_tar_pos[2] - 1) / 2,
                         (self.last_tar_pos[1] + self.last_tar_pos[3] - 1) / 2])
                    self.target_sz = torch.FloatTensor(
                        [(self.last_tar_pos[2] - self.last_tar_pos[0]), (self.last_tar_pos[3] - self.last_tar_pos[1])])
                    break
                else:
                    bound_result = np.delete(bound_result, temp_id)  # temp_cand_num[temp_id]
                    temp_cand_num = np.delete(temp_cand_num, temp_id)
        if center_temp_flag:
            bound_result = []
            self.seq_travel_num_array = []
            self.seq_travel_array = []
            temp_center_similar_array = []
            for temp_i in range(len(candidate_bboxes)):
                if not any(pd.isna(self.good_last_bbox)) and sum(np.array(self.all_global_flag[-50:])==2)<30 and self.verification_lost(200,0.7)==False:
                    temp_current_similar_bbox = np.array(
                        self.calc_similar(candidate_bboxes[temp_cand_num[temp_i]]))
                    temp_center_similar_array.append(temp_current_similar_bbox)
                    temp_current_dist_bbox = np.array(
                        self.calc_last_good_bbox(candidate_bboxes[temp_cand_num[temp_i]]))
                    temp_current_dist_bbox /= temp_current_dist_bbox.min()
                    temp_current_dist_bbox *= temp_current_similar_bbox.min()
                    bound_result.append(temp_current_similar_bbox + temp_current_dist_bbox)
                else:
                    temp_center_current_similar = self.calc_similar(candidate_bboxes[temp_cand_num[temp_i]])
                    temp_center_similar_array.append(temp_center_current_similar)
                    bound_result.append(temp_center_current_similar)
            for t_i in range(self.explore_num):
                bound_result = np.array(bound_result)
                temp_cand_num=np.array(temp_cand_num)
                temp_id = np.argmin(bound_result)
                current_temp_similar = temp_center_similar_array[temp_cand_num[temp_id]]
                temp_bound = False
                if self.frame_num > self.params.stre_begin_frame_num:
                    if self.verification_lost_global2(self.params.global_search_step_num2,
                                                   self.params.global_ver_update_tracker_rate):
                        if (current_temp_similar <= self.seq_stable_similar_array_value4):
                            temp_bound = True
                    else:
                        if self.use_post_flag:
                            if (current_temp_similar <= self.seq_stable_similar_array_value4):
                                temp_bound = True
                            elif (current_temp_similar <= self.seq_stable_similar_array_value3):
                                temp_bound = True
                if temp_bound:
                    self.global_count += 1
                    if self.global_count >= self.params.global_step_num:
                        redet_bboxes = candidate_bboxes[temp_cand_num[temp_id]]
                        self.last_tar_pos = np.array(
                            [redet_bboxes[1], redet_bboxes[0], redet_bboxes[1] + redet_bboxes[3],
                             redet_bboxes[2] + redet_bboxes[0]])
                        self.pos = torch.FloatTensor(
                            [(self.last_tar_pos[0] + self.last_tar_pos[2] - 1) / 2,
                             (self.last_tar_pos[1] + self.last_tar_pos[3] - 1) / 2])
                        self.target_sz = torch.FloatTensor(
                            [(self.last_tar_pos[2] - self.last_tar_pos[0]), (self.last_tar_pos[3] - self.last_tar_pos[1])])
                        self.global_count = 0
                        self.update_num=1
                    break
                else:
                    bound_result = np.delete(bound_result, temp_id)
                    temp_cand_num = np.delete(temp_cand_num, temp_id)

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        # Initialize network
        self.initialize_features()
        self.update_num=0
        self.init_remodel()
        self.metric_init(image, np.array(info['init_bbox']))
        self.global_search_init(image, np.array(info['init_bbox']))
        self.sclt_pre_w_array.append(info['init_bbox'][2])
        self.sclt_pre_h_array.append(info['init_bbox'][3])
        self.max_score_map_confidence_array = []
        self.all_max_map_array = []
        self.all_curren_flag_array = []
        self.temp_state_num = self.params.stable_len
        self.net = self.params.net
        # Time initialization
        tic = time.time()
        # Convert image
        im = numpy_to_torch(image)
        # Get target position and size
        state = info['init_bbox']
        self.all_pre_box.append(state)
        self.all_update_array.append(True)
        self.all_global_update_array.append(True)
        self.last_positive_bbox=state
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)
        self.update=True
        self.stre_result=True
        self.model_update=True
        out = {'time': time.time() - tic}
        return out



    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        # if self.model_update:
        if self.ver_model_update_tracker(20,0.8):
            self.last_pos = self.pos
            self.last_target_sz = self.target_sz
            self.use_post_flag=False


        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales, = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw = self.classify_target(test_x)
        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind,:] + translation_vec
        self.flag=flag
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])


        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None
        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()
            self.dist_flag_type = 1

        self.all_pre_box.append(output_state)
        self.all_pre_dist.append(self.calc_pre_dist())
        self.dist_flag, self.miss_num = self.menory_add_dist()
        self.update_flag = update_flag
        self.current_img = image
        out_score_map=np.array(score_map.cpu().data.numpy())
        self.calc_all_param()
        temp_update,global_track_flag,bound_global_track_flag,=self.stre_module(out_score_map)
        self.global_track_flag=global_track_flag
        self.bound_global_track_flag=bound_global_track_flag
        self.global_track_flag_array.append(self.global_track_flag)
        self.bound_global_track_flag_array.append(self.bound_global_track_flag)
        if bound_global_track_flag:
            self.all_global_flag.append(2)
        elif global_track_flag:
            self.all_global_flag.append(1)
        else:
            self.all_global_flag.append(0)
        self.model_update=temp_update
        self.update_num-=1
        if temp_update:
            train_x = test_x[scale_ind:scale_ind + 1, ...]
            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])
            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind, ...])
        self.out = {}
        if self.frame_num > self.params.stre_begin_frame_num:
            if global_track_flag or bound_global_track_flag:
                self.sppp_module(global_track_flag, bound_global_track_flag)
        self.out['target_bbox'] = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]])).tolist()
        out_score_map = np.array(score_map.cpu().data.numpy())
        self.out['confidence'] = out_score_map.max()
        self.out['object_presence_score'] = out_score_map.max()
        self.out['votlt_flag'] = self.params.votlt_flag
        if temp_update:
            self.sclt_pre_w_array.append(self.out['target_bbox'][2])
            self.sclt_pre_h_array.append(self.out['target_bbox'][3])
        return self.out


    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])


    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)


    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')