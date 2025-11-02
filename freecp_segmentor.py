import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import cv2
from PIL import Image

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from open_clip import tokenizer, create_model, gem

from prompts.base.imagenet_template import *
from prompts.vicuna13b import *

'''
numpy                    1.26.0
mmcv                     2.0.1
mmengine                 0.8.4
mmsegmentation           1.1.1
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-ml-py             12.570.86
torch                    1.13.1
torchvision              0.14.1

torch                  1.10.2+cu113
torchvision            0.11.3
rp_thres=0.1 ap_thres=0.7 bash ./dist_test.sh configs_ours/cfg_city_scapes.py
'''

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

@MODELS.register_module()
class FreeCPSegmentation(BaseSegmentor): 
    def __init__(self, clip_type, vit_type, model_type, name_path, dataset='voc', device=torch.device('cuda'),
                 slide_stride=112, slide_crop=448, with_bkg=False, ignore_bkg=False, rp_thres=0.15, ap_thres=0.7):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        
        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.name_path = name_path
        self.dataset = dataset
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.with_bkg = with_bkg
        self.ignore_bkg = ignore_bkg
        self.rp_thres = rp_thres
        self.ap_thres = ap_thres
        self.create_base_model(clip_type, vit_type, model_type, device)
        self.create_vocabulary(name_path, dataset, device)
    
    def create_base_model(self, clip_type, vit_type, model_type, device):
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')

        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', device=device, precision='fp16')
            self.net = self.net.model
        
        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize
        self.patch_size = self.net.visual.patch_size
        if not isinstance(self.patch_size, tuple):
            self.patch_size = (self.patch_size, self.patch_size)

    def create_vocabulary(self, name_path, dataset, device):
        query_words, self.query_idx, prompts = get_prompts(name_path, dataset, use_llm=False)
        _, _, promptsLLM = get_prompts(name_path, dataset, use_llm=True)
        
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.num_bkg_prompts = self.num_queries - self.num_classes + 1 if self.with_bkg and not self.ignore_bkg else 0
        
        query_features = []
        with torch.no_grad():
            for prompt in prompts:
                query = self.tokenizer(prompt).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        query_features = []
        with torch.no_grad():
            for prompt in promptsLLM:
                query = self.tokenizer(prompt).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_featuresLLM = torch.cat(query_features, dim=0)
        self.dtype = self.query_features.dtype

    def refinement(self, img, normalized_logits, attn_weight_list, cls_label, chunk_size = 1000):
        ###################################### Refine ##########################################
        _, c, w, h = normalized_logits.shape
        # attn_refinement
        attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
        attn_weight = torch.stack(attn_weight, dim=0)[:-1].mean(dim=0).float()[0]
        aff_mat = attn_weight
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        for _ in range(2):
            trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
        trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
        for _ in range(1):
            trans_mat = torch.matmul(trans_mat, trans_mat)
        
        logits_ref = torch.zeros(img.shape[0], self.num_queries, w, h).cuda()
        process_length = len(cls_label)
        aff_masks = torch.zeros(process_length, 1, w * h).cuda()
        for i in range(self.num_bkg_prompts, c):
            grayscale_cam = normalized_logits[0,i] 
            box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
            aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
            for i_ in range(cnt):
                x0_, y0_, x1_, y1_ = box[i_]
                aff_mask[y0_:y1_, x0_:x1_] = 1
            aff_masks[i-self.num_bkg_prompts] = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        def _affinity(cam_to_refine, trans_mat, aff_masks):
            ## trans_mat: [1, h*w, h*w], aff_mask: [c, 1, h*w] --->>> trans_mat: [c, h*w, h*w]
            trans_mat = trans_mat.unsqueeze(0) * aff_masks
            cam_to_refine = cam_to_refine.view(-1, h*w, 1)
            ## trans_mat: [c, h*w, h*w], cam_to_refine: [c, h*w, 1] --->>> cam_refined: [c, h*w, 1] --->>> cam_refined: [c, w, h]
            cam_refined = torch.matmul(trans_mat, cam_to_refine)
            cam_refined = cam_refined.reshape(-1, w, h)
            return cam_refined
        if process_length > chunk_size:
            num_split = process_length // chunk_size + 1
            cam_refined = torch.zeros_like(normalized_logits[0, self.num_bkg_prompts:c]).cuda().float()
            for _i in range(num_split):
                if _i == num_split - 1:
                    cam_to_refine = normalized_logits[0, self.num_bkg_prompts + chunk_size * _i : c].float()
                    cam_refined[chunk_size * _i : c] = _affinity(cam_to_refine, trans_mat, aff_masks[chunk_size * _i : c])
                else:
                    cam_to_refine = normalized_logits[0, self.num_bkg_prompts + chunk_size * _i : self.num_bkg_prompts + chunk_size * (_i + 1)].float()
                    cam_refined[chunk_size * _i : chunk_size * (_i + 1)] = _affinity(cam_to_refine, trans_mat, aff_masks[chunk_size * _i : chunk_size * (_i + 1)])
        else:
            cam_refined = _affinity(normalized_logits[0, self.num_bkg_prompts:c].float(), trans_mat, aff_masks)
        for i in range(process_length):
            cam_refined[i] = (cam_refined[i] - cam_refined[i].min()) / (cam_refined[i].max() - cam_refined[i].min() + 1e-8)
            logits_ref[0, self.num_bkg_prompts + cls_label[i]] = cam_refined[i]
            
        return logits_ref
    
    def redundancy_purification(self, logits, logits_ref):
        iou = (logits[0,self.num_bkg_prompts:] * logits_ref[0,self.num_bkg_prompts:]).sum((-1,-2)) / (logits[0,self.num_bkg_prompts:].sum((-1,-2)) + logits_ref[0,self.num_bkg_prompts:].sum((-1,-2)) + 1e-8)
        candidate_classes = torch.where(iou > self.rp_thres)[0]

        for i in range(self.num_bkg_prompts, logits_ref.shape[1]):
            if i - self.num_bkg_prompts not in candidate_classes:
                logits_ref[:,i] = logits_ref[:,i] * 0.0
        return logits_ref, candidate_classes
        
    def ambiguity_purification(self, logits, logits_ref, candidate_classes, img):
        postref = logits_ref[0][self.num_bkg_prompts + candidate_classes]
        preref = logits[0][self.num_bkg_prompts + candidate_classes]
        postref_expanded = postref.unsqueeze(1)
        # 计算交集和并集 
        intersection = torch.minimum(postref_expanded, postref)
        union = torch.maximum(postref_expanded, postref)

        # 计算 Soft IoU 矩阵
        soft_iou_matrix = torch.sum(intersection, (2, 3)) / torch.sum(union, (2, 3))
        soft_iou_matrix[soft_iou_matrix < self.ap_thres] = 0
        soft_iou_matrix[soft_iou_matrix >= self.ap_thres] = 1

        
        components = connected_components(soft_iou_matrix)
        if len(components) != 0:
            valid_classes = []
            for group in components:
                if len(group) == 1:
                    valid_classes.append(group[0])
                    continue
                fused_activation = postref[group].mean(0)
                box, cnt = scoremap2bbox(scoremap=fused_activation, threshold=0.4, multi_contour_eval=True)
                have_valid = False
                valid_type = 'crop'
                for i_ in range(cnt):
                    if valid_type == 'crop':
                        if 'B' in self.vit_type:
                            x0_, y0_, x1_, y1_ = box[i_] * 16
                        elif 'L' in self.vit_type:
                            x0_, y0_, x1_, y1_ = box[i_] * 14
                        crop_img = img[:,:,y0_:y1_, x0_:x1_]
                        if y0_ == y1_ or x0_ == x1_:
                            valid_classes.extend(group)
                            continue
                        crop_img = torch.nn.functional.interpolate(crop_img, size=(112, 112), mode='bilinear')
                        _, cls_logits, _, _ = self.net.cam(crop_img, self.query_featuresLLM[self.num_bkg_prompts + candidate_classes[group]], self.model_type)
                    elif valid_type == 'avg_emb':
                        x0_, y0_, x1_, y1_ = box[i_]

                        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                        crop_emb = image_feature[:,1:].permute(0, 2, 1).reshape(-1, 512, w, h)[:,:,y0_:y1_, x0_:x1_]
                        
                        crop_emb = crop_emb.mean((-1,-2))
                        cls_logits = crop_emb @ self.query_featuresLLM[self.num_bkg_prompts + candidate_classes[group]].t()
                            
                    true_class = group[cls_logits.argmax(-1).squeeze()]
                    valid_classes.extend([true_class])     
                    have_valid = True
                if not have_valid:
                    valid_classes.extend(group)
            valid_classes = torch.tensor(valid_classes).unique()
            
            candidate_classes = candidate_classes[valid_classes]

            for i in range(self.num_bkg_prompts, logits_ref.shape[1]):
                if i - self.num_bkg_prompts not in candidate_classes:
                    logits_ref[:,i] = logits_ref[:,i] * 0.0
        
        return logits_ref, candidate_classes
    

    def forward_feature(self, img, cls_label=None, logit_size=None):
        if type(img) == list:
            img = img[0]
        patch_size = self.patch_size
        w, h = img[0].shape[-2] // patch_size[0], img[0].shape[-1] // patch_size[1]
        if len(cls_label) == 0:
            logits = torch.zeros(img.shape[0], self.query_features.shape[0], img.shape[-2], img.shape[-1]).cuda()
            if logit_size == None:
                logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
            else:
                logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
            return logits, None, None, cls_label

        text_embedding = self.query_features
        if self.model_type == 'GEM':
            logits, cls_logits, attn_weight_list, image_feature = self.net.cam(img, text_embedding, self.model_type)
        else:
            logits, cls_logits, attn_weight_list, image_feature = self.net.cam(img, text_embedding, self.model_type)
        
        logits = torch.cat([logits[:,:,:self.num_bkg_prompts], 
                            logits[:,:,self.num_bkg_prompts + cls_label]], dim=-1)

        logits = logits.softmax(dim=-1)
        
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        normalized_logits = (logits - logits.min()) / (logits.max() - logits.min())
        attn_weight = None
        
        logits_ref = self.refinement(img, normalized_logits, attn_weight_list, cls_label)
        logits_ref, candidate_classes = self.redundancy_purification(logits, logits_ref)
        if len(candidate_classes) > 1:
            logits_ref, candidate_classes = self.ambiguity_purification(logits, logits_ref, candidate_classes, img)
        logits = logits_ref 
        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        
        return logits, cls_logits, attn_weight, candidate_classes

    def forward_slide(self, img, seg_label, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        local_cls_logits = torch.zeros(h_grids * w_grids, out_channels)
        local_cls_labels = torch.zeros(h_grids * w_grids, out_channels - self.num_bkg_prompts)
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_cls_label = seg_label

                crop_seg_logit, cls_logits, attn_weight, cls_labels = self.forward_feature(crop_img, crop_cls_label)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits, seg_label

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        name = data_samples[0].metainfo['img_path'].split('/')[-1][:-4]
        if self.dataset == 'voc':
            label_path = data_samples[0].metainfo['img_path'].replace('JPEGImages', 'SegmentationClass').replace('jpg', 'png')
            cls_label = torch.tensor(range(self.num_classes)).cuda()
            if not self.ignore_bkg:
                cls_label = cls_label[1:] - 1
        elif self.dataset == 'pc':
            label_path = data_samples[0].metainfo['img_path'].replace('JPEGImages', 'SegmentationClassContext').replace('jpg', 'png')
            cls_label = torch.tensor(range(self.num_classes)).cuda()
            if not self.ignore_bkg:
                cls_label = cls_label[1:] - 1
        elif self.dataset == 'cocoobj':
            label_path = data_samples[0].metainfo['img_path'].replace('images/val2017', 'annotations/object').replace('.jpg', '_instanceTrainIds.png')
            cls_label = torch.tensor(range(self.num_classes)).cuda()
            cls_label = cls_label[1:] - 1
        elif self.dataset == 'cocostuff':
            label_path = data_samples[0].metainfo['img_path'].replace('images/val2017', 'annotations/stuff').replace('.jpg', '_labelTrainIds.png')
            cls_label = torch.tensor(range(self.num_classes)).cuda()
        elif self.dataset == 'ade':
            label_path = data_samples[0].metainfo['img_path'].replace('images', 'annotations').replace('jpg', 'png')
            cls_label = torch.tensor(range(self.num_classes+1)).cuda()  ## ADE need add one bkg class !!!
            cls_label = cls_label[1:] - 1
        elif self.dataset == 'city':
            label_path = data_samples[0].metainfo['img_path'].replace('leftImg8bit', 'gtFine').replace('.png', '_labelTrainIds.png')
            cls_label = torch.tensor(range(self.num_classes)).cuda()

        seg_label = torch.from_numpy(np.array(Image.open(label_path)))
        seg_label = seg_label * (seg_label != 255)
        seg_label = nn.functional.interpolate(seg_label.unsqueeze(0).unsqueeze(0), size=inputs.shape[-2:], mode='nearest')[0,0]

        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        cls_logits=None
        if self.slide_crop > 0:
            candidate_classes_global = cls_label
            seg_logits, candidate_classes_local = self.forward_slide(inputs, candidate_classes_global, batch_img_metas, self.slide_stride, self.slide_crop)
            candidate_classes = candidate_classes_global
        else:
            seg_logits, cls_logits, attn_weight, candidate_classes = self.forward_feature(inputs, cls_label, batch_img_metas[0]['ori_shape'])
            
        return self.postprocess_result(seg_logits, cls_logits, data_samples, cls_label)

    def postprocess_result(self, seg_logits, cls_logits, data_samples, labels):            
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i]
            if len(labels) == 0:
                seg_logits = seg_logits * 0
                seg_pred = seg_logits.argmax(0, keepdim=True)
                data_samples[i].set_data({
                    'seg_logits':
                    PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                    PixelData(**{'data': seg_pred})
                })
                continue

            seg_logits = seg_logits[self.num_bkg_prompts + labels]
            
            if self.with_bkg:
                if self.ignore_bkg:
                    ## Argmax
                    seg_pred = seg_logits.argmax(0, keepdim=False)
                    seg_pred = labels[seg_pred].unsqueeze(0)
                else:
                    ## Background value
                    bg_score = torch.pow(1 - torch.max(seg_logits, dim=0, keepdims=True)[0], 1)
                    seg_logits = torch.cat([bg_score, seg_logits], dim=0)
                    seg_pred = seg_logits.argmax(0, keepdim=False)
                    labels = torch.nn.functional.pad(labels + 1, (1, 0), mode='constant')
                    seg_pred = labels[seg_pred].unsqueeze(0)
            else:
                ## Argmax
                seg_pred = seg_logits.argmax(0, keepdim=False)
                seg_pred = labels[seg_pred].unsqueeze(0)
            
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples


    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """

def get_prompts(path, dataset, use_llm=False):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]

    prompts = []
    if use_llm:
        if 'voc' in dataset:
            dataset_prompts = VOC
        elif 'ade' in dataset:
            dataset_prompts = ADE20K
        elif 'city' in dataset:
            dataset_prompts = CS
        elif 'pc' in dataset:
            dataset_prompts = PC
        elif 'cocoobj' in dataset:
            dataset_prompts = COCOOBJ
        elif 'cocostuff' in dataset:
            dataset_prompts = COCOSTUFF
        for name in class_names:
            prompts.append(dataset_prompts[name])
        return class_names, class_indices, prompts
    
    for name in class_names:
        prompts.append([temp(name) for temp in openai_imagenet_template])
    return class_names, class_indices, prompts

    
def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    scoremap = scoremap.cpu().numpy()
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


def dfs(matrix, visited, node, component):
    visited[node] = True
    component.append(node)
    for neighbor in range(len(matrix)):
        if matrix[node][neighbor] == 1 and not visited[neighbor]:
            dfs(matrix, visited, neighbor, component)

def connected_components(matrix):
    n = len(matrix)
    visited = [False] * n
    components = []
    for node in range(n):
        if not visited[node]:
            component = []
            dfs(matrix, visited, node, component)
            components.append(component)
    return components
