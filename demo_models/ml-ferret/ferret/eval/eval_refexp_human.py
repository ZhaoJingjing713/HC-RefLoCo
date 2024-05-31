"""
Usage:

python ferret/eval/eval_refexp.py \
    --prediction_file final_result/ferret_13b_checkpoint-final/refexp_result/finetune_refcocog_test \
    --annotation_file data/annotations/finetune_refcocog_test.json

"""
import os
import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.data
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from prettytable import PrettyTable

import re
import json
from statistics import mean

from misc.box_ops import box_iou

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def decode_bbox_from_caption(text, img_w, img_h, verbose=False):
    entities = []
    boxes = []
    
    start = 0
    in_brackets = False
    entity = ""
    box = ""
    
    for i, char in enumerate(text):
        if char == '[':
            in_brackets = True
            entity = text[start:i].strip()
            start = i + 1
        elif char == ']':
            in_brackets = False
            box = text[start:i].strip()
            start = i + 1
            
            # Convert box string to list of integers
            box_list = list(map(int, box.split(',')))
            resized_box_list = resize_bbox(box_list, img_w, img_h)
            entities.append(entity)
            boxes.append(resized_box_list)
            
            # Skip until the next entity (ignoring periods or other delimiters)
            while start < len(text) and text[start] not in ['.', ',', ';', '!', '?']:
                start += 1
            start += 1  # Skip the delimiter
        
    return entities, boxes


def are_phrases_similar(phrase1, phrase2):
    # Step 1: Convert to lower case
    phrase1 = phrase1.lower()
    phrase2 = phrase2.lower()
    
    # Step 2: Standardize spacing around punctuation
    phrase1 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase1).strip()
    phrase2 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase2).strip()
    
    # Step 3: Remove all punctuation
    phrase1 = re.sub(r'[^\w\s]', '', phrase1)
    phrase2 = re.sub(r'[^\w\s]', '', phrase2)
    
    # Step 4: Remove extra white spaces
    phrase1 = ' '.join(phrase1.split())
    phrase2 = ' '.join(phrase2.split())
    
    return phrase1 == phrase2

def calculate_iou_acc(pred_bboxes,gt_bboxes, thresh=0.5):
    """
    pred_bboxes: [N,4]
    gt_bboxes: [N,4]
    calculate the iou and acc of the pred_bboxes and gt_bboxes,
    if iou(pred_bboxes[i],gt_bboxes[i])>0.5, then acc+=1
    all pred_bboxes_i and gt_bboxes_i are one to one assigned.
    
    """
    iou=box_iou(pred_bboxes,gt_bboxes,mode='iou', is_aligned=True)
    if(type(thresh) is not list):
        thresh=[thresh]
    accs=dict()
    for t in thresh:
        accs[t]=(iou>t).sum().item()/len(iou)
    return iou,accs

class RefExpEvaluatorFromJsonl(object):
    def __init__(self, refexp_gt_path, k=(1, -1), thresh_iou=0.5):
        self.refexp_gt_path=refexp_gt_path
        assert isinstance(k, (list, tuple))
        with open(refexp_gt_path, 'r') as f:
            self.refexp_gt = json.load(f)
        print(f"Load {len(self.refexp_gt)} annotations")
        self.k = k
        self.thresh_iou = thresh_iou

    def summarize(self,
                  prediction_file: str,
                  verbose: bool = False,):
        
        # get the predictions
        if os.path.isfile(prediction_file):
            predictions = [json.loads(line) for line in open(prediction_file)]
        elif os.path.isdir(prediction_file):
            predictions = [json.loads(line) for pred_file in os.listdir(prediction_file) for line in open(os.path.join(prediction_file, pred_file))]
        else:
            raise NotImplementedError('Not supported file format.')
        
        # sort the predictions based on 'image_id'
        # predictions = sorted(predictions, key=lambda x: x['image_id'])
        assert len(self.refexp_gt) == len(predictions), f"len(self.refexp_gt)={len(self.refexp_gt)}, len(predictions)={len(predictions)}"

        all_gt_bboxes=[]
        all_pred_bboxes=[]
        all_mean_pred_bboxes=[]
        for item_ann, item_pred in tqdm(zip(self.refexp_gt, predictions)):
                
            if item_pred['image_id'] != item_ann['image_id']:
                raise ValueError(f"Ann\n{item_ann} \nis not matched\n {item_pred}")

            target_bbox = item_ann["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            all_gt_bboxes.append(converted_bbox)
            predict_boxes = item_pred["pred_bboxes"]
            
            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase {item_pred}.")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]                                                                                                               

            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4).to(dtype=torch.float32)
            all_pred_bboxes.append(predict_boxes[0])
            all_mean_pred_bboxes.append(predict_boxes.mean(0))
            
        pred_bboxes=torch.stack(all_pred_bboxes)
        mean_pred_bboxes=torch.stack(all_mean_pred_bboxes)
        gt_bboxes=torch.tensor(all_gt_bboxes)
        # create a list fron 0.5 to 0.9 with step 0.05
        thresh=[i/100 for i in range(50,95,5)]
        print(thresh)
        iou,acc=calculate_iou_acc(pred_bboxes,gt_bboxes,thresh)
        print_ths=[0.5,0.7,0.9]
        for k,v in acc.items():
            if(k in print_ths):
                print(f"thresh={k}, acc={v}")
        print(f"iou|0.5:0.9={mean([v for v in acc.values()])}")

        print(iou)
        print(acc)
        print(f"{len(iou)}/{len(self.refexp_gt)}")
        print(f"##############################################")
        print(f"mean_pred_bboxes")
        print(thresh)
        iou,acc=calculate_iou_acc(mean_pred_bboxes,gt_bboxes,thresh)
        print_ths=[0.5,0.7,0.9]
        for k,v in acc.items():
            if(k in print_ths):
                print(f"thresh={k}, acc={v}")
        print(f"iou|0.5:0.9={mean([v for v in acc.values()])}")

        print(iou)
        print(acc)
        print(f"{len(iou)}/{len(self.refexp_gt)}")

        for pred_box,iou_i, annt in zip(pred_bboxes.tolist(), iou, self.refexp_gt):
            annt['ferret_7b_caption_pred']=dict(
                pred_bbox=pred_box,
                iou=iou_i.item()
            )
        # with open('/home/v-jinjzhao/dev/datasets/rec_human/person_annts_val_ready.json', 'w') as f:
        #     json.dump(self.refexp_gt, f, indent=2)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', default='refexp_result/ref_human_13b',help='prediction_file')
    parser.add_argument('--annotation_file', default='/home/v-jinjzhao/dev/datasets/rec_human/person_annts_val_ready.json', help='annotation_file')
    
    args = parser.parse_args()
    
    evaluator = RefExpEvaluatorFromJsonl(
        refexp_gt_path=args.annotation_file, 
        k=(1, 'mean', 'upper bound'), 
        thresh_iou=0.5,
    )
    
    evaluator.summarize(args.prediction_file, verbose=False)
    
    # with open(os.path.join(args.prediction_file, "metric.json"), "w") as f:
    #     json.dump(results, f, indent=2)