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
from hc_refloco import HCRefLoCoEvaluator
from datasets import load_dataset
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
    def __init__(self, refexp_gt_path, data_split, k=(1, -1), thresh_iou=0.5):
        self.refexp_gt_path=refexp_gt_path
        assert isinstance(k, (list, tuple))
        self.hc_refloco_evaluater=HCRefLoCoEvaluator(refexp_gt_path, data_split)
        print(f"Load {len(self.hc_refloco_evaluater.dataset)} annotations")
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
        if len(self.hc_refloco_evaluater.dataset) != len(predictions):
            print(f"Warning: len(dataset)={len(self.hc_refloco_evaluater.dataset)}, len(predictions)={len(predictions)}")

        predictions_to_eval=[]
        for item_pred in tqdm(predictions):
            predict_boxes = item_pred["pred_bboxes"]
            pred_id=item_pred["id"]
            pred_format='xyxy'
            
            if len(predict_boxes) == 0:
                if(verbose):
                    print(f"Can't find valid bbox for the given phrase {item_pred}.")
                    print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]                                                                                                               
            # predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4).to(dtype=torch.float32)[0]
            predict={
                'pred_bbox': predict_boxes[0],
                'id': pred_id,
                'format': pred_format
            }
            predictions_to_eval.append(predict)
            
        self.hc_refloco_evaluater.evaluate(predictions_to_eval)
   
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', default='demo_models/ml-ferret/output_hc_refloco/0_of_1.jsonl',help='prediction_file')
    parser.add_argument('--data_path', help='annotation_file')
    parser.add_argument('--data_split',default='val', help='annotation_file')
    args = parser.parse_args()

    evaluator = RefExpEvaluatorFromJsonl(
        refexp_gt_path=args.data_path, 
        data_split=args.data_split,
        k=(1, 'mean', 'upper bound'), 
        thresh_iou=0.5,
    )
    evaluator.summarize(args.prediction_file, verbose=False)
