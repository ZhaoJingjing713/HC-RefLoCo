import torch
from overlaps import bbox_overlaps
from statistics import mean
from datasets import load_dataset
from hc_refloco import HCRefLoCoEvaluator
# define args func  tions
# to receive the arguments from the command
def define_args():
    import argparse
    parser = argparse.ArgumentParser(description='Eval the prediction.')
    parser.add_argument('--pred_dir', type=str, default='', help='path to the prediction dir.')
    parser.add_argument('--split', type=str, default='val', help='split of the dataset')
    parser.add_argument('--dataset_path', type=str, default='HC-RefLoCo', help='path to the dataset')
    return parser.parse_args()

def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

def box_xyxy_de_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 -= (w - h) // 2
        y2 -= (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 -= (h - w) // 2
    x2 -= (h - w) // 2
    box = x1, y1, x2, y2
    return box

def calculate_iou_acc(pred_bboxes,gt_bboxes, thresh=0.5):
    """
    pred_bboxes: [N,4]
    gt_bboxes: [N,4]
    calculate the iou and acc of the pred_bboxes and gt_bboxes,
    if iou(pred_bboxes[i],gt_bboxes[i])>0.5, then acc+=1
    all pred_bboxes_i and gt_bboxes_i are one to one assigned.
    
    """
    iou=bbox_overlaps(pred_bboxes,gt_bboxes,mode='iou', is_aligned=True)
    if(type(thresh) is not list):
        thresh=[thresh]
    accs=dict()
    for t in thresh:
        accs[t]=(iou>t).sum().item()/len(iou)
    return iou,accs

if __name__=='__main__':
    args=define_args()
    pred_dir=args.pred_dir
    split=args.split
    dataset_path=args.dataset_path

    pred_path=f"{pred_dir}/metrics.pt"
    preds = torch.load(pred_path)
    evaluator=HCRefLoCoEvaluator(dataset_path,split)
    annotations=evaluator.dataset
    pred_dict=[]
    for idx,annt in enumerate(annotations):
        _,annt=annt
        pred_bbox=preds['test_pred_boxes'][idx]
        height,width=annt['height'],annt['width']

        if(pred_bbox!=[0,0,0,0]):
            pred_bbox=list(box_xyxy_de_expand2square(pred_bbox*max(width,height), w=width, h=height))
        pred_dict.append({
            'id':annt['id'],
            'pred_bbox':pred_bbox,
            'format':'xyxy'
        })
    evaluator.evaluate(pred_dict)

    