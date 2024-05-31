import torch
from overlaps import bbox_overlaps
from statistics import mean
from datasets import load_dataset


# define args functions
# to receive the arguments from the command
def define_args():
    import argparse
    parser = argparse.ArgumentParser(description='Eval the prediction')
    parser.add_argument('--output_path', type=str, default='', help='path to the prediction file')
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
    output_path=args.output_path
    split=args.split
    dataset_path=args.dataset_path

    pred_root=f"{output_path}/custom_metrics.pth"
    preds = torch.load(pred_root)
    all_annts=load_dataset(dataset_path)[split]

    pred_bboxes=[]
    gt_bboxes=[]
    for idx,annt in enumerate(all_annts):
        pred_bbox=preds['pred_boxes'][idx]
        # print(pred_bbox)
        img_path=annt['file_name']
        height,width=annt['height'],annt['width']

        gt_bbox=annt['bbox']
        gt_bbox=[gt_bbox[0],gt_bbox[1],gt_bbox[0]+gt_bbox[2],gt_bbox[1]+gt_bbox[3]]
        if(pred_bbox!=[0,0,0,0]):
            pred_bbox=list(box_xyxy_de_expand2square(pred_bbox*max(width,height), w=width, h=height))
        gt_bboxes.append(gt_bbox)
        pred_bboxes.append(pred_bbox)

    pred_bboxes=torch.tensor(pred_bboxes)
    gt_bboxes=torch.tensor(gt_bboxes)
    # create a list fron 0.5 to 0.9 with step 0.05
    thresh=[i/100 for i in range(50,100,5)]
    print(thresh)
    iou,acc=calculate_iou_acc(pred_bboxes,gt_bboxes,thresh)
    print_ths=[0.5,0.75,0.9]

    acc_list=[]
    for k,v in acc.items():
        if(k in print_ths):
            print(f"iou|{k}={v}")
            acc_list.append(v)
    print(f"iou|0.5:0.9={mean([v for v in acc.values()])}")
    acc_list.append(mean([v for v in acc.values()]))
