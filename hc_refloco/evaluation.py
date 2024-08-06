import torch
from .overlaps import bbox_overlaps
from statistics import mean
import pandas as pd
import math
from .dataloader import HCRefLoCoDataset

class HCRefLoCoEvaluator:
    def __init__(self, dataset, split='val', 
                 thresholds=[i/100 for i in range(50,100,5)],
                 show_ths=[0.5, 0.75, 0.9],
                 subjects=['Appearance','Human-Object Interaction','Celebrity','OCR','Action','Location'],
                 small_size_th=128,
                 large_size_th=256
                 ) -> None:
        '''
        dataset (HCRefLoCoDataset|str): The dataset/path_to_dataset to evaluate.
        split (str): The split of the dataset to evaluate. Default is 'val'.
        thresholds (List[float]): The thresholds to evaluate the IoU. Default is [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
        show_ths (List[float]): The thresholds to show in the evaluation results. Default is [0.5, 0.75, 0.9].
        subjects (List[str]): The subjects to evaluate. Default is ['Appearance','Human-Object Interaction','Celebrity','OCR','Action','Location'].
        small_size_th (int): The threshold to define the small size bbox. Default is 128.
        large_size_th (int): The threshold to define the large size bbox. Default is 256.
        '''
        if(isinstance(dataset, str)):
            dataset=HCRefLoCoDataset(dataset,split,load_img=False)
        elif(isinstance(dataset,HCRefLoCoDataset)):
            if(dataset.split!=split):
                print(f"Warning: the split ({split}) is not match with the dataset.split ({dataset.split})")
                print(f"Warning: the split of the dataset is changed to {split}.")
                dataset.change_split(split)
        else:
            raise ValueError("dataset should be a path to the dataset or a HCRefLoCoDataset object.")
        self.dataset = dataset
        self.split=split
        self.accs=dict()
        self.thresholds=thresholds
        self.show_ths=show_ths
        self.subjects=subjects
        self.small_size_th=small_size_th
        self.large_size_th=large_size_th

    def change_split(self, split):
        self.dataset.change_split(split)
        self.split=split
        
    @staticmethod
    def calculate_iou_acc(bboxes_1, bboxes_2, thresh=0.5):
        """
        bboxes_1 (torch.Tensor, numpy.Array): shape=[N,4], format=[x1, y1, x2, y2]
        bboxes_2 (torch.Tensor, numpy.Array): shape=[N,4], format=[x1, y1, x2, y2]
        calculate the iou and acc of the pred_bboxes and gt_bboxes,
        if iou(pred_bboxes[i],gt_bboxes[i])>0.5, then acc+=1
        all pred_bboxes_i and gt_bboxes_i are one to one assigned.
        
        """
        iou=bbox_overlaps(bboxes_1,bboxes_2,mode='iou', is_aligned=True)
        if(type(thresh) is not list):
            thresh=[thresh]
        accs=dict()
        for t in thresh:
            accs[t]=(iou>t).sum().item()/len(iou)
        return iou,accs

    def evaluate(self, predictions, save_file=None):
        """
        Evaluate the metrics for the given dataset and predictions.

        Parameters:
        - dataset (datasets.DatasetDict): The dataset to evaluate.
            e.g. dataset = load_dataset("HC-RefLoCo")['val']
        - predictions (List(Dict)): The predictions to evaluate. 
            The dict contains the keys: 'pred_bbox', 'id' and 'format', 
            where 'id' is the annotation id, 'format' is the bbox format 'xyxy' or 'xywh'.
            e.g.:
            [
                {
                'pred_bbox': [x1, y1, x2, y2],
                'id': '000000',
                'format': 'xyxy'
                },
                ...
            ]
        - save_file (str): The file to save the evaluation results to.
        """
        if(len(predictions)==0):
            print("Warning: No predictions found.")
            return dict()
        gt_bboxes = []
        pred_bboxes = []
        dataset=self.dataset
        preds_gt_each_subjects={k:[] for k in self.subjects}
        small_list=[]
        medium_list=[]
        large_list=[]
        for idx, pred in enumerate(predictions):
            _,data=dataset[idx]
            if(pred['id']!=data['id']):
                for _data in dataset:
                    if(pred['id']==_data['id']):
                        data=_data
                        break
            assert data['id']==pred['id'], f"pred_id:{pred['id']} not found in dataset."
            gt_bbox=data['bbox']
            gt_bboxes.append([gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]])
            pred_bbox=pred['pred_bbox']
            if(pred['format']=='xywh'):
                pred_bbox=[pred_bbox[0], pred_bbox[1], pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]]
            pred_bboxes.append(pred_bbox)

            # gather the predictions and gt_bboxes for each subjects
            subjects=set([subject['category'] for subject in data['labels']])
            for subject in subjects:
                preds_gt_each_subjects[subject].append(idx)
            
            # gather the small, medium and large size bboxes
            obj_size=math.sqrt(gt_bbox[2]*gt_bbox[3])
            if(obj_size<self.small_size_th):
                small_list.append(idx)
            elif(obj_size<self.large_size_th):
                medium_list.append(idx)
            else:
                large_list.append(idx)
            

        iou, accs = self.calculate_iou_acc(torch.tensor(pred_bboxes), 
                                      torch.tensor(gt_bboxes), 
                                      thresh=self.thresholds)

        acc_dict = dict()

        # iou evaluation
        for k, v in accs.items():
            if k in self.show_ths:
                acc_dict[f"iou|{k}"] = v
        mean_acc = mean([v for v in accs.values()])
        acc_dict['iou|0.5:0.95'] = mean_acc

        # Accs for copy
        acc_dict['Accs for copy'] = [round(v, 3) for v in acc_dict.values()]

        # Subject evaluation
        subjects_accs = dict()
        for subject in preds_gt_each_subjects:
            iou_subject = iou[preds_gt_each_subjects[subject]]
            acc_subject = []
            if len(iou_subject) > 0:
                for t in self.thresholds:
                    acc_subject.append((iou_subject > t).sum().item() / len(iou_subject))
                subjects_accs[subject] = mean(acc_subject)
            else:
                subjects_accs[subject] = None
        acc_dict.update({f"Subject-{subject}": acc for subject, acc in subjects_accs.items()})
        acc_dict['Subject evaluation for copy'] = [round(v, 3) if v is not None else None for v in subjects_accs.values()]

        # Size evaluation
        size_accs = {'small': [], 'medium': [], 'large': []}
        for size_list, size_name in zip([small_list, medium_list, large_list], ['small', 'medium', 'large']):
            ious = iou[size_list]
            if len(ious) > 0:
                for t in self.thresholds:
                    size_accs[size_name].append((ious > t).sum().item() / len(ious))
            else:
                size_accs[size_name] = [None] * len(self.thresholds)
        
        acc_dict['Small'] = mean(size_accs['small']) if size_accs['small'] else None
        acc_dict['Medium'] = mean(size_accs['medium']) if size_accs['medium'] else None
        acc_dict['Large'] = mean(size_accs['large']) if size_accs['large'] else None
        acc_dict['Size evaluation for copy'] = [round(acc_dict['Small'], 3) if acc_dict['Small'] is not None else None, 
                                                round(acc_dict['Medium'], 3) if acc_dict['Medium'] is not None else None, 
                                                round(acc_dict['Large'], 3) if acc_dict['Large'] is not None else None]

        # Output as table
        table = []
        table.append(["Item", "Value"])
        for k, v in acc_dict.items():
            if isinstance(v, list):
                table.append([k, ", ".join(map(str, v))])
            else:
                table.append([k, v])

        # Define where to add horizontal lines
        horizontal_lines = {1, 6, 13}  # After header, IoU, and Subject evaluations

        # Print table with selective horizontal lines
        max_len = max(len(row[0]) for row in table)
        for i, row in enumerate(table):
            if i in horizontal_lines:
                print('-' * (max_len + 3 + max(len(str(r[1])) for r in table)))
            print(f"{row[0].ljust(max_len)} | {row[1]}")

        if(save_file is not None):
            acc_dict.pop('Accs for copy')
            acc_dict.pop('Subject evaluation for copy')
            acc_dict.pop('Size evaluation for copy')
            df=pd.DataFrame(acc_dict, index=[0])
            df.to_csv(save_file)

        return acc_dict
