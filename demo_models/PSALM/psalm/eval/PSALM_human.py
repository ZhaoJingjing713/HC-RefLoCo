import argparse
import torch
import os
from enum import Enum
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from scipy.ndimage import label  
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX, REFER_TOKEN_INDEX
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import cv2
from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from psalm.train.train_datasets import DataCollatorForCOCODatasetV2, RefCOCO_dataset

from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
import pickle
from pathlib import Path


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

color = [255, 0, 0]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def parse_outputs(outputs):
    mask_list = []
    bbox_list = []

    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output['instances'].pred_masks
        scores = output['instances'].scores.cpu().numpy()

        topk_scores,idx = torch.topk(torch.tensor(scores),1)
        idx = idx.cpu().numpy()
        topk_preds = pred_mask[idx,:]
        bbox_preds,max_mask= get_bbox_from_mask(topk_preds)
        mask_list.append(max_mask)
        bbox_list.append(bbox_preds)
    return mask_list,bbox_list

def compute_metric(intersection_meter,union_meter,acc_iou_meter, results_list):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    for results in results_list:
        # gt = results['gt']
        preds = results['pred']
        scores = results['scores']
        preds = preds.astype(np.uint8)
        # pick mask with maximum score
        topk_scores,idx = torch.topk(torch.tensor(scores),1)
        idx = idx.cpu().numpy()
        topk_preds = preds[idx,:]
        if results['pred_cls'] is not None:
            topk_pred_cls = results['pred_cls'][idx]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        # here topk=1, len(topk_preds)=1
        for i,pred_ in enumerate(topk_preds):
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_).int().cuda().contiguous().clone(), torch.tensor(gt).int().cuda().contiguous(), 2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(topk_preds[max_i])
        # gt_list.append(gt)

    return pred_list,gt_list

def get_max_connected_component(mask):
    # Define an 8-connected structure  
    structure = np.ones((3, 3), dtype=np.int32)  # This defines the connectivity for 8 directions  
    
    # Label the connected components with the defined structure  
    labeled_array, num_features = label(mask, structure=structure)  
    
    # Find the size of each connected component  
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip the zero label  
    
    # Find the label of the largest component  
    if(len(component_sizes)==0):
        return mask, 0
    max_component_label = np.argmax(component_sizes) + 1  # Plus 1 because we skipped the zero label  
    
    # Create a mask of the largest connected component  
    max_component_mask = (labeled_array == max_component_label)  
    # get the area of the largest connected component
    max_component_area = np.sum(max_component_mask)
    return max_component_mask,max_component_area

def get_bbox_from_mask(pred_masks):
    bbox_img = []
    max_mask = None
    pred_masks = pred_masks.detach().cpu().numpy()
    mask_list = [pred_masks[i] for i in range(pred_masks.shape[0])]
    if len(mask_list) > 0:
        max_area=0
        max_mask=None
        for curr_mask in mask_list:
            curr_mask = curr_mask > 0
    
            curr_mask, curr_area = get_max_connected_component(curr_mask)
            if max_mask is None or curr_area>max_area:
                max_area = curr_area
                max_mask = curr_mask
        if max_area>0:
            contours, _ = cv2.findContours(max_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox_img=[x, y, x + w, y + h]

    if len(bbox_img) == 0:
        bbox_img = [0, 0, 0, 0]
    return bbox_img, max_mask

class RecHumanDataset(RefCOCO_dataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['file_name']
        image_folder = self.data_args.image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['height']
        data_dict['width'] = data['width']
        data_dict['annt_id'] = data['annt_id']
        # data_dict['annotations'] = data['anns']


        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        sentences = data['caption']
        # prefix_inst = 'Referring Segmentation according to the following instruction:'
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = sentences

        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'referring_human'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='/home/v-jinjzhao/datasets/rec_human_celeb_ready/images_v2/val_ready')
    model_path: Optional[str] = field(default="ckpt/PSALM")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = '/home/v-jinjzhao/datasets/rec_human_celeb_ready/person_annts_val_ready.json'
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output_human/'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="referring")




def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    save_suffix = os.path.basename(data_args.json_path).split('.')[0]
    print(f'save suffix is {save_suffix}')
    print(f'current model is {model_path}')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, model_args=data_args, mask_config=data_args.mask_config, device='cuda')

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]

    data_args.refcoco_image_folder = data_args.image_folder
    eval_dataset = RecHumanDataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'],shuffle=False)

    def load_ref_dataset():
        return RecHumanDataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)

    DatasetCatalog.register('refcoco_dataset', load_ref_dataset)
    MetadataCatalog.get('refcoco_dataset').set(stuff_classes=['object'],)
    gt_json_path = data_args.json_path
    with open(gt_json_path) as f:
        gt_data = json.load(f)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device=device,dtype=torch.float).eval()
    save_list = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    os.makedirs(data_args.output_dir,exist_ok=True)
    pred_file_path=os.path.join(data_args.output_dir,'pred.jsonl')
    if(os.path.exists(pred_file_path)):
        pred_file = open(pred_file_path,'r+')
        exist_list=pred_file.readlines()
    else:
        pred_file = open(pred_file_path,'w')
        exist_list=[]
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            if(idx<len(exist_list)):
                continue
            # gt = gt_data[idx]
            # h, w = gt_data[idx]['image_info']['height'], gt_data[idx]['image_info']['width']
            # # generate gt mask
            # masks = []
            # for annotation in gt:
            #     if isinstance(annotation['segmentation'], list):
            #         segm = np.zeros((h, w), dtype=np.uint8)
            #         for poly in annotation['segmentation']:
            #             poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
            #             cv2.fillPoly(segm, [poly], 1)
            #         masks.append(segm.astype(np.bool_))
            #     else:
            #         if isinstance(annotation['segmentation']['counts'], list):
            #             rle = mask.frPyObjects(annotation['segmentation'], *annotation['segmentation']['size'])
            #             segm = mask.decode(rle)
            #         else:
            #             segm = mask.decode(annotation['segmentation'])
            #         masks.append(segm.astype(np.bool_))
            # assert len(masks) == 1
            # gt_mask = masks[0].astype(np.uint8)

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
            outputs = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'].float(),
                seg_info=inputs['seg_info'],
                token_refer_id = inputs['token_refer_id'],
                refer_embedding_indices=inputs['refer_embedding_indices'],
                labels=inputs['labels']
            )
            # gt_cls = inputs['seg_info'][0]['instances'].gt_classes
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mask_list,bbox_list = parse_outputs(outputs)
            pred_file.write(json.dumps({'annt_id': inputs['seg_info'][0]['annt_id'], 'bbox':bbox_list})+'\n')
            pred_file.flush()

            # plot the res
            # for box_i, res_i in zip(bbox_list,mask_list):
            #     mask=res_i>0
            #     img=cv2.imread(inputs['seg_info'][0]['file_name'])
            #     # draw the mask on the image and the bbox on the image

            #     res_i = res_i.astype(np.uint8)
            #     img[mask] = (img * 0.5 + mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[mask]
            #     cv2.rectangle(img, (box_i[0], box_i[1]), (box_i[2], box_i[3]), color, 2)
            #     cv2.imwrite('test.jpg',img)


            # pred,gt_mask = compute_metric(intersection_meter,union_meter,acc_iou_meter, cur_res)
            # save_list.append({'pred':pred[0],'gt':gt_mask[0],'name':inputs['seg_info'][0]['file_name']})
    pred_file.close()
    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # ciou = iou_class[1]
    # giou = acc_iou_meter.avg[1]
    # msg = "benchmark: {}: giou: {:.4f}, ciou: {:.4f}".format(save_suffix, giou, ciou)
    # print(msg)
    # save_path = os.path.join(data_args.model_path,'pred_pkl')
    # Path(save_path).mkdir(parents=True,exist_ok=True)
    # with open(os.path.join(save_path,f'pred_{save_suffix}.txt'),'w') as f:
    #     f.write(msg)







if __name__ == "__main__":
    evaluation()