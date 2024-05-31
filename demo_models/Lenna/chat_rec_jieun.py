import argparse
import os
import sys
from tqdm import tqdm
import cv2
import torch
from transformers import AutoTokenizer, CLIPImageProcessor
import json
from model.Lenna import LennaForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.create_test_annfile_mmdet import load_as_mmdet

os.environ['CURL_CA_BUNDLE'] = ''
prompt_list=[
        "Provide coordinates for the bounding box around the mentioned in:",
        "Please provide the bounding box coordinates for the mentioned in:",
        "Please provide the bounding box coordinates for the in the image mentioned in:",
        "Please provide the bounding box coordinates for the in the image:",
        "Please provide the bounding box coordinates for the in the image shown:",
    ]

def parse_args(args):
    parser = argparse.ArgumentParser(description="Lenna chat")
    parser.add_argument("--ckpt-path", default="ckpt/Lenna-7B")
    parser.add_argument("--json-path", default="/home/v-jinjzhao/datasets/jierun/ref_msra_test_coco_o365.json")
    parser.add_argument("--images-path", default="/home/v-jinjzhao/datasets/jierun/images")
    parser.add_argument("--output-path", default="./output-jierun")
    parser.add_argument("--vis_save_path", default="./vis_output-jierun", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--threshold",default=0.3, type=float)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    with open(args.json_path) as f:
        all_annots = json.load(f)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.vis_save_path, exist_ok=True)
    output_dir = args.vis_save_path
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.det_token_idx = tokenizer("[DET]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32

    kwargs = {
        "torch_dtype": torch_dtype,
    }
    model = LennaForCausalLM.from_pretrained(args.ckpt_path, add_pooling_layer=True, low_cpu_mem_usage=True, det_token_idx=args.det_token_idx, **kwargs)
    trained_weight = torch.load(args.ckpt_path + '/attn_weight.pt')
    for param in model.named_parameters():
        if 'gamma_' in param[0] or 'text_choose_attn' in param[0]:
            layer_name = param[0]
            save_name = layer_name.replace('model.visual_model.', 'base_model.model.model.visual_model.')
            param[1].data = trained_weight[save_name]
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)

    model.eval()
    if(not os.path.exists(os.path.join(args.output_path, 'prediction_during_eval.jsonl'))):
        pred_file = open(os.path.join(args.output_path, 'prediction_during_eval.jsonl'), 'w')
        pred_list=[]
    else:
        pred_file = open(os.path.join(args.output_path, 'prediction_during_eval.jsonl'), 'r+')
        pred_list=pred_file.readlines()
    for annt in tqdm(all_annots[len(pred_list):]):

        conv = conversation_lib.conv_templates['llava_v1'].copy()
        conv.messages = []

        g_dino_caption=annt['caption']

        image_path = os.path.join(args.images_path, annt['file_name'])
        if not os.path.exists(image_path):
            print("[Error] File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
        image_clip = image_clip.float()

        dino_input = load_as_mmdet(image_path, caption=g_dino_caption)
        
        for i in range(len(prompt_list)):
            try:
                prompt =f"{prompt_list[i]} \"{g_dino_caption}\""
                # print('[Lenna] Input prompt: ', prompt)
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                input_ids = input_ids.unsqueeze(0).cuda()
                output_ids, outputs = model.evaluate(
                    image_clip,
                    dino_input,
                    input_ids,
                    None,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                    caption=[prompt],
                )
                output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                text_output = text_output.replace("\n", "").replace("  ", " ")
                # print('text_output:', text_output.split('ASSISTANT: ')[-1].split('</s>')[0])

                pred_bboxes = outputs[0].pred_instances.bboxes
                pred_scores = outputs[0].pred_instances.scores
                threshold = args.threshold

                filt_mask = pred_scores == max(pred_scores)
                pred_bboxes_filt = pred_bboxes[filt_mask]
                break
            except:
                print(f"[Error] Detecting ann_id={annt['ann_id']} failed, using [0,0,0,0] as default bbox.")
                print(f"retry {i+1}/{len(prompt_list)} times")
                pred_bboxes_filt = torch.zeros((0, 4))
        # pred_bboxes_filt[:, 2:] -= pred_bboxes_filt[:, :2]

        
        # image_np = cv2.imread(image_path)
        # img = draw(img=image_np, bboxes=pred_bboxes_filt) 
        # status = cv2.imwrite(output_dir + '/vis_img.jpg', img)

        pred_file.write(json.dumps({"ann_id": annt['ann_id'],    
                                    "file_name": annt["file_name"],
                                    "pred_bboxes": pred_bboxes_filt.tolist(),
                                    }) + "\n")
        pred_file.flush()
    pred_file.close()

def draw(bboxes=None, img=None, color=(0,69,255), line_thickness=4,):
    if bboxes.shape[0] == 0:
        return img
    for bbox in bboxes:
        xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin = int(float(xmin))
        ymin = int(float(ymin))
        w = int(float(w))
        h = int(float(h))
        cv2.rectangle(img, (xmin, ymin), (xmin+w, ymin+h), color, line_thickness)
    return img


if __name__ == "__main__":
    main(sys.argv[1:])
