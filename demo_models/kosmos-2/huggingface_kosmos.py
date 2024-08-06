import os
import json
from tqdm import tqdm
import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor
from hc_refloco import HCRefLoCoDataset, HCRefLoCoEvaluator

def get_args():
    parser = argparse.ArgumentParser(description='Huggingface Kosmos')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output-path', type=str, default='output/kosmos_preds.jsonl')
    parser.add_argument('--resume', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    device = args.device
    dataset_path = args.dataset_path
    split=args.split
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    if len(output_dir) > 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset=HCRefLoCoDataset(dataset_path,split)
    evaluator = HCRefLoCoEvaluator(dataset, split)
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224", device_map=device)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    predictions=[]
    if os.path.exists(output_path) and args.resume:
        ans_file = open(output_path, "r+")
        lines = ans_file.readlines()
    else:
        ans_file = open(output_path, "w")
        lines = []

    for idx, data in enumerate(tqdm(dataset)):
        if(idx<len(lines)):
            pred_item = json.loads(lines[idx].strip())
            predictions.append(pred_item)
            continue
        img, ann = data
        caption=ann['caption']
        prompt = f"<grounding><phrase>{caption}</phrase>"

        # Prepare the inputs
        inputs = processor(text=prompt, images=img, return_tensors="pt", padding=True)

        # Generate with the model
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"].to(device),
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"].to(device),
            use_cache=True,
            max_new_tokens=128
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
        # By default, the generated  text is cleanup and the entities are extracted.
        entities = processor.post_process_generation(processed_text)[1]

        entity_name, (start, end), bboxes=entities[0]
        image_w=ann['width']
        image_h=ann['height']
        for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
            break
        pred_bbox=[orig_x1, orig_y1, orig_x2, orig_y2]

        pred_item = {
            'id': ann['id'],
            'pred_bbox': pred_bbox,
            'format': 'xyxy',
        }
        predictions.append(pred_item)
        ans_file.write(json.dumps(pred_item) + "\n")
        ans_file.flush()
    
    evaluator.evaluate(predictions)
    ans_file.close()
    
