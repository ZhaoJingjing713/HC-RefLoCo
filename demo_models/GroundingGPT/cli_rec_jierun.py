import argparse
from tqdm import tqdm
import json
import torch
import os
from lego.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_SOUND_TOKEN
from lego.conversation import SeparatorStyle
from lego import conversation as conversation_lib
from lego.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, load_image_square, postprocess_output
from lego.model.builder import CONFIG, load_pretrained_model
from video_llama.processors.video_processor import load_video
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from lego.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
                           DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN, DEFAULT_SOUND_PATCH_TOKEN, DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN

def replace_bounding_box_prompts(description,idx=1):
    prompts = [
        f"Identify and return the bounding box coordinates for the person described in: \"{description}\"",
        f"Generate bounding box coordinates for the person described: \"{description}\"",
        f"Output the bounding box coordinates for the individual characterized as follows: \"{description}\"",
        f"Locate and specify the bounding box coordinates for the person in this description: \"{description}\"",
        f"Calculate the bounding box coordinates for the person described here: \"{description}\"",
        f"Determine and output the bounding box coordinates for the following description of a person: \"{description}\"",
        f"Find the bounding box coordinates for the person with these details: \"{description}\"",
        f"Retrieve and display bounding box coordinates for the person whose description is: \"{description}\"",
        f"Give the bounding box coordinates for the individual described in the following: \"{description}\"",
        f"Generate the coordinates for the bounding box of the person as described here: \"{description}\""
    ]
    return prompts[idx]

def main(args):
    with open(args.json_path) as f:
        annts=json.load(f)
    model, tokenizer, image_processor, video_transform, context_len = load_pretrained_model(args.model_path)
    video_tensor = None
    sound_tensor = None
    os.makedirs(args.output, exist_ok=True)
    answers_file=os.path.join(args.output,'answers.json')
    ans_file = open(answers_file, "a+")
    exist_preds=len(ans_file.readlines())
    for annt in tqdm(annts[exist_preds:]):
        file_name=annt['file_name']
        image_path=os.path.join(args.image_dir,file_name)
        try :
            image = load_image_square(image_path,image_processor)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            conv = conversation_lib.default_conversation.copy()
        except:
            print(f'{image_path} is not a correct image path.')
            continue
        

        inp=f"Provide coordinates for the bounding box around the person mentioned in: \"{annt['caption']}\""
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len + DEFAULT_IMAGE_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None

        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                videos=video_tensor,
                sounds=sound_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, 
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if image_path is not None:
            outputs = postprocess_output(outputs, image_path, absolute_coordinates=True)
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        ans_file.write(json.dumps({"annt_id": annt['ann_id'],    
                                    "file_name": annt["file_name"],
                                    "pred_bboxes": outputs,
                                    }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/GroundingGPT")
    parser.add_argument("--image_dir", type=str, default="/home/v-jinjzhao/datasets/rec_human/images_v2/val_ready/")
    parser.add_argument("--json_path", type=str, default='/home/v-jinjzhao/datasets/rec_human/person_annts_val_ready.json')
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--sound_file", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=str, default='output/')
    args = parser.parse_args()
    main(args)