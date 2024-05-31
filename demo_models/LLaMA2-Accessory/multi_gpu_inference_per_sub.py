from SPHINX import SPHINXModel
from PIL import Image
import torch
import torch.distributed as dist
import multiprocessing as mp
import json
import os
import time
import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')  
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
os.chdir("/home/aiscuser/REC_Inference/LLaMA2-Accessory")

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
    # randomly choose one prompt from prompts
    # gen a random index
    # idx = torch.randint(0, len(prompts), (1,)).item()

    return prompts[idx-1]

def postprocess_box(box, ori_w, ori_h):
    if ori_w == ori_h:
        return box
    if ori_w > ori_h:
        x1, y1, x2, y2 = box
        y1 -= (ori_w - ori_h) // 2
        y2 -= (ori_w - ori_h) // 2
        box = x1, y1, x2, y2
        return box
    x1, y1, x2, y2 = box
    x1 -= (ori_h - ori_w) // 2
    x2 -= (ori_h - ori_w) // 2
    box = x1, y1, x2, y2
    return box

def pickup_subject_annts(json_path, subject):
    with open(json_path) as f:
        person_annts = json.load(f)
    subject_annts = []
    for annt in person_annts:
        caption=annt["caption"]
        sub_sentences=tokenizer.tokenize(caption)
        all_subject_sentences=[]
        for label in annt["labels_v2"]:
            if label["category"] == subject:
                start_indice=label['location'][0]
                end_indice=label['location'][1]
                patch_caption=caption[start_indice:end_indice]
                # find the sub-sentence where the patch_caption is located
                subject_sentence = None
                for sent in sub_sentences:
                    if patch_caption in sent:
                        subject_sentence = sent
                        break
                if subject_sentence is None:
                    subject_sentence=patch_caption
                all_subject_sentences.append(subject_sentence)
                break
        if len(all_subject_sentences)>0:
            annt['subject_caption']=" ".join(all_subject_sentences)
            subject_annts.append(annt)
    return subject_annts

def main(ckpt_path, mode, json_path, img_folder, output_dir, local_rank=0) -> None:
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    print("World size: ", world_size)
    print("Rank: ", rank)
    if(world_size>1):
        dist.init_process_group(
                world_size=world_size, rank=rank,
                backend="nccl", init_method=f"env://",
            )
        use_dist=True
        if(rank==0):
            print("Distributed training enabled")
    else:
        use_dist=False
    torch.cuda.set_device(rank)

    person_annts = pickup_subject_annts(json_path, args.subject)
    
    # mp_group tells the model which ranks will work together
    # through model parallel to compose a complete model.
    # When mp_group is None, a single-rank process group will
    # be created and used, which means model parallel size = 1 (not enabled)
    model = SPHINXModel.from_pretrained(
        pretrained_path=ckpt_path, with_visual=True,
        mp_group=dist.new_group(ranks=list(range(world_size))) if use_dist else None
    ) 
    
    # it's important to make sure that ranks within the same 
    # model parallel group should always receive the same input simultaneously
    pred_bboxes=[]
    answers_file=os.path.join(output_dir, args.subject.replace(' ','_'), "preds_during_training.json")
    if(rank==0):
        os.makedirs(os.path.join(output_dir, args.subject.replace(' ','_')), exist_ok=True)
        ans_file = open(answers_file, "w")
    start_time=time.time()
    for annt in tqdm(person_annts):
        image = Image.open(os.path.join(img_folder, annt["file_name"]))
        description = annt[mode]
        qas = [[f"Provide coordinates for the bounding box around the person mentioned in: \"{description}\"", None]]
        try_idx=1
        while(try_idx<11):
            response = model.generate_response(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
            try:
                pred_bbox=eval(response)
                break
            except:
                print(f"Error in response: \"{response}\" {try_idx} times")
                pred_bbox=[0,0,0,0]
                qas=[[replace_bounding_box_prompts(description,try_idx), None]]
                try_idx+=1
                print(f"Trying again with prompt: \"{qas[0][0]}\"")
                continue
        height, width = annt["height"], annt["width"]
        if(type(pred_bbox) is list):
            try:
                max_ori=max(height, width)
                x1, y1, x2, y2 = pred_bbox
                square_bbox = round(x1 * max_ori,3), round(y1 * max_ori,3), round(x2 * max_ori,3), round(y2 * max_ori)
                pred_bbox = postprocess_box(square_bbox, width, height)
                # pred_bbox = [pred_bbox[0]*width, pred_bbox[1]*height, pred_bbox[2]*width, pred_bbox[3]*height]
            except:
                pred_bbox=[0,0,0,0]
        else:
            pred_bbox=[0,0,0,0]
        pred_bboxes.append({"pred_bbox":pred_bbox,
                            "annt_id":annt["annt_id"]})
        if(rank==0):
            ans_file.write(json.dumps({"annt_id": annt['annt_id'],    
                                    "file_name": annt["file_name"],
                                    'caption': description,
                                    "pred_bboxes": pred_bbox,
                                    }) + "\n")
            ans_file.flush()

        # img=cv2.imread(os.path.join(img_folder, annt["file_name"]))
        # img=cv2.rectangle(img, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 255, 0), 2)
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.savefig("output.png")

        
        # print(rank,": ",response)
        # break
    if(rank==0):
        ans_file.close()
        print("Time taken: ", time.time()-start_time)
        #save the results
        with open(os.path.join(output_dir, args.subject.replace(' ','_'), "pred_bboxes.json"), "w") as f:
            # save with format
            json.dump(pred_bboxes, f)
            print("Results saved at: ", os.path.join(output_dir, args.subject.replace(' ','_'), "pred_bboxes.json"))

    if(use_dist):
        dist.destroy_process_group()

if __name__ == "__main__":
    # use args
    label_categroy={
    0:"Appearance",
    1:"Human-Object Interaction",
    2:"Celebrity",
    3:"OCR",
    4:"Action",
    5:"Location",
    }
    category_label={v:k for k,v in label_categroy.items()}

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="Human-Object Interaction")
    parser.add_argument("--mode", type=str, default="subject_caption")
    parser.add_argument("--json_path", type=str, default="/home/aiscuser/datasets/rec_human/person_annts_val_ready_label_v2.json")
    parser.add_argument("--img_folder", type=str, default="/home/aiscuser/datasets/rec_human/val_ready")
    parser.add_argument("--ckpt_path", type=str, default="/home/aiscuser/ckpts/LLaMA2-Accessory/finetune/mm/SPHINX/SPHINX")
    parser.add_argument("--output_dir", type=str, default="output_per_sub/SPHINX/")
    parser.add_argument("--local-rank", type=int, default=0)
    
    args = parser.parse_args()
    if(args.subject=='Human-Object'):
        args.subject='Human-Object Interaction'
    mode = args.mode
    json_path = args.json_path
    img_folder = args.img_folder
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    local_rank = args.local_rank
    main(ckpt_path, mode, json_path, img_folder, output_dir, local_rank)
    # N_GPU = 2
    # assert N_GPU in [1, 2, 4, 8]
    # if N_GPU == 1:
    #     main(world_size=1, rank=0)
    # else:
    #     # You can use whatever method, e.g. torchrun, slurm, etc. for distributed launch
    #     # Just be sure to initialize torch distributed (by invoking dist.init_process_group)
    #     # before creating the SPHINX model if model parallel size > 1 is used
    #     mp.set_start_method("spawn")
    #     for rank in range(N_GPU):
    #         process = mp.Process(target=main, args=(N_GPU, rank, ckpt_path, mode, json_path, img_folder))
    #         process.start()
    