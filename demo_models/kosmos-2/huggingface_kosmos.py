from transformers import AutoProcessor, AutoModelForVision2Seq
import os
import json
from PIL import Image
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForVision2Seq, AutoProcessor
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, img_dir):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            img_dir (string): Directory with all the images.
        """
        with open(json_file, 'r') as f:
            self.img_annotations = json.load(f)[:20]
        self.img_dir = img_dir
 
    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_annotations[idx]['file_name'])
        image = Image.open(img_path)
        caption = f"<grounding><phrase>{self.img_annotations[idx]['caption']}</phrase>"
        annt_id = self.img_annotations[idx]['annt_id']

        return image,caption,annt_id

from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    """处理 PIL 图像和其他数据的批处理函数。
    """
    batch_mod = []
    for items in batch:
        image, caption, annt_id = items
        batch_mod.append({
            'image': image,
            'caption': caption,
            'annt_id': annt_id 
        })
    return {key: default_collate([d[key] for d in batch_mod]) if key != 'image' else [d['image'] for d in batch_mod] for key in batch_mod[0]}

    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on multiple GPUs using DDP")
    parser.add_argument('--json_file', type=str,  help='Path to the JSON file containing annotations')
    parser.add_argument('--img_dir', type=str, help='Directory where images are stored')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--use_ddp', action='store_true', help='Use distributed data parallel')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed from torch.distributed.launch')
    parser.add_argument('--save_frequent', type=int, default=2, help='save frequent')
    return parser.parse_args()

def train(args):

    if args.use_ddp:
        setup(args.local_rank, torch.cuda.device_count())
    
    dataset = ImageCaptionDataset(json_file=args.json_file, img_dir=args.img_dir)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    if args.use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), rank=args.local_rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
        model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").cuda(args.local_rank).eval()
        model = DDP(model, device_ids=[args.local_rank])
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").cuda().eval()

    pred_tmp_results=[]
    for iter_idx, inputs in enumerate(tqdm(dataloader)):
        images = inputs['image']   # 这是一个 PIL 图像列表
        captions = inputs['caption']
        annt_ids = inputs['annt_id']
        inputs=processor(text=captions, images=images, return_tensors="pt", padding=True)
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"].cuda(args.local_rank),
            input_ids=inputs["input_ids"].cuda(args.local_rank),
            attention_mask=inputs["attention_mask"].cuda(args.local_rank),
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"].cuda(args.local_rank),
            use_cache=True,
            max_new_tokens=128
        ).cpu()
        # batch_tmp_res={k:v for k,v in zip(annt_ids,generated_ids)}
        pred_tmp_results.append(generated_ids)
        if((iter_idx+1)%args.save_frequent==0):
            # gather from all devices
            if(args.use_ddp):
                rank=dist.get_rank() 
                torch.save(pred_tmp_results,f'output/ckpt_iter{iter_idx}_rank{rank}.pt')
            else:
                torch.save(pred_tmp_results,f'output/ckpt_iter{iter_idx}.pt')
    if(args.use_ddp):
        rank=dist.get_rank()
        torch.save(pred_tmp_results,f'output/pred_mid_res_rank{rank}.pt')
    else:
        torch.save(pred_tmp_results,f'output/pred_mid_res.pt')
    
    pred_results=[]
    for decoder_inputs in pred_tmp_results:
        generated_texts = processor.batch_decode(decoder_inputs, skip_special_tokens=True)

        # Process the output texts
        processed_texts = [processor.post_process_generation(gt, cleanup_and_extract=False) for gt in generated_texts]
        entities = [processor.post_process_generation(text)[1] for text in processed_texts]

        pred_results.extend(entities)

    if(args.use_ddp):
        rank=dist.get_rank()
        torch.save(pred_results,f'output/pred_res_rank{rank}.pt')
    else:
        torch.save(pred_results,f'output/pred_res.pt')


    if args.use_ddp:
        cleanup()

if __name__ == "__main__":

    device = 'cuda'

    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224", device_map=device)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    # Load images
    with open('/home/v-jinjzhao/dev/datasets/rec_human/person_annts_val_ready.json') as f:
        data=json.load(f)
    prompts=[]
    images=[]
    resuls=[]
    for i in tqdm(data):
        img_name=i['file_name']
        caption=i['rewrite']
        img_dir='/home/v-jinjzhao/dev/datasets/rec_human/images_v2/val_ready'
        # print(os.path.join(img_dir, img_name))
        # print(caption)
        prompt = f"<grounding><phrase>{caption}</phrase>"
        prompts.append(prompt)
        img=Image.open(os.path.join(img_dir, img_name))
        images.append(img)

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

        # print(processed_text)
        # `<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.`

        # By default, the generated  text is cleanup and the entities are extracted.
        entities = processor.post_process_generation(processed_text)[1]

    # print(processed_text)
    # `An image of a snowman warming himself by a fire.`
        resuls.append(entities)
        # print(entities)
    torch.save(resuls,'output/rewrite_pred_result.pt')