import os
# import multi thread
import threading


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

def download_model(local_dir,repo_id,filename,subfolder,token):
    print(f'开始下载\n仓库：{repo_id}\n大模型：{filename}\n如超时不用管，会自定继续下载，直至完成。中途中断，再次运行将继续下载。')
    while True:   
        try:
            hf_hub_download(local_dir=local_dir,
            repo_id=repo_id,
            token=token,
            filename=filename,
            subfolder=subfolder,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=100
            )
        except Exception as e :
            print(e)
        else:
            print(f'下载完成，大模型保存在：{local_dir}\{filename}')
            break
            
if __name__ == '__main__':
    repo_id='Alpha-VLLM/LLaMA2-Accessory'
    num_all=2
    filename=[f'consolidated.0{i}-of-0{num_all}.model.pth' for i in range(num_all)]
    subfolder='finetune/mm/SPHINX/SPHINX'
    token='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'
    local_dir = r'/home/aiscuser/ckpts/LLaMA2-Accessory/'
    # download with multi thread
    threads = []
    for i in range(num_all):
        threads.append(threading.Thread(target=download_model,args=(local_dir,repo_id,filename[i],subfolder,token)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('下载完成')
