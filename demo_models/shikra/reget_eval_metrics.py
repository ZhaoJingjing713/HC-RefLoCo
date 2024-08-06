from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, prepare_target_processor
import numpy as np
import os
import torch
from transformers.trainer import EvalPrediction

cfg, training_args = prepare_args()
model, preprocessor = load_pretrained(cfg.model_args, training_args)
# Some ugly codes to inject target_processor into preprocessor.
# maybe effect model. (e.g. add special token; resize embedding)
model, preprocessor = prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
print_trainable_params(model)

# Prepare data_collator
collator_kwargs = cfg.data_args.collator_kwargs
trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)


pred=np.load(os.path.join(cfg.training_args.output_dir,'test_predictions.npy'))
label_id=np.load(os.path.join(cfg.training_args.output_dir,'test_label_ids.npy'))

metrics = compute_metrics(EvalPrediction(predictions=pred, label_ids=label_id))
print(metrics)
torch.save(metrics,os.path.join(cfg.training_args.output_dir,'custom_metrics.pth'))
print('done')