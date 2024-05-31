# HC-RefLoCo
Dataloader and inference demo of several LMMs on HC-RefLoCo benchmark.

## Dataloader

## Evaluation

## Demo Models

Before inference the HC-RefLoCo, you need to install models according to their instructions and download the weights of them. The instructions are placed in every model's folder.

### 1. Shikra

1. change the dataset path and split in 4-th and 5-th rows of `demo_models/shikra/config/_base_/dataset/DEFAULT_TEST_HC-RefLoCo.py`.
2. change the output path in 4-th row of `demo_models/shikra/config/shikra_eval_hc_refloco.py`.