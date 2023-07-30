# TSPMN
Matching-based Term Semantics Pre-training for Spoken Patient Query Understanding, ICASSP 2023.

## Requirement

* pytorch 1.10.0
* transformers 4.18.0
* deepspeed 0.6.3
* apex 0.1
* 
## Usage

### Pretraining

Download the [MedDialog](https://drive.google.com/drive/folders/11sglwm6-cY7gjeqlZaMxL_MDKDMLdhym), [kamed](https://drive.google.com/drive/folders/1i-qiwVgOHS9Cs_7YSNdUCWwviP2HOgqI) and [ReMeDi-large](https://drive.google.com/drive/folders/1nxVEci21eU5KSejiWM4fwRlRELvkncpe) datasets and save them to data4pretrain  Folder. 
```
cd data4pretrain/VRBot-sigir2021-datasets
unzip 'kamed-*.zip'
cd ..
```

Processing medical dialogues:
```
python preprocess4pretrain.py 
```
Constructing the Dict. Download the [sougou medical dictionary](https://pinyin.sogou.com/dict/detail/index/15125) and the dictionary [THUOCL](https://github.com/thunlp/THUOCL) in medical domain. Then:
```
python medical_dict/dict.py
```
Constructing the dialogue-term pairs for pretraining.
```
cd data4pretrain, python medical_word_extract.py
```
Pretraining:
```
sh pretraining_deepspeed/run_train.sh
```
### Finetuning

Download the [MSL](https://github.com/xmshi-trio/MSL) Dataset. Then:
```
python data_process_MSL.py
```
Training:
```
nohup CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 19600 --nproc_per_node=1 train_Parallel.py > run.log 2>&1 &
```
Evaluating:
```
python evaluate.py
```
