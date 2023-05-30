# TSPMN
Matching-based Term Semantics Pre-training for Spoken Patient Query Understanding, ICASSP 2023.

### pretraining

1. Download the MedDialog, kamed and ReMeDi Dataset. 

2. preprocessing for pretraining
```
python make_data_MedDialog.py 
python make_data_ReMeDi.py 
python make_data_VRBot.py 
```
3. Constructing the Dict

Download the  sougoupinyin medical dictionary and the medical dictionary THUOCL. 
```
python dict.py
```
4. Constructing the data for pretraining
```
cd data/data_for_pretrain, python medical_word_extract.py
```
5. pretraining
```
sh pretraining_deepspeed/run_train.sh
```
### finetuning

1. Download the MSL Dataset. 
```
cd data_processed, python data_process_MSL.py
```
2. training
```
nohup CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 19600 --nproc_per_node=1 train_Parallel.py > run.log 2>&1 &
```
3. evaluating
```
python evaluate.py
```
