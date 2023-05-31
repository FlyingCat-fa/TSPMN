# TSPMN
Matching-based Term Semantics Pre-training for Spoken Patient Query Understanding, ICASSP 2023.

### pretraining

Download the MedDialog, kamed and ReMeDi Dataset. 

Preprocessing for pretraining:
```
python make_data_MedDialog.py 
python make_data_ReMeDi.py 
python make_data_VRBot.py 
```
Constructing the Dict. Download the sougoupinyin medical dictionary and the medical dictionary THUOCL. Then:
```
python dict.py
```
Constructing the data for pretraining.
```
cd data/data_for_pretrain, python medical_word_extract.py
```
Pretraining:
```
sh pretraining_deepspeed/run_train.sh
```
### finetuning

Download the MSL Dataset. Then:
```
cd data_processed, python data_process_MSL.py
```
Training:
```
nohup CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 19600 --nproc_per_node=1 train_Parallel.py > run.log 2>&1 &
```
Evaluating:
```
python evaluate.py
```
