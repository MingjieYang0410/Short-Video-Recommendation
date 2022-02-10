# Short-Video-Recommendation
### I. Guidance
Step 1: Build required repository by running:
```bash
sh ./scr/prepare_dir.sh
```
Step 2: Download the dataset and unzip the data files onto ./Data/original/
https://pan.baidu.com/s/1mw7Dq2YGO_Hytg64xK5rUA 
Code: 0d63

Step 3: Pre-train embeddings by running: 

(Time Warning: 30 minutes)
```bash
sh ./scr/pre_training.sh
```
Step  4: Training model by running:
```bash
python main.py
```
### II. Dependencies



###
###I. Description
This project aims at optimizing multiple 
tasks with a single model. 
We implemented two state-of-the-art models, MMOE and 
PLE 
