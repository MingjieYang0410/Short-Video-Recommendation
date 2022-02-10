# Short-Video-Recommendation
### I. Guidance
**Step 1**: Build required repository by running:
```bash
sh ./scr/prepare_dir.sh
```


**Step 2**: Download the dataset and unzip the data files onto ./Data/original/
https://pan.baidu.com/s/1mw7Dq2YGO_Hytg64xK5rUA 
Code: 0d63


**Step 3**: Pre-train embeddings by running: 

(Time Warning: 30 minutes)
```bash
sh ./scr/pre_training.sh
```


**Step  4**: Training model by running:
```bash
python main.py
```


### II. Environment
Python 3.7

Tensorflow 2.3

Gensim 3.8.3 (Note: The latest version raises errors)

Scikit-learn 1.0.1

Pandas 1.3.5
