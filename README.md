# Short-Video-Recommendation

This project aims at optimizing seven tasks for short video 
recommendation. 

Tasks: Read Comment, Like, Click Avatar, Forward, Favorite, Comment, Follow

Keywords: MMOE, PLE, DeepWalk, TF-IDF + Truncated SVD, Doc2Vec

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

### III. References
Modeling Task Relationships in Multi-task Learning with
Multi-gate Mixture-of-Experts https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations https://dl.acm.org/doi/10.1145/3383313.3412236

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction https://www.ijcai.org/Proceedings/2017/0239.pdf