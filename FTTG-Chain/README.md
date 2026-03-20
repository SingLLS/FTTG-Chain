\#Dataset retrieval link：

https://www.kaggle.com/datasets/loveffc/hfbtp-a-blockchain-performance-dataset

https://www.kaggle.com/datasets/loveffc/blockchain-performance



\#FTTG: A Hybrid framework of FT-Transformer and Graph Attention Network for Blockchain Performance Prediction and Optimization：

（1）Model training：python train\_hybrid.py  --dataset ./data/HFBTP.csv  --max\_epochs 200   --use\_gpu



（2）Performance reasoning：python infer\_hybrid.py --model modelH/Hybrid\_FTT\_RaftGAT.pth --arrival 100 --orderers 3   --block 50



（3）Optimal block recommendation：python recommend\_block.py  --model modelH/Hybrid\_FTT\_RaftGAT.pth   --arrival 90  --orderers 9  --block\_min 10  --block\_max 800  --epsilon 0.05   --delta 0.05  --L\_max 1.0  --beta 0.5

