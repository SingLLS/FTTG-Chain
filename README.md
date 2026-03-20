#Dataset retrieval link：

https://www.kaggle.com/datasets/loveffc/hfbtp-a-blockchain-performance-dataset

https://www.kaggle.com/datasets/loveffc/blockchain-performance

#FTTG: A Hybrid framework of FT-Transformer and Graph Attention Network for Blockchain Performance Prediction and Optimization：

（1）Model training：python train_hybrid.py --dataset ./data/HFBTP.csv --max_epochs 200 --use_gpu

（2）Performance reasoning：python infer_hybrid.py --model modelH/Hybrid_FTT_RaftGAT.pth --arrival 100 --orderers 3 --block 50

（3）Optimal block recommendation：python recommend_block.py --model modelH/Hybrid_FTT_RaftGAT.pth --arrival 90 --orderers 9 --block_min 10 --block_max 800 --epsilon 0.05 --delta 0.05 --L_max 1.0 --beta 0.5
