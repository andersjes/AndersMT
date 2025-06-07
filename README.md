# AndersMT
This project is heavily inspired by the repository [luisnakayama/BRSET](https://github.com/luisnakayama/BRSET).  
Much of the evaluation pipeline and structure in this codebase builds on their work.  
We gratefully acknowledge their contribution to the open-source community.

---

To reproduce results or use pretrained models, weights can be obtained from the following sources:

- **RetFound Pretrained Weights**  
  - Source: [[ReTFound GitHub Repository](https://github.com/rmaphoh/RETFound_MAE) ] 

- **VisionFM Pretrained Weights**  
  - Source: [[VisionFM GitHub Repository](https://github.com/ABILab-CUHK/VisionFM) ]  

---


"internal_train_test.py" contains code for training and test models for BRSET (objective 1 to 6).

"external_evalution.py" contains code for external validation with mBRSET (objective 7).

"evaluationn.R" contains code for macro AUC-ROC, macro F1-score, PDI, ordinal PDI, calibration plot and confusion matrix.

"distribution_predicted_probabilities.R" contains code for distribution plot of predicted probabilities.

"bootstrap_AUC" contains code for confidence interval and test of significant of macro AUC-ROC (objective 1 and 2).

"bootstrap_F1" contains code for confidence interval and test of significant of macro F1-score (objective 1 and 2).
