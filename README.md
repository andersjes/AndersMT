# AndersMT

This project builds upon and adapts code and concepts from the BRSET repository by luisnakayama. In particular, the implementations training and testing models. Many thanks to the authors for making their code openly available.

"internal_train_test.py" contains code for training and test models for BRSET (objective 1 to 6).
"external_evalution.py" contains code for external validation with mBRSET (objective 7).
"evaluationn.R" contains code for macro AUC-ROC, macro F1-score, PDI, ordinal PDI, calibration plot and confusion matrix
"distribution_predicted_probabilities.R" contains code for distribution plot of predicted probabilities.
"bootstrap_AUC" contains code for confidence interval and test of significant of macro AUC-ROC (objective 1 and 2)
"bootstrap_F1" contains code for confidence interval and test of significant of macro F1-score (objective 1 and 2)
