# Required Libraries
library(caret)
library(pROC)
library(dplyr)

# Set your working directory
setwd("C:/Users/ander/OneDrive - Aarhus universitet/Skrivebord/calibration")

# Load your data
df_overlap <- read.csv("convnext_100_reproduced.csv")
df_no_overlap <- read.csv("convnext_100.csv")

# Swap class 0 and class 1
swap_columns <- function(df) {
  tmp_test <- df$y_test_0; df$y_test_0 <- df$y_test_1; df$y_test_1 <- tmp_test
  tmp_prob <- df$y_prob_0; df$y_prob_0 <- df$y_prob_1; df$y_prob_1 <- tmp_prob
  return(df)
}
df_overlap <- swap_columns(df_overlap)
df_no_overlap <- swap_columns(df_no_overlap)

# === ROC (AUC) Tests ===
roc_overlap <- lapply(0:2, function(i) roc(df_overlap[[paste0("y_test_", i)]], df_overlap[[paste0("y_prob_", i)]]))
roc_no_overlap <- lapply(0:2, function(i) roc(df_no_overlap[[paste0("y_test_", i)]], df_no_overlap[[paste0("y_prob_", i)]]))
auc_overlap <- sapply(roc_overlap, function(r) as.numeric(auc(r)))
auc_no_overlap <- sapply(roc_no_overlap, function(r) as.numeric(auc(r)))
delong_tests <- mapply(function(r1, r2) roc.test(r1, r2, method = "delong"), roc_overlap, roc_no_overlap, SIMPLIFY = FALSE)

# === Bootstrap Macro AUC Difference ===
set.seed(123)  # For reproducibility

bootstrap_macro_auc_diff <- function(y_overlap, p_overlap, y_no, p_no, n_boot = 1000) {
  n <- length(y_overlap[[1]])
  diffs <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(1:n, replace = TRUE)
    auc_overlap <- sapply(1:3, function(j) auc(roc(y_overlap[[j]][idx], p_overlap[[j]][idx])))
    auc_no <- sapply(1:3, function(j) auc(roc(y_no[[j]][idx], p_no[[j]][idx])))
    diffs[i] <- mean(auc_overlap) - mean(auc_no)
  }
  ci <- quantile(diffs, c(0.025, 0.975))
  list(mean_diff = mean(diffs), ci = ci)
}
y_overlap_list <- lapply(0:2, function(i) df_overlap[[paste0("y_test_", i)]])
p_overlap_list <- lapply(0:2, function(i) df_overlap[[paste0("y_prob_", i)]])
y_no_list <- lapply(0:2, function(i) df_no_overlap[[paste0("y_test_", i)]])
p_no_list <- lapply(0:2, function(i) df_no_overlap[[paste0("y_prob_", i)]])
macro_auc_boot <- bootstrap_macro_auc_diff(y_overlap_list, p_overlap_list, y_no_list, p_no_list)

# === Bootstrap F1 Difference (Per Class & Macro) ===
set.seed(123)

bootstrap_f1_diff <- function(y_true, prob1, prob2, threshold = 0.5, n_boot = 1000) {
  n <- length(y_true)
  diffs <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(1:n, replace = TRUE)
    p1 <- ifelse(prob1[idx] > threshold, 1, 0)
    p2 <- ifelse(prob2[idx] > threshold, 1, 0)
    f1_1 <- F_meas(factor(p1), factor(y_true[idx]))
    f1_2 <- F_meas(factor(p2), factor(y_true[idx]))
    diffs[i] <- f1_1 - f1_2
  }
  ci <- quantile(diffs, c(0.025, 0.975))
  list(mean_diff = mean(diffs), ci = ci)
}

bootstrap_macro_f1_diff <- function(y_list1, p_list1, y_list2, p_list2, threshold = 0.5, n_boot = 1000) {
  n <- length(y_list1[[1]])
  diffs <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(1:n, replace = TRUE)
    f1_1 <- sapply(1:3, function(j) F_meas(factor(ifelse(p_list1[[j]][idx] > threshold, 1, 0)), factor(y_list1[[j]][idx])))
    f1_2 <- sapply(1:3, function(j) F_meas(factor(ifelse(p_list2[[j]][idx] > threshold, 1, 0)), factor(y_list2[[j]][idx])))
    diffs[i] <- mean(f1_1) - mean(f1_2)
  }
  ci <- quantile(diffs, c(0.025, 0.975))
  list(mean_diff = mean(diffs), ci = ci)
}

f1_boot_per_class <- lapply(0:2, function(i) {
  bootstrap_f1_diff(
    y_true = df_overlap[[paste0("y_test_", i)]],
    prob1 = df_overlap[[paste0("y_prob_", i)]],
    prob2 = df_no_overlap[[paste0("y_prob_", i)]]
  )
})
macro_f1_boot <- bootstrap_macro_f1_diff(y_overlap_list, p_overlap_list, y_no_list, p_no_list)

# === Summary Table ===
summary_table <- data.frame(
  Class = c("Class 0", "Class 1", "Class 2", "Macro"),
  AUC_Overlap = c(auc_overlap, mean(auc_overlap)),
  AUC_No_Overlap = c(auc_no_overlap, mean(auc_no_overlap)),
  AUC_p_value = c(sapply(delong_tests, function(x) x$p.value), NA),
  AUC_boot_CI = c(rep(NA, 3), paste0(round(macro_auc_boot$ci[1], 3), " to ", round(macro_auc_boot$ci[2], 3))),
  F1_boot_diff = c(sapply(f1_boot_per_class, function(x) round(x$mean_diff, 3)), round(macro_f1_boot$mean_diff, 3)),
  F1_CI = c(
    sapply(f1_boot_per_class, function(x) paste0(round(x$ci[1], 3), " to ", round(x$ci[2], 3))),
    paste0(round(macro_f1_boot$ci[1], 3), " to ", round(macro_f1_boot$ci[2], 3))
  )
)

print(summary_table)
