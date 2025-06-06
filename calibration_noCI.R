library(CalibrationCurves)
library(pROC)
library(mcca)
library(Hmisc)
library(caret)
library(ggplot2)

setwd("C:/Users/ander/OneDrive - Aarhus universitet/Skrivebord/calibration")

# Load dataset
df <- read.csv("visionfm_100.csv")

# Swap y_test_0 with y_test_1 and y_prob_0 with y_prob_1
temp_test <- df$y_test_0
df$y_test_0 <- df$y_test_1
df$y_test_1 <- temp_test

temp_prob <- df$y_prob_0
df$y_prob_0 <- df$y_prob_1
df$y_prob_1 <- temp_prob

# Clip probabilities to avoid extremes
clip_probs <- function(p) {
  p[p == 0] <- 0.0001
  p[p == 1] <- 0.9999
  return(p)
}
df$y_prob_0 <- clip_probs(df$y_prob_0)
df$y_prob_1 <- clip_probs(df$y_prob_1)
df$y_prob_2 <- clip_probs(df$y_prob_2)

# Calibration Curves
class0 <- val.prob.ci.2(df$y_prob_0, df$y_test_0)
class1 <- val.prob.ci.2(df$y_prob_1, df$y_test_1)
class2 <- val.prob.ci.2(df$y_prob_2, df$y_test_2)

cl0xy <- class0$CalibrationCurves$FlexibleCalibration
cl1xy <- class1$CalibrationCurves$FlexibleCalibration
cl2xy <- class2$CalibrationCurves$FlexibleCalibration

# Plot
par(cex.lab = 1.5, cex.axis = 1.3, cex.main = 1.8, mar = c(5, 5, 4, 2) + 0.1)
plot(c(cl0xy$x), c(cl0xy$y), type = 'l', col = 'darkgreen', lwd = 2, 
     xlim = c(0, 1), ylim = c(0, 1), 
     xlab = 'Predicted Probability', 
     ylab = 'Observed Proportion',
     main = 'VisionFM (100%)')
lines(c(cl1xy$x), c(cl1xy$y), col = 'orange', lwd = 2)
lines(c(cl2xy$x), c(cl2xy$y), col = 'darkred', lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "black")

# AUC
df$true_labels <- apply(df[, c("y_test_0", "y_test_1", "y_test_2")], 1, function(x) which(x == 1) - 1)
true_labels <- df$true_labels

roc0 <- roc(true_labels == 0, df$y_prob_0)
roc1 <- roc(true_labels == 1, df$y_prob_1)
roc2 <- roc(true_labels == 2, df$y_prob_2)

auc0 <- roc0$auc
auc1 <- roc1$auc
auc2 <- roc2$auc
macro_auc <- mean(c(auc0, auc1, auc2))

# PDI
one_hot_label <- df[, c("y_test_0", "y_test_1", "y_test_2")]
label <- colnames(one_hot_label)[apply(one_hot_label, 1, which.max)]
data <- df[, c("y_prob_0", "y_prob_1", "y_prob_2")]
pdi <- pdi(y = label, d = data, method = "prob")
pdi_value <- pdi$measure

# Ordinal PDI
ordinal_pdi <- function(true_labels, predicted_probs) {
  class_ranks <- 0:(ncol(predicted_probs) - 1)
  expected_scores <- predicted_probs %*% class_ranks
  
  concordant <- 0
  denom <- 0
  n <- length(true_labels)
  
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      d <- abs(true_labels[i] - true_labels[j])
      if (d == 0) next
      correct_order <- sign(true_labels[i] - true_labels[j])
      predicted_order <- sign(expected_scores[i] - expected_scores[j])
      concordant <- concordant + d * (correct_order == predicted_order)
      denom <- denom + d
    }
  }
  return(concordant / denom)
}

# Normalize predicted probabilities
prob_matrix <- as.matrix(df[, c("y_prob_0", "y_prob_1", "y_prob_2")])
prob_matrix <- prob_matrix / rowSums(prob_matrix)
ordinal_pdi_value <- ordinal_pdi(df$true_labels, prob_matrix)

# Macro F1
df$predicted_class <- apply(prob_matrix, 1, which.max) - 1
conf_matrix <- confusionMatrix(factor(df$predicted_class), factor(df$true_labels))

precision <- conf_matrix$byClass[, "Precision"]
recall <- conf_matrix$byClass[, "Recall"]
f1_scores <- 2 * (precision * recall) / (precision + recall)
macro_f1 <- mean(f1_scores, na.rm = TRUE)

# Create Results Table
results_table <- data.frame(
  Metric = c("Macro-AUC", "PDI", "Ordinal PDI", "Macro F1 Score"),
  Value = c(macro_auc, pdi_value, ordinal_pdi_value, macro_f1)
)
print(results_table)

# Confusion Matrix Plot
conf_matrix_table <- as.data.frame(conf_matrix$table)
colnames(conf_matrix_table) <- c("Predicted", "Actual", "Frequency")
label_map <- c("0" = "No DR", "1" = "NPDR", "2" = "PDR")
conf_matrix_table$Predicted <- factor(conf_matrix_table$Predicted, levels = names(label_map), labels = label_map)
conf_matrix_table$Actual <- factor(conf_matrix_table$Actual, levels = names(label_map), labels = label_map)

conf_matrix_plot <- ggplot(conf_matrix_table, aes(x = Predicted, y = Actual, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "VisionFM (100%)", x = "Predicted Class", y = "Actual Class") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  )
print(conf_matrix_plot)

