# Required Libraries
library(caret)
library(pROC)
library(dplyr)
library(ggplot2)
library(hrbrthemes)

# Set your working directory
setwd("C:/Users/ander/OneDrive - Aarhus universitet/Skrivebord/calibration")

# Load your data
df_overlap <- read.csv("y_resnet200d_fine_tune_3class_reproduced.csv")
df_no_overlap <- read.csv("resnet_100.csv")

# print columns names

# Swap y_test_0 with y_test_1 and y_prob_0 with y_prob_1
temp_test <- df_overlap$y_test_0
df_overlap$y_test_0 <- df_overlap$y_test_1
df_overlap$y_test_1 <- temp_test

temp_prob <- df_overlap$y_prob_0
df_overlap$y_prob_0 <- df_overlap$y_prob_1
df_overlap$y_prob_1 <- temp_prob

temp_test <- df_no_overlap$y_test_0
df_no_overlap$y_test_0 <- df_no_overlap$y_test_1
df_no_overlap$y_test_1 <- temp_test

temp_prob <- df_no_overlap$y_prob_0
df_no_overlap$y_prob_0 <- df_no_overlap$y_prob_1
df_no_overlap$y_prob_1 <- temp_prob

# Set the number of bootstrap samples
n_bootstrap <- 1000

# Bootstrap for df_overlap
bootstrap_overlap <- replicate(
  n_bootstrap,
  df_overlap[sample(nrow(df_overlap), replace = TRUE), ],
  simplify = FALSE
)

# Bootstrap for df_no_overlap
bootstrap_no_overlap <- replicate(
  n_bootstrap,
  df_no_overlap[sample(nrow(df_no_overlap), replace = TRUE), ],
  simplify = FALSE
)

# argmax function
get_argmax_labels <- function(df, prefix) {
  mat <- as.matrix(df[, paste0(prefix, 0:2)])
  max_indices <- max.col(mat) - 1  # class indices: 0, 1, 2
  return(as.factor(max_indices))
}

add_argmax_labels <- function(df) {
  y_true <- get_argmax_labels(df, "y_test_")
  y_pred <- get_argmax_labels(df, "y_prob_")
  
  df$y_true_class <- y_true
  df$y_pred_class <- y_pred
  return(df)
}


# Add argmax labels to each bootstrap sample
bootstrap_overlap_with_classes <- lapply(bootstrap_overlap, add_argmax_labels)
bootstrap_no_overlap_with_classes <- lapply(bootstrap_no_overlap, add_argmax_labels)

# print first 5 rows of the first bootstrap sample
head(bootstrap_overlap_with_classes[[1]], 5)
head(bootstrap_no_overlap_with_classes[[1]], 5)


# F1-score function

compute_macro_f1 <- function(df) {
  cm <- confusionMatrix(df$y_pred_class, df$y_true_class)
  f1_per_class <- cm$byClass[, "F1"]
  macro_f1 <- mean(f1_per_class, na.rm = TRUE)
  return(macro_f1)
}

# Compute F1-scores for each bootstrap sample
macro_f1_overlap <- sapply(bootstrap_overlap_with_classes, compute_macro_f1)
macro_f1_no_overlap <- sapply(bootstrap_no_overlap_with_classes, compute_macro_f1)

# print the first 5 rows of the F1-scores
head(macro_f1_overlap, 5)
head(macro_f1_no_overlap, 5)

# Create a combined data frame
macro_f1_df <- data.frame(
  value = c(macro_f1_overlap, macro_f1_no_overlap),
  type = rep(c("Overlap", "No Overlap"), each = length(macro_f1_overlap))
)

p <- ggplot(macro_f1_df, aes(x = value, fill = type)) +
  geom_histogram(color = "#e9ecef", alpha = 0.6, position = 'identity', bins = 30) +
  scale_fill_manual(values = c("#69b3a2", "#404080")) +
  theme_ipsum() +
  labs(title = "Distribution of Macro F1-scores",
       x = "Macro F1-score",
       y = "Count",
       fill = "")
print(p)

#calculate difference between overlap and no overlap
macro_f1_diff <- macro_f1_overlap - macro_f1_no_overlap
# print the first 5 rows of the macro F1-score difference
head(macro_f1_diff, 5)
# print histogram of the macro F1-score difference
hist(macro_f1_diff, main = "Macro F1-score Difference", xlab = "Macro F1-score Difference", breaks = 30)
# calculate the mean of the macro F1-score for overlap and no overlap
mean_macro_f1_overlap <- mean(macro_f1_overlap)
mean_macro_f1_no_overlap <- mean(macro_f1_no_overlap)
# print the mean of the macro F1-score for overlap and no overlap
cat("Mean Macro F1-score Overlap:", mean_macro_f1_overlap, "\n")
cat("Mean Macro F1-score No Overlap:", mean_macro_f1_no_overlap, "\n")


#calculate the mean and 95% confidence interval of the macro F1-score difference
mean_macro_f1_diff <- mean(macro_f1_diff)
ci_macro_f1_diff <- quantile(macro_f1_diff, c(0.025, 0.975))

# print the mean and 95% confidence interval of the macro F1-score difference
cat("Mean Macro F1-score Difference:", mean_macro_f1_diff, "\n")
cat("95% CI of Macro F1-score Difference:", ci_macro_f1_diff[1], ci_macro_f1_diff[2], "\n")

# print histogram of the macro F1-score difference
hist(macro_f1_diff, main = "Macro F1-score Difference", xlab = "Macro F1-score Difference", breaks = 30)

# Add mean line
abline(v = mean_macro_f1_diff, col = "blue", lwd = 2, lty = 2)

# Add CI lines
abline(v = ci_macro_f1_diff[1], col = "red", lwd = 2, lty = 3)
abline(v = ci_macro_f1_diff[2], col = "red", lwd = 2, lty = 3)

# Add legend
legend("topright",
       legend = c("Mean", "95% CI"),
       col = c("blue", "red"),
       lty = c(2, 3),
       lwd = 2,
       bty = "n")
