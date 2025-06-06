# Required Libraries
library(caret)
library(pROC)
library(dplyr)
library(ggplot2)
library(hrbrthemes)

# Load your data
df_overlap <- read.csv("path_to_reproduced_results.csv")
df_no_overlap <- read.csv("path_to_patient_stratified__results.csv")

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

# function of calculating AUC
compute_auc_per_class <- function(df) {
  aucs <- sapply(0:2, function(class_idx) {
    roc_obj <- roc(response = df[[paste0("y_test_", class_idx)]],
                   predictor = df[[paste0("y_prob_", class_idx)]],
                   quiet = TRUE)
    auc(roc_obj)
  })
  return(aucs)  # Returns a vector: [AUC_class_0, AUC_class_1, AUC_class_2]
}


# Compute AUCs for each bootstrap sample (returns a 1000 x 3 matrix)
auc_overlap_results <- t(sapply(bootstrap_overlap, compute_auc_per_class))
auc_no_overlap_results <- t(sapply(bootstrap_no_overlap, compute_auc_per_class))

# print the first 5 rows of the AUC results
head(auc_overlap_results, 5)
head(auc_no_overlap_results, 5)

# calculate macro AUC of each row
macro_auc_overlap <- rowMeans(auc_overlap_results, na.rm = TRUE)
macro_auc_no_overlap <- rowMeans(auc_no_overlap_results, na.rm = TRUE)

# print the first 5 rows of the macro AUC results
head(macro_auc_overlap, 5)
head(macro_auc_no_overlap, 5)

# Create a combined data frame
macro_auc_df <- data.frame(
  value = c(macro_auc_overlap, macro_auc_no_overlap),
  type = rep(c("Overlap", "No Overlap"), each = length(macro_auc_overlap))
)

p <- ggplot(macro_auc_df, aes(x = value, fill = type)) +
  geom_histogram(color = "#e9ecef", alpha = 0.6, position = 'identity', bins = 30) +
  scale_fill_manual(values = c("#69b3a2", "#404080")) +
  theme_ipsum() +
  labs(title = "Distribution of Macro AUCs",
       x = "Macro AUC",
       y = "Count",
       fill = "")
print(p)


#hist(macro_auc_overlap, main = "Macro AUC Overlap", xlab = "Macro AUC", breaks = 30)
#hist(macro_auc_no_overlap, main = "Macro AUC No Overlap", xlab = "Macro AUC", breaks = 30)

# calculate the mean of the macro AUC for overlap and no overlap
mean_macro_auc_overlap <- mean(macro_auc_overlap)
mean_macro_auc_no_overlap <- mean(macro_auc_no_overlap)
# print the mean of the macro AUC for overlap and no overlap
cat("Mean Macro AUC Overlap:", mean_macro_auc_overlap, "\n")
cat("Mean Macro AUC No Overlap:", mean_macro_auc_no_overlap, "\n")

# calculate the difference in macro AUC between overlap and no overlap
macro_auc_diff <- macro_auc_overlap - macro_auc_no_overlap

# print the first 5 rows of the macro AUC difference
head(macro_auc_diff, 5)

# calculate the mean and 95% confidence interval of the macro AUC difference
mean_macro_auc_diff <- mean(macro_auc_diff)
ci_macro_auc_diff <- quantile(macro_auc_diff, probs = c(0.025, 0.975))
names(ci_macro_auc_diff)
# print the mean and 95% confidence interval of the macro AUC difference
cat("Mean Macro AUC Difference:", mean_macro_auc_diff, "\n")
cat("95% Confidence Interval of Macro AUC Difference:", ci_macro_auc_diff, "\n")


# create histogram of the macro AUC difference
hist(macro_auc_diff, main = "Macro AUC Difference", xlab = "Macro AUC Difference", breaks = 30)

# Add mean line
abline(v = mean_macro_auc_diff, col = "blue", lwd = 2, lty = 2)

# Add CI lines
abline(v = ci_macro_auc_diff[1], col = "red", lwd = 2, lty = 3)
abline(v = ci_macro_auc_diff[2], col = "red", lwd = 2, lty = 3)

# Add legend
legend("topright",
       legend = c("Mean", "95% CI"),
       col = c("blue", "red"),
       lty = c(2, 3),
       lwd = 2,
       bty = "n")
