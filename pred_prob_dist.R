# Load required packages
library(ggplot2)
library(tidyr)
library(dplyr)

# Set working directory (adjust path if needed)
setwd("C:/Users/ander/OneDrive - Aarhus universitet/Skrivebord/calibration")

# Load dataset
df <- read.csv("resnet_80_mBRSET.csv")

# Swap y_test_0 with y_test_1 and y_prob_0 with y_prob_1
temp_test <- df$y_test_0
df$y_test_0 <- df$y_test_1
df$y_test_1 <- temp_test

temp_prob <- df$y_prob_0
df$y_prob_0 <- df$y_prob_1
df$y_prob_1 <- temp_prob

# Show number of observations in each class (true labels)
cat("True class distribution:\n")
print(table(df$y_test_0))
print(table(df$y_test_1))
print(table(df$y_test_2))

# Show summary of predicted probabilities
cat("\nPredicted probability summaries:\n")
print(summary(df$y_prob_0))
print(summary(df$y_prob_1))
print(summary(df$y_prob_2))

# Clip extreme values (0 or 1) to avoid issues in log or plots
clip_probs <- function(p) {
  p[p == 0] <- 0.0001
  p[p == 1] <- 0.9999
  return(p)
}
df$y_prob_0 <- clip_probs(df$y_prob_0)
df$y_prob_1 <- clip_probs(df$y_prob_1)
df$y_prob_2 <- clip_probs(df$y_prob_2)

# Reshape data for plotting
df_long <- df %>%
  pivot_longer(cols = starts_with("y_prob_"),
               names_to = "Class",
               values_to = "Probability") %>%
  mutate(Class = factor(Class,
                        levels = c("y_prob_0", "y_prob_1", "y_prob_2"),
                        labels = c("No DR", "NPDR", "PDR")))

# Create the plot with custom colors, larger text, no legend, and fixed x-axis range
p <- ggplot(df_long, aes(x = Probability, fill = Class)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("No DR" = "darkgreen", "NPDR" = "orange", "PDR" = "darkred")) +
  scale_x_continuous(limits = c(0, 1)) +  # FIXED X-AXIS RANGE
  labs(title = "ResNet-200d (80%)",
       x = "Predicted Probability",
       y = "Density") +
  theme_minimal() +
  theme(
    text = element_text(size = 16),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.position = "none",
    plot.title = element_text(size = 20, face = "bold")
  )

# Print the plot
print(p)
