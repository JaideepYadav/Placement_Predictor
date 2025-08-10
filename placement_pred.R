# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)

# Load dataset
data <- read.csv(file.choose())
View(data)

# Check structure and missing values
str(data)
sum(is.na(data))

# Handle missing values
data$gpa[is.na(data$gpa)] <- median(data$gpa, na.rm = TRUE)
data$salary[is.na(data$salary)] <- 0

# Convert categorical columns to numeric/factor
data$specialisation <- as.factor(data$specialisation) # Data Science=1, Full Stack=2
data$specialisation <- as.numeric(data$specialisation)
data$Internship_Exp <- as.factor(data$Internship_Exp)
data$Internship_Exp <- as.numeric(data$Internship_Exp)

# Convert status to factor for classification
data$status <- as.factor(data$status)

# Split data into training and testing
set.seed(123)
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))
train <- data[train_indices, ]
test <- data[-train_indices, ]

# Scale numeric features and store means/stds
numeric_features <- c("High.School", "Secondary.School", "Verbal.Ability", "Programming", "gpa", "Analytical.Skill")

# Compute means and stds for training data
feature_means <- apply(train[, numeric_features], 2, mean)
feature_sds <- apply(train[, numeric_features], 2, sd)

# Scale numeric columns
train[numeric_features] <- scale(train[numeric_features])
test[numeric_features] <- scale(test[numeric_features], center = feature_means, scale = feature_sds)

# Decision Tree Model for Placement Prediction
decision_tree <- rpart(
  status ~ High.School + Secondary.School + Verbal.Ability + Internship_Exp + Programming +
    specialisation + gpa + Analytical.Skill,
  data = train,
  method = "class"
)

# Visualize the tree
rpart.plot(decision_tree, main = "Decision Tree for Placement Prediction")

# Evaluate the model
predicted_status <- predict(decision_tree, newdata = test, type = "class")
confusion_matrix <- table(Predicted = predicted_status, Actual = test$status)
acc2 <- sum(diag(confusion_matrix)) / sum(confusion_matrix) * 100
cat("Decision Tree Accuracy:", acc2, "%\n")

# Predict placement probability for a new candidate
predict_placement <- function(high_school, secondary_school, verbal, intern, program, special, cgpa, analytics, model) {
  # Standardize new inputs using training data statistics
  new_data <- data.frame(
    High.School = (high_school - feature_means["High.School"]) / feature_sds["High.School"],
    Secondary.School = (secondary_school - feature_means["Secondary.School"]) / feature_sds["Secondary.School"],
    Verbal.Ability = (verbal - feature_means["Verbal.Ability"]) / feature_sds["Verbal.Ability"],
    Internship_Exp = as.numeric(intern),  # Convert internship experience to numeric
    Programming = (program - feature_means["Programming"]) / feature_sds["Programming"],
    specialisation = as.numeric(special),  # Ensure specialisation is numeric
    gpa = (cgpa - feature_means["gpa"]) / feature_sds["gpa"],
    Analytical.Skill = (analytics - feature_means["Analytical.Skill"]) / feature_sds["Analytical.Skill"]
  )
  
  # Predict placement probability
  prob <- predict(model, newdata = new_data, type = "prob")[, 2]
  
  return(paste("Placement chances:", round(prob * 100, 2), "%"))
}

# Example Usage
result <- predict_placement(
  high_school = 80,
  secondary_school = 80,
  verbal = 90,
  intern = 0,  # Assuming 1 = Yes, 0 = No
  program = 89
  ,
  special = 1,  # Assuming 1 = Data Science, 2 = Full Stack
  cgpa = 9.1,
  analytics =89,
  model = decision_tree
)

print(result)
