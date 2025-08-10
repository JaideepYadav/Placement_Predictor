
#Student Academic and Placement Dataset
data<-read.csv(file.choose())
View(data)
head(data)

sum(is.na(data))

# Filling missing values 
data$gpa[is.na(data$gpa)] <- median(data$gpa, na.rm = TRUE)
str(data)
data$specialisation <- as.factor(data$specialisation)
data$specialisation <- as.numeric(data$specialisation) #datascience=1, fullstack=2
sum(is.na(data$salary))
data$salary[is.na(data$salary)]<-0;

#splitting the data
set.seed(123)
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))
train <- data[train_indices, ] 
test <- data[-train_indices, ] 

#predi salary using multiregresssion

model <- lm(salary ~ High.School + Secondary.School + Verbal.Ability + Internship_Exp +
              Programming + specialisation + gpa + Analytical.Skill, data = train)

# View the model summary
summary(model)   
predicted_salary <- predict(model, newdata = test)

# Actual salaries from the test dataset
actual_salary <- test$salary
new_input <- data.frame(High.School = 67,Secondary.School = 91,Verbal.Ability = 58,Internship_Exp = "No",         
  Programming = 55,specialisation = 1,gpa = 6.88,Analytical.Skill=58.8)
predicted_salary <- predict(model, newdata = new_input)
predicted_salary
# Residuals
residuals <- actual_salary - predicted_salary

# **********************************************************************************************
#  Decision Tree Model
library(rpart)
library(rpart.plot)
data$status <- as.factor(data$status)
set.seed(123)  
sample <- sample(1:nrow(data), 0.8 * nrow(data))  
train <- data[sample, ]
test <- data[-sample, ]


decision_tree <- rpart(
  status ~ High.School + Secondary.School + Verbal.Ability + Internship_Exp +  Programming + specialisation + gpa + Analytical.Skill,data = train,method = "class")
rpart.plot(decision_tree, main = "Decision Tree for Placement Prediction")

predicted_status <- predict(decision_tree, newdata = test, type = "class")
library(caret)
confusion_matrix <- table(Predicted = predicted_status, Actual = test$status)
acc2<-sum(diag(confusion_matrix)) / sum(confusion_matrix)
acc2<-acc2*100
acc2       #75%

#***************************************************************************************************

#  Naive Bayes
library("e1071")
data$status <- as.factor(data$status)
set.seed(123)
sample<-sample(1:nrow(data),.8*nrow(data))
train<-data[sample,]
test<-data[-sample,]

nb_model <- naiveBayes(status ~ High.School + Secondary.School + Verbal.Ability + Internship_Exp + Programming + specialisation + gpa + Analytical.Skill, data = train)
predicted_status <- predict(nb_model, newdata = test)
head(predicted_status)

confusion_matrix <- table(Predicted = predicted_status, Actual = test$status)
print(confusion_matrix)
acc3<-sum(diag(confusion_matrix))/sum(confusion_matrix)
acc3<-acc3*100
acc3       ##  83%
new_input <- data.frame(High.School = 85,Secondary.School = 88,Verbal.Ability = 70,Internship_Exp = 1,Programming = 80,specialisation = 1,gpa = 8.5,Analytical.Skill = 75
)
placement_prediction <- predict(nb_model, newdata = new_input)
print(placement_prediction)


#****************************************************************************************************

#Kmeans Clustering

install.packages("arules")
install.packages("cluster")

library(arules)
library(cluster)


#  a normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Prepare the data by removing irrelevant columns not having numerical value
data2 <- data[,-1]  

data2 <- data2[,-1] 
data2<-data2[,-4]
data2<-data2[,-5]
data2<-data2[,-7]
View(data2)

data_normalized <- as.data.frame(lapply(data2, normalize))
View(data_normalized)

set.seed(250)  
kme <- kmeans(data_normalized, centers = 2, nstart = 20)  # Set centers to 2 for binary clustering

print(kme)
print(kme$cluster)   
print(kme$centers)   

conf <- table(data$status, kme$cluster)
print(conf)
acc4 <- sum(diag(conf)) / sum(conf)
acc4<-acc4*100
acc4         ## 63%
library(ggplot2)
ggplot(data_normalized, aes(x = High.School, y = Secondary.School, color = as.factor(kme$cluster))) +
geom_point(size = 3) +labs(title = "K-Means Clustering Visualization",x = "High School Marks",y = "12th marks")
#*********************************************

model_results <- data.frame(Model = c("Multilinear Regression", "Decision Tree", "Naive Bayes", "K-Means Clustering"),Metric = c(45, acc2, acc3,acc4))
ggplot(model_results, aes(x = Model, y = Metric, fill = Model)) +
geom_bar(stat = "identity", width = 0.6) +labs(title = "Model Comparison by Accuracy", x = "Models",
y = "Acuracy")
  



