rm(list=ls())
setwd('D:/Data Science/edWisor/Project 1')
getwd()

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "e1071")
lapply(x, require, character.only = TRUE)
library(psych)
library(scales)
library(ROCR)
library(caTools)

train_data = read.csv('./data/Train_data.csv', header = T)
test_data = read.csv('./data/Test_data.csv', header = T)

var = colnames(train_data)
var

#missing_val = data.frame(apply(train_data,2,function(x){sum(is.na(x))}))
#missing_val

str(train_data)

##Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(train_data)){
  
  if(class(train_data[,i]) == 'factor'){
    
    train_data[,i] = factor(train_data[,i], labels=(1:length(levels(factor(train_data[,i])))))
    
  }
}
str(train_data)


for(i in 1:ncol(test_data)){
  
  if(class(test_data[,i]) == 'factor'){
    
    test_data[,i] = factor(test_data[,i], labels=(1:length(levels(factor(test_data[,i])))))
    
  }
}
# For target variable Class 1 = False, Class 2 = True, changing it to 0 and 1 respectively

train_data$Churn = as.factor(gsub(1, 0, as.numeric(as.character(train_data$Churn))))
train_data$Churn = as.factor(gsub(2, 1, as.numeric(as.character(train_data$Churn))))
test_data$Churn = as.factor(gsub(1, 0, as.numeric(as.character(test_data$Churn))))
test_data$Churn = as.factor(gsub(2, 1, as.numeric(as.character(test_data$Churn))))

# Selecting Numeric Data

num_index  = sapply(train_data,is.numeric)

num_data = train_data[,num_index]

cnames = colnames(num_data)

for (i in 1:length(cnames)){
  assign(paste0("gn",i),ggplot(train_data, aes_string(x = cnames[i])) + 
           geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
           scale_y_continuous(breaks=pretty_breaks(n=10)) + 
           scale_x_continuous(breaks=pretty_breaks(n=10))+
           theme_bw() + xlab(cnames[i]) + ylab("Frequency") + ggtitle("Distribution Plot") +
           theme(text=element_text(size=20)))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,gn6,gn7,gn8,ncol=4)
gridExtra::grid.arrange(gn9,gn10,gn11,gn12,gn13,gn14,gn15,gn16,ncol=4)

train_col = c( "state","account.length", "international.plan","voice.mail.plan","number.vmail.messages","total.day.calls","total.day.charge","total.eve.calls" ,"total.eve.charge","total.night.calls","total.night.charge" ,"total.intl.calls","total.intl.charge","number.customer.service.calls","Churn")
train = train_data[train_col]
test = test_data[train_col]




# Splitting the training set in training & validation sets and oversampling the training set

mod.index = createDataPartition(train$Churn, p=0.80, list = FALSE)
mod_train = train[mod.index,]
mod_vald = train[-mod.index,]
summary(mod_train$Churn)
str(mod_train)
mod_train_over = SMOTE(Churn~.,mod_train,perc.over = 200, perc.under = 200)
summary(mod_train_over)


# Model Devlopment

########################## Decision Tree

C50_model = C5.0(Churn ~., mod_train_over, trials = 100, rules = TRUE)
summary(C50_model)

# Model Evaluation
C50_pred = predict(C50_model, mod_vald[,-15], type = "class")
conf_matrixC50 = table(mod_vald$Churn,C50_pred)
confusionMatrix(conf_matrixC50)


# Predicting on Test set

C50_test_pred = predict(C50_model, test[,-15], type = "class")
cm_c50_test = table(test$Churn,C50_test_pred)
confusionMatrix(cm_c50_test)

#Accuracy = 93.88
#FNR = 21.42


######################## Random Forest

RF_model = randomForest(Churn ~ ., mod_train_over, importance = TRUE, ntree = 200)


# Model Evaluation

RF_pred = predict(RF_model, mod_vald[,-15], type ="class")


conf_martrixRF = table(mod_vald$Churn,RF_pred)
confusionMatrix(conf_martrixRF)

# Predicting on Test set

RF_test_pred = predict(RF_model, test[,-15],type = "class")

cm_RF_test = table(test$Churn,RF_test_pred)
confusionMatrix(cm_RF_test)

#Accuracy = 87.88
#FNR = 18.30




######################### Logistic Regression

logit_model = glm(Churn ~ ., data = mod_train_over, family = "binomial")
summary(logit_model)

# Model Evaluation

logit_prob = predict(logit_model, newdata = mod_vald[,-15], type = "response")

# Deciding Decision Threshold using ROC Curve

pred_lr = prediction(logit_prob,mod_vald$Churn)
perf_lr = performance(pred_lr, "tpr", "fpr")
plot(perf_lr)

logit_pred = ifelse(logit_prob<0.4, 0, 1)
cm_logit = table(mod_vald$Churn,logit_pred)
confusionMatrix(cm_logit)

# Predicting on Test set

logit_test_prob = predict(logit_model, newdata = test[,-15], type= "response")
logit_test_pred = ifelse(logit_test_prob<0.4 , 0, 1)
cm_test_logit = table(test$Churn, logit_test_pred)
confusionMatrix(cm_test_logit)

#Accuracy = 73,25
#FNR = 31.69

###################### Kernel SVM
library(e1071)
svm_model = svm(Churn~., mod_train_over, kernel = 'radial', gamma = 0.001, cost = 10)


svm_pred = predict(svm_model, newdata = mod_vald[,-15])
cm_svm = table(mod_vald$Churn,svm_pred)
confusionMatrix(cm_svm)

# Test set Predictions

cm_test_pred = predict(svm_model, newdata = test[,-15])
cm_test_svm = table(test$Churn, cm_test_pred)
confusionMatrix(cm_test_svm)

#Accuracy = 83.14
#FNR = 50.89


############### XGBoost Classifier

#install.packages('xgboost')
library(xgboost)

# Data preperation for XG Boost
xgb_train = data.frame(lapply(mod_train_over, function(x) as.numeric(as.character(x))))
xgb_vald = data.frame(lapply(mod_vald, function(x) as.numeric(as.character(x))))
xgb_test = data.frame(lapply(test, function(x) as.numeric(as.character(x))))

XGBoost_model = xgboost(data = as.matrix(xgb_train[,-15]), label = xgb_train$Churn, nrounds = 18)

xgb_pred = predict(XGBoost_model, newdata = as.matrix(xgb_vald[,-15]))
xgb_pred = ifelse(xgb_pred<0.5, 0, 1)
cm_xgb = table(xgb_vald$Churn, xgb_pred)
confusionMatrix(cm_xgb)

# Test Set Predictions
xgb_test_pred = predict(XGBoost_model, newdata = as.matrix(xgb_test[,-15]))
xgb_test_pred = ifelse(xgb_test_pred<0.5, 0, 1)
cm_test_xgb = table(xgb_test$Churn, xgb_test_pred)
confusionMatrix(cm_test_xgb)


#Accuracy = 92.26
#FNR = 24.53








########### Random Forest seems to be working best