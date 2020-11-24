#Clearing the environment
rm(list=ls())

#Setting working directory
setwd("F:\\Heart_disease_final _")
dir()

# Loading librarieshbbmnbmbbnbmnbnbmbm
library(corrplot)
library(caTools)
library(glmnet)
library(class)
library(e1071)
library(rpart)
library(randomForest)
library(h2o)

#Reading the data
heart <- read.table("heart_failure_clinical_records_dataset.csv",sep=",",header=TRUE)

#-------------------------------------------------Exploratory Data Analysis------------------------------------------------------
# Shape of the data
dim(heart)
# Viewing data
View(heart)
# Structure of the data
str(heart)
# Variable names of the data
colnames(heart)

# Checking for missing value
sum(is.na(heart))

#-------------------------------------Outlier Analysis-------------------------------------#
# BoxPlots - Distribution and Outlier Check
boxplot(heart$age)
boxplot(heart$creatinine_phosphokinase)
boxplot(heart$ejection_fraction)
boxplot(heart$platelets)
boxplot(heart$serum_creatinine)
boxplot(heart$serum_sodium)

#Remove outliers using boxplot method
#Replace all outliers with median
summary(heart$creatinine_phosphokinase)
heart$creatinine_phosphokinase <- replace(heart$creatinine_phosphokinase,heart$creatinine_phosphokinase >1350,581.8)
summary(heart$ejection_fraction)
heart$ejection_fraction <- replace(heart$ejection_fraction,heart$ejection_fraction >65,38)
summary(heart$platelets)
heart$platelets <-(ifelse(is.na(heart$platelets),mean(heart$platelets,na.rm = TRUE),heart$platelets))
summary(heart$serum_creatinine)
heart$serum_creatinine <- replace(heart$serum_creatinine,heart$serum_creatinine>1.7,1.100)
summary(heart$serum_sodium)
heart$serum_sodium <- replace(heart$serum_sodium,heart$serum_sodium<125,136)

#Write into csv file
write.csv(heart,"heart_disease.csv")
setwd("F:\\Heart_disease_final _")
heart1 <- read.table("heart_disease.csv",sep=",",header=TRUE)

#-----------------------------------Feature Selection------------------------------------------#

## Correlation Plot 
heart_cor <- cor(heart1)
corrplot(heart_cor, method="circle")

# Kendal test for input as Categorical variable and output as numerical variable
res12 <- cor.test(heart1$DEATH_EVENT,heart1$age,method = "kendall")
res13 <- cor.test(heart1$DEATH_EVENT,heart1$creatinine_phosphokinase,method = "kendall")
res14 <- cor.test(heart1$DEATH_EVENT,heart1$ejection_fraction,method = "kendall")
res15 <- cor.test(heart1$DEATH_EVENT,heart1$platelets,method = "kendall")
res16 <- cor.test(heart1$DEATH_EVENT,heart1$serum_creatinine,method = "kendall")
res17 <- cor.test(heart1$DEATH_EVENT,heart1$serum_sodium,method = "kendall")
# Chi-Square test for input as categorical variables with output as categorical variables
chisq.test(heart1$DEATH_EVENT,heart1$anaemia)
chisq.test(heart1$DEATH_EVENT,heart1$diabetes)
chisq.test(heart1$DEATH_EVENT,heart1$high_blood_pressure)
chisq.test(heart1$DEATH_EVENT,heart1$sex)
chisq.test(heart1$DEATH_EVENT,heart1$smoking)
chisq.test(heart1$DEATH_EVENT,heart1$DEATH_EVENT)

# Pearson,Spearman test for input as numerical variables and output as numerical variables
res <- cor.test(heart1$age, heart1$age,method = "pearson")
res1 <- cor.test(heart1$age, heart1$creatinine_phosphokinase,method = "spearman",exact=FALSE)
res2 <- cor.test(heart1$age,heart1$ejection_fraction,method = "spearman",exact=FALSE)
res3 <- cor.test(heart1$age,heart1$platelets,method = "spearman",exact = FALSE)
res4 <- cor.test(heart1$age,heart1$serum_creatinine,method = "spearman",exact=FALSE)
res5 <- cor.test(heart1$age,heart1$serum_sodium,method ="spearman",exact = FALSE)
# kendall test for input as numerical variable and output as categorical variables
res6 <- cor.test(heart1$age,heart1$anaemia,method = "kendall")
res7 <- cor.test(heart1$age,heart1$diabetes,method = "kendall")
res8 <- cor.test(heart1$age,heart1$high_blood_pressure,method = "kendall")
res9 <- cor.test(heart1$age,heart1$sex,method = "kendall")
res10 <- cor.test(heart1$age,heart1$smoking,method = "kendall")
res11 <- cor.test(heart1$age,heart1$DEATH_EVENT,method = "kendall")

#------------------------------------------Model Development--------------------------------------------#
#Encoding the target variable as factor
heart1$DEATH_EVENT = factor(heart1$DEATH_EVENT,levels = c(0,1))

set.seed(123)
spl1 <- sample.split(heart1$age,SplitRatio = 0.8)
training_set1 <- subset(heart1,spl1 == TRUE)
testing_set1 <- subset(heart1,spl == FALSE)

set.seed(123)
spl <- sample.split(heart1$DEATH_EVENT,SplitRatio = 0.80)
training_set <- subset(heart1,spl == TRUE)
testing_set <- subset(heart1,spl == FALSE)

#Feature Scaling
training_set[,c(1,3,5,7:9)]<- scale(training_set[,c(1,3,5,7:9)])
testing_set[,c(1,3,5,7:9)] <- scale(testing_set[,c(1,3,5,7:9)])

#------------------------------------------Linear Regression-------------------------------------------#
# Building regression model on training set
reg <- lm((age)~ejection_fraction
          +serum_creatinine+
            DEATH_EVENT,data=training_set)

#Summary of linear regression model
summary(reg)

#Predicting linear regression model on testing set
y_pred <- predict(reg,data=testing_set)

#Plotting linear regression
par(mfrow=c(2,2))
plot(regressor)

#Calculating Residual sum of squares 
RSS <- sum( (regressor$residuals)^2 )

#--------------------------------------------Logistics Regression---------------------------------------#
#Selecting range of data
heart1 <- heart1[,2:14]
#Building logistic model on traing set
classifier <- glm(formula = DEATH_EVENT~age+
                    ejection_fraction+
                    serum_creatinine+time,family=binomial,
                  data=training_set)

#Summary of logistic regression on testing set
summary(classifier)

#Predicting logistic model on testing set
prob_pred <-predict(classifier,type = 'response',newdata = testing_set[-13])
y_pred <- ifelse(prob_pred > 0.5,1,0)

# Creating cofussion matrix of the model
cm <- table(testing_set[,13],y_pred)

#Calculating accuracy of the model
accuracy <- 100*sum(diag(cm))/sum(cm)

#--------------------------------------------KNN-------------------------------------------------#
#Building knn model on training set
y_pred1 <- knn(train = training_set[, c(1,5,8,12)],
               test = testing_set[, c(1,5,8,12)],
               cl = training_set[, 13],k = 6)

#Creating confusion matrix 
cm1 <- table(testing_set[, 13],y_pred1)

#Calculating accuracy
accuracy <- 100*sum(diag(cm1))/sum(cm1)

#--------------------------------------------svm---------------------------------------------------#
#Building svm model on training set
model <- svm(formula = DEATH_EVENT~.,data=training_set,type = 'C-classification',
             kernel = 'linear')

#Predicting model on testing set
pred <- predict(model,newdata = testing_set[,-13])

#Creating confusion matrix
cm2 <- table(testing_set[,13],pred)

#Calculating accuracy of the model
svm_accuracy <- 100*sum(diag(cm2))/sum(cm2)

#-------------------------------------------Naive Bayes---------------------------------------------#
#Building naive bayes model on training set
model_nb <- naiveBayes(x=training_set[c(-4,-11,-13)],y=training_set$DEATH_EVENT)

#Predicting model on testing set
pred_nb <- predict(model_nb,newdata = testing_set[,c(-4,-11,-13)])

#Creating confusion matrix
cm3 <- table(testing_set[, 13],pred_nb)

#Calculating accuracy of the model
nb_accuracy <- 100*sum(diag(cm3))/sum(cm3)

#------------------------------------------Decision Tree---------------------------------------------#
#Building model using training set
classifier <- rpart(formula=DEATH_EVENT~.,data=training_set)

#Predicting the model on testing set
pred1 <- predict(classifier,newdata = testing_set[-13],type = "class")

#Creating confusion matrix
cm <- table(testing_set[, 13],pred1)

#Calculating accuracy of the model
dt_accuracy <- 100*sum(diag(cm))/sum(cm)

#Ploting and texting decision tree diagram 
plot(classifier)
text(classifier)

#------------------------------------------Random Forest----------------------------------------------#
#Building model using training set
classifier1  <- randomForest(x=training_set[-13],y=training_set$DEATH_EVENT,ntree = 200)

#Predicting the model using testing set
rf_pred <- predict(classifier1,newdata=testing_set[-13])

#Creating confusion matrix
cm1 <- table(testing_set[,13],rf_pred)

#Calculating accuracy of the model
rf_accuracy <- 100*sum(diag(cm1))/sum(cm1)

#-----------------------------------------K means clustering------------------------------------------#
#Setting the seed of R's random number 
set.seed(6)
wcss <- vector()
for(i in 1:10)wcss[i] <- sum(kmeans(heart1,i)$withinss)
plot(1:10,wcss,type = "b",main =paste('clusters'),xlab ='number of clusters',ylab ='Wcss')

# appyling kmeans to our data set
set.seed(29)
kmeans <- kmeans(heart1,4,iter.max = 300,nstart = 10)

#-----------------------------------------Hierarical clustering---------------------------------------#
#Creating dendogram and plotting the same
dendogram <- hclust(dist(heart1,method = 'euclidean'),method = 'ward.D')
plot(dendogram,main = paste('dendogram'),xlab = "deaths",ylab = 'euclidean distance')

#Creating the model
hc <- hclust(dist(heart1,method = 'euclidean'),method = 'ward.D')
y_hc <- cutree(hc,3)

#-----------------------------------------ANN----------------------------------------------------------#
#connecting system to default available server
h2o.init(nthreads = -1)

#building the ANN model on training set
classifier <- h2o.deeplearning(y = 'DEATH_EVENT',
                               training_frame = as.h2o(training_set),
                               activation = 'Rectifier',
                               hidden = c(7,7),epochs = 100,
                               train_samples_per_iteration = -2)

#Predicting the ANN model on testing set
prob_pred <- h2o.predict(classifier,newdata = as.h2o(testing_set[-13]))
y_pred <- (prob_pred >0.5)
y_pred <- as.vector(y_pred)

#Creating confusion matrix
cm <- table(testing_set[,13],y_pred)

#Calculating accuracy of the model
ann_accuarcy <- 100*sum(diag(cm))/sum(cm)

#Shutting down the h2o instance running at this address
h2o.shutdown()