# preprocess training data further to come to Input for different algorithms

###### Joran Lokkerbol; data exploration - sept 2019
###### Adapted by: Dirk Geurts, Gijs van de Veen, Fiona Zegwaard - nov 2020
options(scipen=999)
rm(list = ls())  # Delete everything that is in R's memory
set.seed(12345)


##############################
#NOTES
##############################

# Three prediction models will be created:
#   1: predicting whether subjects deteriorate in terms of depressive symptoms
#   2: predicting whether subjects will drop out
#   3: predicting whether subjects will either deteriorate and/or will drop out
# 
# For each model the following steps will be followed:
#   A. Prepare training data 
#       a. make appropriate outcome vector Y
#           I. print pivottable to determine base-rate / positive predictive value
#       b. select appropriate case.names
#   B. tune a variety of ML models 
#       a. Regression models 
#           I.  Lasso 
#           II. Ridge
#           III.Elastic net
#           IV. Interim conclusion
#         
#       b. Tree-based models 
#           I.  simple tree
#           II. random forrest
#           III.extreme gradient boosting)
#       c. explore in winning models a/b whether internal upsampling works ('smote')
#       d. use best tuning parameters of all models in SuperLearner
#   C. vizualize main outcomes
#   D. Test best model (superlearner? or best a./b.) on test_data
#   E. Visualize main outcomes test data
#   F. draw conclusions
  


# 0: set up and load traindata ------------------------------------------------
   
   

# ## go to Jorans GitHub to find optimizations other than ROC positive predictive value
# 
# install.packages("rmarkdown")
# install.packages("psych")
# install.packages("ggplot2")
# install.packages("summarytools")
# install.packages("corrplot")
# install.packages("purrr")
# install.packages("tidyr")
# install.packages("caret")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("rattle")
# install.packages("dplyr")
# install.packages("DescTools")
# install.packages("PerformanceAnalytics")
# install.packages("caret")
# install.packages("xgboost") #ERROR
# install.packages("SuperLearner")
# 
# 
# # Necessary for packages to work in DRE
# install.packages("tmvnsim", dependencies=TRUE)
# install.packages("bitops", dependencies=TRUE)
# install.packages("gld", dependencies=TRUE)
# install.packages("Exact", dependencies=TRUE)
# install.packages("rootSolve", dependencies=TRUE)
# install.packages('ISLR')
# install.packages('rpart')
# install.packages('rpart.plot')
# install.packages('plotROC')
# install.packages('DMwR')
# install.packages('MLeval')
# install.packages('arsenal')
# install.packages('ranger')
# install.packages('arm')
# install.packages("Hmisc")
# install.packages("bartMachine")
# install.packages("MLmetrics")
# install.packages("ROCR")
# install.packages("PRROC")

# for R/Rstudio/Markdown/visualization/organisation
library(rmarkdown) #ERROR
    library(tmvnsim)
library(psych)
library(ggplot2)
library(corrplot)
library(purrr)
library(tidyr)
library(dplyr)

# Machine learning
library(caret)
library(rpart)
library(rpart.plot)
    library(bitops)
library(rattle)
library(gld)
    library(Exact)
    library(rootSolve)
library(broom)
library(xgboost) #ERROR
library(SuperLearner)
library(glmnet)
library(SuperLearner)
  library(ranger)
  library(arm)
  library(Hmisc)
  library(bartMachine)

# visualiation of ML
library(plotROC)
library(ISLR)
library(grid)
library(DMwR)
library(MLeval)
library(DescTools)
library(PerformanceAnalytics)
library(MLmetrics)

#tableby
library(arsenal)
require(knitr)
require(survival)


library(ROCR)
library(PRROC)

setwd("Z:/inbox/WorkInProgress") #set your working directory
# load train data
df_train <- read.csv("Z:/inbox/WorkInProgress/Werkdatabestand/df_trainDO")

# A. Prepare training data ---------------------------------------------------


## Impute data #################################################################
select <- dplyr::select
# for demographics add NA group --> No NAs anymore at this point (group for missing)

# for questionnaires impute rounded median
df_train %>% select(starts_with("Q.")) %>% summarise_all(funs(sum(is.na(.))))

# df_train <- df_train %>% transmute_if(is.integer, funs(as.numeric(as.integer(.))))
str(df_train)
medians.Q <- df_train%>% select(starts_with("Q.")) %>% summarise(across(everything(),~ median(., na.rm =TRUE)))

df_train <- df_train %>% mutate_at(vars(starts_with("Q.")), ~ifelse(is.na(.), median(., na.rm =TRUE,), .))
df_train %>% select(starts_with("Q.")) %>% summarise_all(funs(sum(is.na(.))))

# for DSM imputation NA --> no / 0 
df_train %>% select(starts_with("DSM.")) %>% summarise_all(funs(sum(is.na(.))))
df_train <- df_train %>% mutate_at(vars(starts_with("DSM.")), ~ifelse(is.na(.),0,.))

# for Dem. imputation NA --> no / 0 
df_train %>% select(starts_with("Dem.")) %>% summarise_all(funs(sum(is.na(.))))
df_train <- df_train %>% mutate_at(vars(starts_with("Dem.")), ~ifelse(is.na(.),0,.))

# for Clin
# df_train %>% select(starts_with("Clin")) %>% str()
df_train %>% select(starts_with("Clin.")) %>% summarise_all(funs(sum(is.na(.))))
df_train <- df_train %>% mutate_at(vars(starts_with("Clin."),-starts_with("Clin.Intox.Alcohol")), ~ifelse(is.na(.),0,.))
medians.alcohol <- median(df_train$Clin.Intox.Alcohol)
df_train <- df_train %>% mutate_at(vars(starts_with("Clin.Intox.Alcohol")), ~ifelse(is.na(.), median(., na.rm =TRUE,), .))

# Check near zero variance
## Identify near-zero variances predictors
# nzvar <- nearZeroVar(Input, saveMetrics = TRUE)
# nzvarnm <- row.names(nzvar[nzvar$nzv,])

# remove near-zero variancd variables --> DSM.ASD and Clin.Med.lithium
# for (i in nzvarnm){
#   Input <- Input%>%select(-c(i))}

# Select predictors
Input <- df_train%>%select((starts_with("DSM.")|starts_with("Clin.")|starts_with("Q.")|starts_with("Dem."))&
                             -starts_with("ID")&
                             -starts_with("DSM.ASD")&
                             -starts_with("Clin.Med.Lithium")&
                             -starts_with("Y") )

## a. make appropriate outcome vector Y ========================================

# We need 3 outcome vectors for the three different models
#   1: predicting whether subjects deteriorate in terms of depressive symptoms
#   2: predicting whether subjects will drop out
#   3: predicting whether subjects will either deteriorate and/or will drop out
#           I. print pivottable to determine base-rate / positive predictive value
#       b. select appropriate case.names

# Select output and cases for 1.Depr ############################################

# construct output
InputDeprb <- Input
YAbs <- df_train$Q.BDI.total.pre - df_train$Qout.BDI.total.post
YRel <- YAbs/df_train$Q.BDI.total.pre
YRel0 <- vector(mode="logical", length = length(YRel))
#YRel0[df_train$Q.BDI.total.pre<13& !is.na(YAbs)] <- YAbs[df_train$Q.BDI.total.pre<13 & !is.na(YAbs)]<(-3) # everyone who does not score 'depressed', 9 point in symptom as relevant, arbitrary. 
YRelAll <- YRel<(-.14)
YRelAll[df_train$Q.BDI.total.pre<13] <- YAbs[df_train$Q.BDI.total.pre<13]<(-3)# based on Button et al, psych med, 2016 MCI difference on the BDI-II - according to the patient's perspective
InputDeprb$Y <- as.factor(YRelAll)
levels(InputDeprb$Y) <- c("worse","SameBetter")



# select data: only those with pre and post measurement
InputDepr <- InputDeprb[  !is.na(df_train$Q.BDI.total.pre) & 
                            !is.na(df_train$Qout.BDI.total.post) & !is.na(InputDeprb$Y)&!is.na(YAbs),]
YAbssemitest <- YAbs[  !is.na(df_train$Q.BDI.total.pre) & 
                         !is.na(df_train$Qout.BDI.total.post) & !is.na(InputDeprb$Y)&!is.na(YAbs)]
YRelAll <- YRelAll[  !is.na(df_train$Q.BDI.total.pre) & 
                       !is.na(df_train$Qout.BDI.total.post) & !is.na(InputDeprb$Y)&!is.na(YAbs)]

ggplot(data = InputDepr,aes(x=Q.BDI.total.pre, y=YAbssemitest, group = YRelAll)) +
  geom_point(aes(color = YRelAll))

print('base PPV =')
Freq(InputDepr$Y)

# Select output andcases for 2. DO ######################################################
# select data: all participants for whom numbses is known ###############################
InputDO <- Input[!is.na(df_train$Numbses),]

# construct output
InputDO$Y <- as.factor(df_train$Numbses<5)
levels(InputDO$Y) <- c("worse","In")

print('base PPV =')
Freq(InputDO$Y)

# Select output andcases for 3.DoDepr ###################################################

# construct output
InputDODepr <- Input
InputDODepr$Y <- as.factor(YRelAll | df_train$Numbses<5)
levels(InputDODepr$Y) <- c("worse","AdviseIn")

# select data: all participants for whom numbses <5 OR those for whom pre and post BDI is known
InputDODepr <- InputDODepr[df_train$Numbses<5|
                       (!is.na(df_train$Q.BDI.total.pre) & !is.na(df_train$Qout.BDI.total.post))&
                         !is.na(InputDODepr$Y),]

print('base PPV =')
Freq(InputDODepr$Y)


## Exploratory datavisualisation ##############################################


# Check for outliers with correlation matrix for numeric predictors
InputDODepr %>% select(Q.OQ.symptomdistress, Q.OQ.socialrole, Q.OQ.interpersonalrelations,
                 Q.BDI.cognitive, Q.BDI.somatic, Q.BDI.affective,
                 Q.PSWQ.worry, Q.PSWQ.AbsenceOfWorry,
                 Q.FFMQ.observe,Q.FFMQ.describe,Q.FFMQ.actawareness,Q.FFMQ.nonjudging,Q.FFMQ.nonreactivity,
                 Q.SCS.selfjudgement,Q.SCS.commonhumanity,Q.SCS.mindfulness,
                 Q.SCS.overidentified,Q.SCS.selfkindness,Q.SCS.isolation,
                 Dem.age, Clin.Intox.Alcohol)%>%
  chart.Correlation(.,histogram=TRUE, method = 'pearson', pch=19)


# assess whether there are obvious differences between the outcome groups

## create a vecor specifying the variable names:
myvars <- names(InputDepr)
myvars <- myvars[myvars!="Y"]
## select all except Y and paste them together with the + sign
IndVars <- paste(myvars, collapse="+")

tab1.Depr <- tableby(as.formula(paste('Y~',IndVars)),data=InputDepr)
df <- as.data.frame(tab1.Depr)
print(df[df$p.value<.05,])

tab1.DO <- tableby(as.formula(paste('Y~',IndVars)),data=InputDO)
df <- as.data.frame(tab1.DO)
print(df[df$p.value<.05,])

tab1.DODepr <- tableby(as.formula(paste('Y~',IndVars)),data=InputDODepr)
df <- as.data.frame(tab1.DODepr)
print(df[df$p.value<.05,])






# Construct metric for loss function ML: PPV ##################################

ppvSummary <- function (data, lev = NULL, model = NULL) {
  tn <- table(data$pred == "worse",data$obs=='worse')[1]
  fp <- table(data$pred == "worse",data$obs=='worse')[2]
  fn <- table(data$pred == "worse",data$obs=='worse')[3]
  tp <- table(data$pred == "worse",data$obs=='worse')[4]
  if(is.na(tp)){tp <- 0}
  if(is.na(fn)){fn <- 0}
  if(is.na(fp)){fp <- 0}
  if(is.na(tn)){tn <- 0}
  out <- (tp)/(fp+tp)
  names(out) <- "ppv"
  out
}

sensSummary <- function (data, lev = NULL, model = NULL) {
  tn <- table(data$pred == "worse",data$obs=='worse')[1]
  fp <- table(data$pred == "worse",data$obs=='worse')[2]
  fn <- table(data$pred == "worse",data$obs=='worse')[3]
  tp <- table(data$pred == "worse",data$obs=='worse')[4]
  if(is.na(tp)){tp <- 0}
  if(is.na(fn)){fn <- 0}
  if(is.na(fp)){fp <- 0}
  if(is.na(tn)){tn <- 0}
  out <- (tp)/(fn+tp)
  names(out) <- "sens"
  out
}

prdSummary <- function (data, lev = NULL, model = NULL) 
{
  caret:::requireNamespaceQuietStop("MLmetrics")
  if (length(levels(data$obs)) > 2) 
    stop(paste("Your outcome has", length(levels(data$obs)), 
               "levels. `prSummary`` function isn't appropriate.", 
               call. = FALSE))
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("Levels of observed and predicted data do not match.", 
         call. = FALSE)
  pr_auc <- try(MLmetrics::PRAUC(y_pred = data[, lev[2]], y_true = ifelse(data$obs == 
                                                                            lev[2], 1, 0)), silent = TRUE)
  if (inherits(pr_auc, "try-error")) 
    pr_auc <- NA
  c(AUC = pr_auc, Precision = caret:::precision.default(data = data$pred, 
                                                reference = data$obs, relevant = lev[2]), Recall = caret:::recall.default(data = data$pred, 
                                                                                                                  reference = data$obs, relevant = lev[2]), F = caret:::F_meas.default(data = data$pred, 
                                                                                                                                                                               reference = data$obs, relevant = lev[2]))
} 

npvSummary <- function (data, lev = NULL, model = NULL) {
  tn <- table(data$pred == "worse",data$obs=='worse')[1]
  fp <- table(data$pred == "worse",data$obs=='worse')[2]
  fn <- table(data$pred == "worse",data$obs=='worse')[3]
  tp <- table(data$pred == "worse",data$obs=='worse')[4]
  if(is.na(tp)){tp <- 0}
  if(is.na(fn)){fn <- 0}
  if(is.na(fp)){fp <- 0}
  if(is.na(tn)){tn <- 0}
  out <- (tn)/(fn+tn)
  names(out) <- "npv"
  out
}

auprcSummary <- function(data, lev=NULL, model = NULL){
  
  index_class2 <- data$obs == "SameBetter"|data$obs="AdviseIn"
  index_class1 <- data$obs == "worse"
  
  predictions <- predict(model, data, type="prob")
  
  the_curve <- PRcurve(data$worse[index_class2], data$worse[index_class1], curve = FALSE )
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
}




#   B. tune a variety of ML models ---------------------------------------------
#       For each dataset:
#       a. Regression models for each dataset
#           I.  Lasso 
#           II. Ridge
#           III.Elastic net
#           IV. Interim conclusion

#   B1. Depression symptoms====================================================

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# alpha = 0, Ridge Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#Custom control parameters
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs =TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE)#,
 # sampling = "smote")


# Determine optimal lamba with CV, i.e. highest ppv
lambda <- 10^seq(-3,1,length=100)
set.seed(12345)
RidgeFit.Depr <- train(Y ~ ., 
                  data = InputDepr, 
                  method = "glmnet",
                  metric = "AUC",
                  family = "binomial",
                  trControl = control, #defined above
                  tuneGrid = expand.grid(alpha = 0, lambda = lambda))

plot(RidgeFit.Depr )

ThrSeq <- seq(.7,1,length=100)
th.RidgeFit.Depr <- thresholder(RidgeFit.Depr, ThrSeq, final =TRUE, statistics = 'all' )
opt_thr <- th.RidgeFit.Depr$Sensitivity*th.RidgeFit.Depr$`Neg Pred Value`
th.RidgeFit.Depr[opt_thr==max(opt_thr,na.rm=TRUE),]
th.RidgeFit.Depr$NPV <- th.RidgeFit.Depr$`Neg Pred Value`

plotthr <- ggplot(data=th.RidgeFit.Depr) + 
                geom_point(aes(y=Specificity, x = as.numeric(row.names(th.RidgeFit.Depr))))+
                geom_line(aes(y=Specificity, x = as.numeric(row.names(th.RidgeFit.Depr))))+
                geom_text(aes(y=Specificity, x = as.numeric(row.names(th.RidgeFit.Depr)), label=rownames(th.RidgeFit.Depr)))+
                geom_point(aes(y=NPV, x = as.numeric(row.names(th.RidgeFit.Depr)))) +
                geom_line(aes(y=NPV, x = as.numeric(row.names(th.RidgeFit.Depr)))) 

# choose 27
threshold.RidgeFit.Depr = th.RidgeFit.Depr$prob_threshold[47]

# RidgeFit # provides information over the model
plot(RidgeFit.Depr) #Lambda (x-as) vs ppv (y-as)
# coef(RidgeFit.Depr$finalModel, RidgeFit.Depr$bestTune$lambda) ##Shows coefficient for each feature. Higher value = more important
plot(varImp(RidgeFit.Depr, scale = F)) #Shows the most important features

table(RidgeFit.Depr$pred$pred, RidgeFit.Depr$pred$obs)
x <- evalm(RidgeFit.Depr)
plot(varImp(RidgeFit.Depr)) 

#Sensitivity
Ridgefit.sensitivy <- sum(diag(table(RidgeFit.Depr$pred$pred, RidgeFit.Depr$pred$obs)))/sum(table(RidgeFit.Depr$pred$pred, RidgeFit.Depr$pred$obs))
Ridgefit.sensitivy

confusionMatrix(RidgeFit.Depr$pred$pred, RidgeFit.Depr$pred$obs, positive = 'worse')
confmat.Ridge.Depr <- confusionMatrix(RidgeFit.Depr, mode ='everything')
ppv.Ridge.Depr <- confmat.Ridge.Depr$table[4]/(confmat.Ridge.Depr$table[2]+confmat.Ridge.Depr$table[4])
sens.Ridge.Depr <- confmat.Ridge.Depr$table[4]/(confmat.Ridge.Depr$table[3]+confmat.Ridge.Depr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# alpha = 1, Lasso Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

lambda <- 10^seq(-3,0.5,length=100)
set.seed(12345)
LassoFit.Depr <- train(Y ~ ., 
                  data = InputDepr, 
                  method = "glmnet",
                  metric = "AUC",
                  family = "binomial",
                  trControl = control,
                  tuneGrid = expand.grid(alpha = 1, lambda = lambda))

# LassoFit.Depr
plot(LassoFit.Depr) #Lambda (x-as) vs ppv (y-as)
#coef(LassoFit.Depr$finalModel, LassoFit.Depr$bestTune$lambda)
plot(varImp(LassoFit.Depr, scale = F))
#plot(LassoFit.Depr$finalModel)

table(LassoFit.Depr$pred$pred, LassoFit.Depr$pred$obs)
x <- evalm(LassoFit.Depr)


ThrSeq <- seq(.7,1,length=100)
th.LassoFit.Depr <- thresholder(LassoFit.Depr, ThrSeq, final =TRUE, statistics = 'all' )
opt_thr <- th.LassoFit.Depr$Sensitivity*th.LassoFit.Depr$`Neg Pred Value`
th.LassoFit.Depr[opt_thr==max(opt_thr,na.rm=TRUE),]
th.LassoFit.Depr$NPV <- th.LassoFit.Depr$`Neg Pred Value`

plotthr.lasso.Depr <- ggplot(data=th.LassoFit.Depr) + 
  geom_point(aes(y=Specificity, x = as.numeric(row.names(th.LassoFit.Depr))))+
  geom_line(aes(y=Specificity, x = as.numeric(row.names(th.LassoFit.Depr))))+
  geom_text(aes(y=Specificity, x = as.numeric(row.names(th.LassoFit.Depr)), label=rownames(th.LassoFit.Depr)))+
  geom_point(aes(y=NPV, x = as.numeric(row.names(th.LassoFit.Depr))),color = "red") +
  geom_line(aes(y=NPV, x = as.numeric(row.names(th.LassoFit.Depr))),color = "red") +
  geom_text(aes(y=NPV[1], x = 0, label ="pos pred val", color = "red"))

# choose 27
threshold.lasso.Depr = th.LassoFit.Depr$prob_threshold[27]

#Sensitivity
LassoFit.sensitivy <- sum(diag(table(LassoFit.Depr$pred$pred, LassoFit.Depr$pred$obs)))/sum(table(LassoFit.Depr$pred$pred, LassoFit.Depr$pred$obs))
LassoFit.sensitivy

confusionMatrix(LassoFit.Depr$pred$pred, LassoFit.Depr$pred$obs, positive = 'worse')
confmat.Lasso.Depr <- confusionMatrix(LassoFit.Depr, mode ='everything')
ppv.Lasso.Depr <- confmat.Lasso.Depr$table[4]/(confmat.Lasso.Depr$table[2]+confmat.Lasso.Depr$table[4])
sens.Lasso.Depr <- confmat.Lasso.Depr$table[4]/(confmat.Lasso.Depr$table[3]+confmat.Lasso.Depr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 0 < alpha < 1, Elastic Net Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
lambda <- 10^seq(-3,1,length=100)
set.seed(12345)
NetFit.Depr <- train(Y ~ ., 
                data = InputDepr, 
                method = "glmnet",
                metric = "AUC",
                family = "binomial",
                trControl = control,
                tuneGrid = expand.grid(alpha = seq(0,1, length=10), lambda = lambda))



ThrSeq <- seq(0.7,1,length=30)
th.NetFit.Depr <- thresholder(NetFit.Depr, ThrSeq, final =TRUE, statistics = 'all' )
opt_thr <- th.NetFit.Depr$Sensitivity*th.NetFit.Depr$`Neg Pred Value`
th.NetFit.Depr[opt_thr==max(opt_thr,na.rm=TRUE),]
th.NetFit.Depr$NPV <- th.NetFit.Depr$`Neg Pred Value`

plotthr.NetFit.Depr <- ggplot(data=th.NetFit.Depr) + 
  geom_point(aes(y=Specificity, x = as.numeric(row.names(th.NetFit.Depr))))+
  geom_line(aes(y=Specificity, x = as.numeric(row.names(th.NetFit.Depr))))+
  geom_text(aes(y=Specificity, x = as.numeric(row.names(th.NetFit.Depr)), label=rownames(th.NetFit.Depr)))+
  geom_point(aes(y=NPV, x = as.numeric(row.names(th.NetFit.Depr))),color = "red") +
  geom_line(aes(y=NPV, x = as.numeric(row.names(th.NetFit.Depr))),color = "red") +
  geom_text(aes(y=NPV[1], x = 0, label ="pos pred val", color = "red"))

# choose 16
threshold.NetFit.Depr = th.NetFit.Depr$prob_threshold[16]



#NetFit
plot(NetFit.Depr) #lambda values (x-as) vs ppv (y-as), several alpha's
#coef(NetFit.Depr$finalModel, NetFit.Depr$bestTune$lambda)
plot(varImp(NetFit.Depr, scale = F))
#plot(NetFit.Depr$finalModel)

table(NetFit.Depr$pred$pred, NetFit.Depr$pred$obs)

x <- evalm(NetFit.Depr)

#Sensitivity
NetFit.sensitivy <- sum(diag(table(NetFit.Depr$pred$pred, NetFit.Depr$pred$obs)))/sum(table(NetFit.Depr$pred$pred, NetFit.Depr$pred$obs))
NetFit.sensitivy

confusionMatrix(NetFit.Depr$pred$pred, NetFit.Depr$pred$obs, positive = 'worse')
confmat.Net.Depr <- confusionMatrix(NetFit.Depr)
ppv.Net.Depr <- confmat.Net.Depr$table[4]/(confmat.Net.Depr$table[2]+confmat.Net.Depr$table[4])
sens.Net.Depr <- confmat.Net.Depr$table[4]/(confmat.Net.Depr$table[3]+confmat.Net.Depr$table[4])

evalm(NetFit.Depr$predpred[NetFit.Depr$pred$lambda==NetFit.Depr$bestTune$lambda &
                                 NetFit.Depr$pred$alpha==NetFit.Depr$bestTune$alpha],
                NetFit.Depr$pred$obs[NetFit.Depr$pred$lambda==NetFit.Depr$bestTune$lambda &
                                   NetFit.Depr$pred$alpha==NetFit.Depr$bestTune$alpha],
                positive = 'worse')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Comparing Lasso, Ridge and Net regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

model_list.Depr <- list(Lasso = LassoFit.Depr, Ridge = RidgeFit.Depr, ElasticNet= NetFit.Depr)
res <- resamples(model_list) # part of caret package to compare the models
summary(res) # tabel overview of models
bwplot(res) #boxplot
xyplot(res, metric = 'ppv')

evalm(model_list.Depr)

# PPV without smote around .4, but quite variable because of small number of cases
# therefore with SMOTE, much more stable ppv within CV procedure
# Elastic net out-performs the other two with a PPV of 54%
NetFit.Depr$results[NetFit.Depr$results$ppv==max(NetFit.Depr$results$ppv),]
# Against a baseline prevalence of 11.8% a five-fold improvement in information
Freq(InputDepr$Y)
# However, only 15 of 485 (3%) are found to be cases
predNetFit.Depr <- InputDepr %>% select(-c('Y')) %>% predict(NetFit.Depr,.)
Freq(predNetFit.Depr)
# This will only be of marginal influence on clinical practice.
# Because only 3.1% of subjects get information to rethink indication of MBCT of whom 9 are rightfully rethought.
table(predNetFit.Depr,InputDepr$Y)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Tree Based analyses

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Classification tree

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs =TRUE,
  #sampling = "smote",
  summaryFunction = prSummary,
  savePredictions = TRUE)


tunegrid <- expand.grid(cp=seq(0,0.01, length=10))

set.seed(12345)
CT.Depr <- train(Y~.,
                   data = InputDepr,
                   method = "rpart",
                   metric =  "AUC",
                   tuneGrid = tunegrid,
                   #tuneLength = 15,
                   trControl = control)

x <- evalm(CT.Depr)

print(CT.Depr)
plot(CT.Depr)
plot(CT.Depr$finalModel, uniform =TRUE, main="Classification Tree")
text(CT.Depr$finalModel, use.n.=TRUE, all=TRUE, cex=.8)


confusionMatrix(CT.Depr$pred$pred, CT.Depr$pred$obs, positive = 'worse')
confmat.CT.Depr <- confusionMatrix(CT.Depr, mode ='everything')
ppv.CT.Depr <- confmat.CT.Depr$table[4]/(confmat.CT.Depr$table[2]+confmat.CT.Depr$table[4])
sens.CT.Depr <- confmat.CT.Depr$table[4]/(confmat.CT.Depr$table[3]+confmat.CT.Depr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Random Forest

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs =TRUE,
  #sampling = "smote",
  summaryFunction =  prSummary,
  savePredictions = TRUE)

mtry <- c(7,8,9,10,11,12,13) #firsr did random search: Best <10
tunegrid <- expand.grid(.mtry=mtry)
ntree <- 1500

set.seed(12345)
#InputDepr$Y <- recode(InputDepr$Y,'worse'=1, 'SameBetter'=0)
RF.Depr <- train(Y~.,
                   data = InputDepr,
                   method = 'rf',
                   metric =  "AUC",
                   tuneGrid = tunegrid,
                   trControl = control)

# helper function for the plots

RF.Depr
evalm(RF.Depr)


plot(RF.Depr)
plot(varImp(RF.Depr))
predRF.Depr <- InputDepr %>% select(-c('Y')) %>% predict(RF.Depr,.)
Freq(predRF.Depr)
table(predRF.Depr,InputDepr$Y)

confusionMatrix(RF.Depr$pred$pred, RF.Depr$pred$obs, positive = 'worse')
confmat.RF.Depr <- confusionMatrix(RF.Depr, mode ='everything')
ppv.RF.Depr <- confmat.RF.Depr$table[4]/(confmat.RF.Depr$table[2]+confmat.RF.Depr$table[4])
sens.RF.Depr <- confmat.RF.Depr$table[4]/(confmat.RF.Depr$table[3]+confmat.RF.Depr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Extreme gradient boosting

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  savePredictions = TRUE,
  #sampling = "smote",
  classProbs = TRUE,
  summaryFunction = prSummary,
  verboseIter = FALSE, #no training log,
  allowParallel = TRUE #False for reproducible results
)

# # Tuning in following steps:
# 1. fixing learning rate eta an number of iterations nrounds
      # eta c(0.01, 0.025, 0.05) --> c(0.001,0.005,0.01) --> c(0.001,0.005)
      # nrounds c(100,200,300,400,500) --> c(10,50 100,500,1000) --> c(100,500,1000)-->c(500)
# 2. maximum depth of search max_depth and child weight min_child_weight
      # max_depth c(1,2,5,10) --> c(2,3,4) --> c(2)
      # minchild_weight c(1,5,10) -->c(2,3) --> c(3)
# 3. setting column colsample_bytree and row sampling subsample
      #colsample_bytree: c(0.1, 0.5, 1) --> c(0.05,0.1,0.15)-->c(0.1)
      #subsample: c(0.1, 0.5, 1) --> c(0.05,0.1,0.15)
# 4. experimmenting with different gamma values

# 5. Reducing the learning rate.

#InputDepr$Y <- InputDepr$Y %>% recode(.,'worse'= "a",'SameBetter' ="b")

# for ppv : aucPR: 0.35
# set.seed(12345)
# tunegrid <- expand.grid(nrounds=c(300),#10,50, 100,500,1000
#                         eta =  c(0.01),#0.001,0.005,0.01
#                         max_depth = c(1),
#                         min_child_weight = c(1),
#                         colsample_bytree = c(0.9),
#                         subsample = c(0.5),
#                         gamma = c(1))

set.seed(12345)
tunegrid <- expand.grid(nrounds=c(700),#10,50, 100,500,1000
                        eta =  c(0.001),#0.001,0.005,0.01
                        max_depth = c(1),
                        min_child_weight = c(3),
                        colsample_bytree = c(0.9),
                        subsample = c(0.5),
                        gamma = c(1))


XGB.Depr <- train(Y ~., data = InputDepr,
                    method = "xgbTree",
                    metric = "AUC",
                    trControl=fitControl,
                    #weights = model_weights,
                    tuneGrid = tunegrid)

XGB.Depr
plot(XGB.Depr)
evalm(XGB.Depr)


ThrSeq <- seq(0,1,length=100)
th.XGB.Depr <- thresholder(XGB.Depr, ThrSeq, final =TRUE, statistics = 'all' )
opt_thr <- th.XGB.Depr$Sensitivity*th.XGB.Depr$`Neg Pred Value`
th.XGB.Depr[opt_thr==max(opt_thr,na.rm=TRUE),]
th.XGB.Depr$NPV <- th.XGB.Depr$`Neg Pred Value`

plotthr.XGB.Depr <- ggplot(data=th.XGB.Depr) + 
  geom_point(aes(y=Specificity, x = as.numeric(row.names(th.XGB.Depr))))+
  geom_line(aes(y=Specificity, x = as.numeric(row.names(th.XGB.Depr))))+
  geom_text(aes(y=Specificity, x = as.numeric(row.names(th.XGB.Depr)), label=rownames(th.XGB.Depr)))+
  geom_point(aes(y=NPV, x = as.numeric(row.names(th.XGB.Depr))),color = "red") +
  geom_line(aes(y=NPV, x = as.numeric(row.names(th.XGB.Depr))),color = "red") +
  geom_text(aes(y=NPV[1], x = 0, label ="pos pred val", color = "red"))

# choose 64
threshold.XGB.Depr = th.XGB.Depr$prob_threshold[63]


# helper function for the plots
tuneplot <- function(x, probs = .16){
  ggplot(x) +
    coord_cartesian(ylim = c(0.1, 1)) +
    theme_bw()
}

tuneplot(XGB.Depr)
print(XGB.Depr)

VI <- varImp(XGB.Depr, scale =FALSE)
plot(VI)

confusionMatrix(XGB.Depr$pred$pred, XGB.Depr$pred$obs, positive = 'a')
confmat.XGB.Depr <- confusionMatrix(XGB.Depr, norm ='overall')
ppv.XGB.Depr <- confmat.XGB.Depr$table[4]/(confmat.XGB.Depr$table[2]+confmat.XGB.Depr$table[4])
sens.XGB.Depr <- confmat.XGB.Depr$table[4]/(confmat.XGB.Depr$table[3]+confmat.XGB.Depr$table[4])


# PPV without smote around .4, but quite variable because of small number of cases
# therefore with SMOTE, much more stable ppv within CV procedure
# Elastic net out-performs the other two with a PPV of 54%
XGB.Depr$results[XGB.Depr$results$ppv==max(XGB.Depr$results$ppv),]
# Against a baseline prevalence of 11.8% a five-fold improvement in information
Freq(InputDepr$Y)
# However, only 15 of 485 (3%) are found to be cases
predXGB.Depr <- InputDepr %>% select(-c('Y')) %>% predict(XGB.Depr,.)
Freq(predXGB.Depr)
# This will only be of marginal influence on clinical practice.
# Because only 3.1% of subjects get information to rethink indication of MBCT of whom 9 are rightfully rethought.
table(predXGB.Depr,InputDepr$Y)

model_list.Depr <- list(Ridge = RidgeFit.Depr, Lasso = LassoFit.Depr,  ElasticNet= NetFit.Depr, ClassificationTree = CT.Depr,RandomForrest = RF.Depr,  ExtremeGradientBoost = XGB.Depr)
res <- resamples(model_list.Depr) # part of caret package to compare the models
summary(res) # tabel overview of models
bwplot(res) #boxplot
xyplot(res, metric = 'ppv')

evalm(model_list.Depr)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# # Super Learner
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# set.seed(1)
# listWrappers()
# # change data to numeric 
# InputDeprNum <- data.matrix()
# InputDeprNum <- InputDepr %>% mutate_if(is.logical,as.numeric)
# InputDeprNum <- InputDeprNum %>% mutate_if(is.factor,as.numeric)
# InputDeprNumY <- InputDeprNum$Y
# InputDeprNumY[InputDeprNumY==2] <- 0
# InputDeprNumX <- InputDeprNum %>% select(-starts_with('Y'))
# 
# # sl_lasso.Depr = SuperLearner(Y = InputDeprNumY, X = InputDeprNumX, family=binomial(), SL.library = "SL.glmnet")
# # sl_lasso.Depr # risk is winning model
# # str(sl_lasso.Depr$fitLibrary$SL.glmnet_All$object, max.level = 1)
# # 
# # sl_rf.Depr = SuperLearner(Y = InputDeprNumY, X = InputDeprNumX, family=binomial(), SL.library = "SL.ranger")
# # sl_rf.Depr # risk is winning model
# # str(sl_rf.Depr$fitLibrary$SL.ranger_All$object, max.level = 1)
# # 
# # sl.Depr = SuperLearner(Y = InputDeprNumY, X = InputDeprNumX, family=binomial(), 
# #                        SL.library = c("SL.mean", "SL.glmnet", "SL.ranger"))
# # sl.Depr
# 
# # go to cross validation
# # cv_sl.Depr = CV.SuperLearner(Y = InputDeprNumY, X = InputDeprNumX, family=binomial(), 
# #                           V = 5,
# #                           SL.library = c("SL.mean", "SL.glmnet", "SL.ranger"))
# # summary(cv_sl.Depr)
# # table(simplify2array(cv_sl.Depr$whichDiscreteSL))
# # plot(cv_sl.Depr) + theme_bw()
# 
# #RF tuning
# tune.RF =list(mtry_seq = c(1:10))
# #XGB tuning
# tune.xgb = list(ntrees = c(100, 500, 1000),
#               max_depth = c(2,3,4),
#               shrinkage = c(0.001, 0.01, 0.1),
#               minobspernode = c(1,2,4,8))
# 
# 
# learners.RF.Depr  = create.Learner(c("SL.ranger"), tune = tune.RF)
# learners.XGB.Depr = create.Learner(c("SL.xgboost"), tune = tune.xgb, detailed_names = TRUE, name_prefix = "xgb" )
# 
# cv_sl.Depr = CV.SuperLearner(Y = InputDeprNumY, X = InputDeprNumX, family=binomial(), 
#                              V = 5,
#                              method = "method.AUC",
#                              SL.library = c("SL.mean", "SL.glmnet", learners.RF.Depr$names, learners.XGB.Depr$names,"SL.ranger"))
# summary(cv_sl.Depr)
# plot(cv_sl.Depr) + theme_bw()
# 
# # calculate positive predictive value from superlearner
# pred.sl.Depr = stats::predict(cv_sl.Depr, InputDeprNumX)
# str(pred.sl.Depr)
# qplot(pred.sl.Depr[,1])+theme_minimal()
# qplot(InputDeprNumY,pred.sl.Depr[,1])+theme_minimal()
# pred_rocr = ROCR::prediction(pred.sl.Depr$pred, InputDeprNumY)
# auc = ROCR::perfomance(pred_rocr, measure="AUC", x.measure = "cutoff")@y.values[[1]]
# auc
# 
# # assess best models pivottable
# table(cv_sl.Depr$library.predict[,3]<.5, InputDeprNumY)
# table(cv_sl.Depr$SL.predict<.5, InputDeprNumY)

#   B2. Drop Out====================================================


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# alpha = 0, Ridge Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#Custom control parameters
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  #sampling = "smote",
  classProbs =TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE)


# Determine optimal lamba with CV, i.e. highest ppv
lambda <- 10^seq(-3,0,length=100)
set.seed(12345)
RidgeFit.DO <- train(Y ~ ., 
                  data = InputDO, 
                  method = "glmnet",
                  metric = "AUC",
                  family = "binomial",
                  trControl = control, #defined above
                  tuneGrid = expand.grid(alpha = 0, lambda = lambda))

# RidgeFit.DO # provides information over the model
plot(RidgeFit.DO) #Lambda (x-as) vs ppv (y-as)
# coef(RidgeFit.DO$finalModel, RidgeFit.DO$bestTune$lambda) ##Shows coefficient for each feature. Higher value = more important
plot(varImp(RidgeFit.DO, scale = F)) #Shows the most important features

table(RidgeFit.DO$pred$pred, RidgeFit.DO$pred$obs)

confusionMatrix(RidgeFit.DO$pred$pred, RidgeFit.DO$pred$obs, positive = 'worse')
confmat.Ridge.DO <- confusionMatrix(RidgeFit.DO, mode ='everything')
ppv.Ridge.DO <- confmat.Ridge.DO$table[4]/(confmat.Ridge.DO$table[2]+confmat.Ridge.DO$table[4])
sens.Ridge.DO <- confmat.Ridge.DO$table[4]/(confmat.Ridge.DO$table[3]+confmat.Ridge.DO$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# alpha = 1, Lasso Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

lambda <- 10^seq(-3,0.5,length=100)
set.seed(12345)
LassoFit.DO <- train(Y ~ ., 
                  data = InputDO, 
                  method = "glmnet",
                  metric = "AUC",
                  family = "binomial",
                  trControl = control,
                  tuneGrid = expand.grid(alpha = 1, lambda = lambda))

# LassoFit.DO
plot(LassoFit.DO) #Lambda (x-as) vs ppv (y-as)
#coef(LassoFit.DO$finalModel, LassoFit.DO$bestTune$lambda)
plot(varImp(LassoFit.DO, scale = F))
#plot(LassoFit.DO$finalModel)

table(LassoFit.DO$pred$pred, LassoFit.DO$pred$obs)

predLasso.DO <- InputDO %>% select(-c('Y')) %>% predict(LassoFit.DO,.)
Freq(predLasso.DO)
table(predLasso.DO,InputDO$Y)

confusionMatrix(LassoFit.DO$pred$pred, LassoFit.DO$pred$obs, positive = 'worse')
confmat.Lasso.DO <- confusionMatrix(LassoFit.DO, mode ='everything')
ppv.Lasso.DO <- confmat.Lasso.DO$table[4]/(confmat.Lasso.DO$table[2]+confmat.Lasso.DO$table[4])
sens.Lasso.DO <- confmat.Lasso.DO$table[4]/(confmat.Lasso.DO$table[3]+confmat.Lasso.DO$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 0 < alpha < 1, Elastic Net Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
lambda <- 10^seq(-3,0,length=100)
set.seed(12345)
NetFit.DO <- train(Y ~ ., 
                data = InputDO, 
                method = "glmnet",
                metric = "AUC",
                family = "binomial",
                trControl = control,
                tuneGrid = expand.grid(alpha = seq(0,1, length=10), lambda = lambda))

#NetFit.DO
plot(NetFit.DO) #lambda values (x-as) vs ppv (y-as), several alpha's
#coef(NetFit.DO$finalModel, NetFit.DO$bestTune$lambda)
plot(varImp(NetFit.DO, scale = F))
#plot(NetFit.DO$finalModel)

table(NetFit.DO$pred$pred, NetFit.DO$pred$obs)

predNet.DO <- InputDO %>% select(-c('Y')) %>% predict(NetFit.DO,.)
Freq(predNet.DO)
table(predNet.DO,InputDO$Y)

bestpredmod <- NetFit.DO$pred[NetFit.DO$pred$lambda== NetFit.DO$bestTune$lambda & NetFit.DO$pred$alpha== NetFit.DO$bestTune$alpha, ]
prSummary(bestpredmod)
table(bestpredmod$pred,bestpredmod$obs)

confusionMatrix(NetFit.DO$pred$pred, NetFit.DO$pred$obs, positive = 'worse')
confmat.Net.DO <- confusionMatrix(NetFit.DO, mode ='everything')
ppv.Net.DO <- confmat.Net.DO$table[4]/(confmat.Net.DO$table[2]+confmat.Net.DO$table[4])
sens.Net.DO <- confmat.Net.DO$table[4]/(confmat.Net.DO$table[3]+confmat.Net.DO$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Comparing Lasso, Ridge and Net regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

model_list <- list(Lasso = LassoFit.DO, Ridge = RidgeFit.DO, ElasticNet= NetFit.DO)
res <- resamples(model_list) # part of caret package to compare the models
summary(res) # tabel overview of models
bwplot(res) #boxplot
xyplot(res, metric = 'ppv')

# therefore with SMOTE, much more stable ppv within CV procedure
# Elastic net out-performs the other two with a PPV of 12%
NetFit.DO$results[NetFit.DO$results$ppv==max(NetFit.DO$results$ppv),]
# Against a baseline prevalence of 8% a 1.5-fold improvement in information
Freq(InputDO$Y)
# However, only 20 of 749 (2.7%) are found to be cases
predNetFit.DO <- InputDO %>% select(-c('Y')) %>% predict(NetFit.DO,.)
Freq(predNetFit.DO)
# This will only be of marginal influence on clinical practice.
# Because only 20 patients (2.7%) of subjects get information to rethink indication of MBCT of whom 6 are rightfully rethought.
table(predNetFit.DO,InputDO$Y)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Tree Based analyses

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Classification tree

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs =TRUE,
  #sampling = "smote",
  summaryFunction = prSummary,
  savePredictions = TRUE)

set.seed(12345)
CT.DO <- train(Y~.,
                   data = InputDO,
                   method = "rpart",
                   metric =  "AUC",
                   tuneLength = 15,
                   trControl = control)

print(CT.DO)
plot(CT.DO)
plot(CT.DO$finalModel, uniform =TRUE, main="Classification Tree")
text(CT.DO$finalModel, use.n.=TRUE, all=TRUE, cex=.8)

confusionMatrix(CT.DO$pred$pred, CT.DO$pred$obs, positive = 'worse')
confmat.CT.DO <- confusionMatrix(CT.DO, mode ='everything')
ppv.CT.DO <- confmat.CT.DO$table[4]/(confmat.CT.DO$table[2]+confmat.CT.DO$table[4])
sens.CT.DO <- confmat.CT.DO$table[4]/(confmat.CT.DO$table[3]+confmat.CT.DO$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Random Forest

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs =TRUE,
  #sampling = "smote",
  summaryFunction = prSummary,
  savePredictions = TRUE)

mtry <- c(1:10,50) # first did random search: Best <10
tunegrid <- expand.grid(.mtry=mtry)

set.seed(12345)
RF.DO <- train(Y~.,
                   data = InputDO,
                   method = 'rf',
                   metric =  "AUC",
                   tuneGrid = tunegrid,
                   trControl = control)

print(RF.DO)
plot(RF.DO)
plot(varImp(RF.DO))

x <- evalm(RF.DO)

predRF.DO <- InputDO %>% select(-c('Y')) %>% predict(RF.DO,.)
Freq(predRF.DO)
table(predRF.DO,InputDO$Y)

confusionMatrix(RF.DO$pred$pred, RF.DO$pred$obs, positive = 'worse')
confmat.RF.DO <- confusionMatrix(RF.DO, mode ='everything')
ppv.RF.DO <- confmat.RF.DO$table[4]/(confmat.RF.DO$table[2]+confmat.RF.DO$table[4])
sens.RF.DO <- confmat.RF.DO$table[4]/(confmat.RF.DO$table[3]+confmat.RF.DO$table[4])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Extreme gradient boosting

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  savePredictions = TRUE,
  #sampling = "smote",
  classProbs = TRUE,
  summaryFunction = prSummary,
  verboseIter = FALSE, #no training log,
  allowParallel = TRUE #False for reproducible results
)

# # Tuning in following steps:
# 1. fixing learning rate eta an number of iterations nrounds
# eta c(0.01, 0.025, 0.05) --> c(0.001, 0.01, 0.05,0.1) --> c(0.001) --> 2nd c(0.0001, 0.001, 0.1)
# nrounds c(50, 200, 500, 750, 1000) --> c(50, 200, 300,2000) --> c(300)
# 2. maximum depth of search max_depth and child weight min_child_weight
# max_depth c(1,2,5,10) --> c(1,10,20,50) --> c(20)
# minchild_weight c(1,5,10) -->c(4,5,6) --> c(3)
# 3. setting column colsample_bytree and row sampling subsample
#colsample_bytree: c(0.1, 0.5, 1) --> c(0.5)
#subsample: c(0.1, 0.5, 1) --> c(0.05,0.1,0.15)
# 4. experimmenting with different gamma values
# gamma: c(0,5,10,15)) --> c(0,1,2,3)
# 5. Reducing the learning rate.
# eta c(0.0001,0.001,0.1)


tunegrid <- expand.grid(nrounds=c(50, 200, 500, 750, 1000),
                        eta = c(0.001, 0.05,0.1),
                        max_depth = c(1),
                        min_child_weight = c(1),
                        colsample_bytree = c(0.5),
                        subsample = c(0.15),
                        gamma = c(2))

set.seed(12345)
XGB.DO <- train(Y ~., data = InputDO,
                  method = "xgbTree",
                  metric = "AUC",
                  trControl=fitControl,
                  tuneGrid = tunegrid)




# helper function for the plots
tuneplot <- function(x, probs = .90){
  ggplot(x) +
    coord_cartesian(ylim = c(0.0, 0.3)) +
    theme_bw()
}

tuneplot(XGB.DO)
print(XGB.DO)
VI <- varImp(XGB.DO, scale =FALSE)
plot(VI)

confusionMatrix(XGB.DO$pred$pred, XGB.DO$pred$obs, positive = 'worse')
confmat.XGB.DO <- confusionMatrix(XGB.DO, mode ='everything')
ppv.XGB.DO <- confmat.XGB.DO$table[4]/(confmat.XGB.DO$table[2]+confmat.XGB.DO$table[4])
sens.XGB.DO <- confmat.XGB.DO$table[4]/(confmat.XGB.DO$table[3]+confmat.XGB.DO$table[4])



model_list.DO <- list(Lasso = LassoFit.DO, Ridge = RidgeFit.DO, ElasticNet= NetFit.DO, ClassificationTree = CT.DO, RandomForrest = RF.DO, ExtremeGradientBoost = XGB.DO)
res <- resamples(model_list.DO) # part of caret package to compare the models
summary(res) # tabel overview of models
bwplot(res) #boxplot
xyplot(res, metric = 'ppv')

evalm(model_list.DO)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# # Super Learner
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# set.seed(1)
# listWrappers()
# # change data to numeric 
# InputDONum <- InputDO %>% mutate_if(is.logical,as.numeric)
# InputDONum <- InputDONum %>% mutate_if(is.factor,as.numeric)
# InputDONumY <- InputDONum$Y
# InputDONumY[InputDONumY==2] <- 0
# InputDONumX <- InputDONum %>% select(-starts_with('Y'))
# 
# 
# #RF tuning
# tune.RF =list(mtry_seq = c(1:10))
# #XGB tuning
# tune.xgb = list(ntrees = c(100, 500, 1000),
#                 max_depth = c(2,3,4),
#                 shrinkage = c(0.001, 0.01, 0.1),
#                 minobspernode = c(1,2,4,8))
# 
# 
# learners.RF.DO  = create.Learner(c("SL.ranger"), tune = tune.RF)
# learners.XGB.DO = create.Learner(c("SL.xgboost"), tune = tune.xgb, detailed_names = TRUE, name_prefix = "xgb" )
# 
# cv_sl.DO = CV.SuperLearner(Y = InputDONumY, X = InputDONumX, family=binomial(), 
#                              V = 5,
#                              method = "method.AUC",
#                              SL.library = c("SL.mean", "SL.glmnet", learners.RF.DO$names, learners.XGB.DO$names,"SL.ranger"))
# summary(cv_sl.DO)
# plot(cv_sl.DO) + theme_bw()
# 
# # calculate positive predictive value from superlearner
# pred.sl.DO = predict(cv_sl.DO, InputDONumX)
# str(pred.sl.DO)
# qplot(pred.sl.DO[,1])+theme_minimal()
# qplot(InputDONumY,pred.sl.DO[,1])+theme_minimal()
# pred_rocr = ROCR::prediction(pred.sl.DO$pred, InputDONumY)
# auc = ROCR::perfomance(pred_rocr, measure="AUC", x.measure = "cutoff")@y.values[[1]]
# auc
# 
# # assess best models pivottable
# table(cv_sl.DO$library.predict[,3]<.5, InputDONumY)
# table(cv_sl.DO$SL.predict<.5, InputDONumY)


#   B3. Drop Out and DOession====================================================

#Custom control parameters
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  #sampling = "smote",
  classProbs =TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# alpha = 0, Ridge Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




# Determine optimal lamba with CV, i.e. highest ppv
lambda <- 10^seq(-3,2,length=100)

set.seed(12345)
RidgeFit.DODepr <- train(Y ~ ., 
                     data = InputDODepr, 
                     method = "glmnet",
                     metric = "AUC",
                     family = "binomial",
                     trControl = control, #defined above
                     tuneGrid = expand.grid(alpha = 0, lambda = lambda))

# RidgeFit.DODepr # provides information over the model
plot(RidgeFit.DODepr) #Lambda (x-as) vs ppv (y-as)
# coef(RidgeFit.DODepr$finalModel, RidgeFit.DODepr$bestTune$lambda) ##Shows coefficient for each feature. Higher value = more important
plot(varImp(RidgeFit.DODepr, scale = F)) #Shows the most important features

table(RidgeFit.DODepr$pred$pred, RidgeFit.DODepr$pred$obs)

confusionMatrix(RidgeFit.DODepr$pred$pred, RidgeFit.DODepr$pred$obs, positive = 'worse')
confmat.Ridge.DODepr <- confusionMatrix(RidgeFit.DODepr, mode ='everything')
ppv.Ridge.DODepr <- confmat.Ridge.DODepr$table[4]/(confmat.Ridge.DODepr$table[2]+confmat.Ridge.DODepr$table[4])
sens.Ridge.DODepr <- confmat.Ridge.DODepr$table[4]/(confmat.Ridge.DODepr$table[3]+confmat.Ridge.DODepr$table[4])



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# alpha = 1, Lasso Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

lambda <- 10^seq(-3,0.5,length=100)
set.seed(12345)
LassoFit.DODepr <- train(Y ~ ., 
                     data = InputDODepr, 
                     method = "glmnet",
                     metric = "AUC",
                     family = "binomial",
                     trControl = control,
                     tuneGrid = expand.grid(alpha = 1, lambda = lambda))

# LassoFit.DODepr
plot(LassoFit.DODepr) #Lambda (x-as) vs ppv (y-as)
#coef(LassoFit.DODepr$finalModel, LassoFit.DODepr$bestTune$lambda)
plot(varImp(LassoFit.DODepr, scale = F))
#plot(LassoFit.DODepr$finalModel)

table(LassoFit.DODepr$pred$pred, LassoFit.DODepr$pred$obs)

confusionMatrix(LassoFit.DODepr$pred$pred, LassoFit.DODepr$pred$obs, positive = 'worse')
confmat.Lasso.DODepr <- confusionMatrix(LassoFit.DODepr, mode ='everything')
ppv.Lasso.DODepr <- confmat.Lasso.DODepr$table[4]/(confmat.Lasso.DODepr$table[2]+confmat.Lasso.DODepr$table[4])
sens.Lasso.DODepr <- confmat.Lasso.DODepr$table[4]/(confmat.Lasso.DODepr$table[3]+confmat.Lasso.DODepr$table[4])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 0 < alpha < 1, Elastic Net Regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
lambda <- 10^seq(-3,0,length=100)
set.seed(12345)
NetFit.DODepr <- train(Y ~ ., 
                   data = InputDODepr, 
                   method = "glmnet",
                   metric = "AUC",
                   family = "binomial",
                   trControl = control,
                   tuneGrid = expand.grid(alpha = seq(0,1, length=10), lambda = lambda))

#NetFit.DODepr
plot(NetFit.DODepr) #lambda values (x-as) vs ppv (y-as), several alpha's
#coef(NetFit.DODepr$finalModel, NetFit.DODepr$bestTune$lambda)
plot(varImp(NetFit.DODepr, scale = F))
#plot(NetFit.DODepr$finalModel)

table(NetFit.DODepr$pred$pred, NetFit.DODepr$pred$obs)


confusionMatrix(NetFit.DODepr$pred$pred, NetFit.DODepr$pred$obs, positive = 'worse')
confmat.Net.DODepr <- confusionMatrix(NetFit.DODepr, mode ='everything')
ppv.Net.DODepr <- confmat.Net.DODepr$table[4]/(confmat.Net.DODepr$table[2]+confmat.Net.DODepr$table[4])
sens.Net.DODepr <- confmat.Net.DODepr$table[4]/(confmat.Net.DODepr$table[3]+confmat.Net.DODepr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Comparing Lasso, Ridge and Net regression

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

model_list <- list(Lasso = LassoFit.DODepr, Ridge = RidgeFit.DODepr, ElasticNet= NetFit.DODepr)
res <- resamples(model_list) # part of caret package to compare the models
summary(res) # tabel overview of models
bwplot(res) #boxplot
xyplot(res, metric = 'ppv')

# PPV without smote around .4, but quite variable because of small number of cases
# therefore with SMOTE, much more stable ppv within CV procedure
# Elastic net out-performs the other two with a PPV of 49%
NetFit.DODepr$results[NetFit.DODepr$results$ppv==max(NetFit.DODepr$results$ppv),]
# Against a baseline prevalence of 21.4% a twofold improvement in information
Freq(InputDODepr$Y)
# However, only 25 of 542 (4.6%) are found to be cases
predNetFit.DODepr <- InputDODepr %>% select(-c('Y')) %>% predict(NetFit.DODepr,.)
Freq(predNetFit.DODepr)
# This will only be of marginal influence on clinical practice.
# Because only 25 of subjects get information to rethink indication of MBCT of whom 18 are rightfully rethought.
table(predNetFit.DODepr,InputDODepr$Y)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Tree Based analyses

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Classification tree

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs =TRUE,
  #sampling = "smote",
  summaryFunction = prSummary,
  savePredictions = TRUE)

set.seed(12345)
CT.DODepr <- train(Y~.,
                   data = InputDODepr,
                   method = "rpart",
                   metric =  "AUC",
                   tuneLength = 15,
                   trControl = control)

print(CT.DODepr)
plot(CT.DODepr)
plot(CT.DODepr$finalModel, uniform =TRUE, main="Classification Tree")
text(CT.DODepr$finalModel, use.n.=TRUE, all=TRUE, cex=.8)

confusionMatrix(CT.DODepr$pred$pred, CT.DODepr$pred$obs, positive = 'worse')
confmat.CT.DODepr <- confusionMatrix(CT.DODepr, mode ='everything')
ppv.CT.DODepr <- confmat.CT.DODepr$table[4]/(confmat.CT.DODepr$table[2]+confmat.CT.DODepr$table[4])
sens.CT.DODepr <- confmat.CT.DODepr$table[4]/(confmat.CT.DODepr$table[3]+confmat.CT.DODepr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Random Forest

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  #sampling = "smote",
  classProbs =TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE)

mtry <- c(1:10) # firsr did random search: Best <10
tunegrid <- expand.grid(.mtry=mtry)

set.seed(12345)
RF.DODepr <- train(Y~.,
                           data = InputDODepr,
                           method = 'rf',
                           metric =  "AUC",
                           tuneGrid = tunegrid,
                           trControl = control)

print(RF.DODepr)
plot(RF.DODepr)
plot(varImp(RF.DODepr))

x <- evalm(RF.DODepr)



predRF.DODepr <- InputDODepr %>% select(-c('Y')) %>% predict(RF.DODepr,.)
Freq(predRF.DODepr)
table(predRF.DODepr,InputDODepr$Y)

confusionMatrix(RF.DODepr$pred$pred, RF.DODepr$pred$obs, positive = 'worse')
confmat.RF.DODepr <- confusionMatrix(RF.DODepr, mode ='everything')
ppv.RF.DODepr <- confmat.RF.DODepr$table[4]/(confmat.RF.DODepr$table[2]+confmat.RF.DODepr$table[4])
sens.RF.DODepr <- confmat.RF.DODepr$table[4]/(confmat.RF.DODepr$table[3]+confmat.RF.DODepr$table[4])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Extreme gradient boosting

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Tuning in following steps:
# 1. fixing learning rate eta an number of iterations nrounds
# eta c(0.01, 0.025, 0.05) --> c(0.0001, 0.001, 0.01, 0.015) --> c(0.0001)
# nrounds c(50, 200, 500, 750, 1000) --> c(50, 200, 300,2000) --> 
# 2. maximum depth of search max_depth and child weight min_child_weight
# max_depth c(1,2,5,10) --> c(1) 
# minchild_weight c(1,5,10) --> 1
# 3. setting column colsample_bytree and row sampling subsample
#colsample_bytree: c(0.1, 0.5, 1) --> c(0.5)
#subsample: c(0.1, 0.5, 1) --> c(0.05,0.1,0.15)
# 4. experimmenting with different gamma values
# gamma: c(0,5,10,15)) --> c(0,1,2,3)
# 5. Reducing the learning rate.
# eta c(0.0001,0.001,0.1)

set.seed(12345)
tunegrid <- expand.grid(nrounds=c(10, 25, 50, 100,250),
                        eta = c(0.0001),
                        max_depth = c(1),
                        min_child_weight = c(1),
                        colsample_bytree = c(0.1),
                        subsample = c(0.5,1),
                        gamma = c(0,5,10))

set.seed(12345)

XGB.DODepr <- train(Y ~., data = InputDODepr,
                method = "xgbTree",
                metric = "AUC",
                trControl=fitControl,
                tuneGrid = tunegrid)


ThrSeq <- seq(0,1,length=100)
th.XGB.DODepr <- thresholder(XGB.DODepr, ThrSeq, final =TRUE, statistics = 'all' )
opt_thr <- th.XGB.DODepr$Sensitivity*th.XGB.DODepr$`Neg Pred Value`
th.XGB.DODepr[opt_thr==max(opt_thr,na.rm=TRUE),]
th.XGB.DODepr$NPV <- th.XGB.DODepr$`Neg Pred Value`

plotthr.XGB.DODepr <- ggplot(data=th.XGB.DODepr) + 
  geom_point(aes(y=Specificity, x = as.numeric(row.names(th.XGB.DODepr))))+
  geom_line(aes(y=Specificity, x = as.numeric(row.names(th.XGB.DODepr))))+
  geom_text(aes(y=Specificity, x = as.numeric(row.names(th.XGB.DODepr)), label=rownames(th.XGB.DODepr)))+
  geom_point(aes(y=NPV, x = as.numeric(row.names(th.XGB.DODepr))),color = "red") +
  geom_line(aes(y=NPV, x = as.numeric(row.names(th.XGB.DODepr))),color = "red") +
  geom_text(aes(y=NPV[1], x = 0, label ="pos pred val", color = "red"))

# choose 64
threshold.XGB.DODepr = th.XGB.DODepr$prob_threshold[51]


# helper function for the plots
tuneplot <- function(x, probs = .90){
  ggplot(x) +
    coord_cartesian(ylim = c(0.0, 0.5)) +
    theme_bw()
}

tuneplot(XGB.DODepr)
print(XGB.DODepr)
VI <- varImp(XGB.DODepr, scale =FALSE)
plot(VI)

x <- evalm(XGB.DODepr)

confusionMatrix(XGB.DODepr$pred$pred, XGB.DODepr$pred$obs, positive = 'worse')
confmat.XGB.DODepr <- confusionMatrix(XGB.DODepr, mode ='everything')
ppv.XGB.DODepr <- confmat.XGB.DODepr$table[4]/(confmat.XGB.DODepr$table[2]+confmat.XGB.DODepr$table[4])
sens.XGB.DODepr <- confmat.XGB.DODepr$table[4]/(confmat.XGB.DODepr$table[3]+confmat.XGB.DODepr$table[4])

# evaluate all models --------------------------------------------
model_list.DODepr <- list(Lasso = LassoFit.DODepr, Ridge = RidgeFit.DODepr, ElasticNet= NetFit.DODepr, RanDODeprmForrest = RF.DODepr, ClassificationTree = CT.DODepr, ExtremeGradientBoost = XGB.DODepr)
res <- resamples(model_list.DODepr) # part of caret package to compare the models
summary(res) # tabel overview of models
bwplot(res) #boxplot
xyplot(res, metric = 'ppv')

evalm(model_list.DODepr)
evalm(model_list.Depr)
evalm(model_list.DO)
#save.image("Z:/inbox/WorkInProgress/image270121.RData")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# # Super Learner
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# set.seed(1)
# listWrappers()
# # change data to numeric 
# InputDODeprNum <- InputDODepr %>% mutate_if(is.logical,as.numeric)
# InputDODeprNum <- InputDODeprNum %>% mutate_if(is.factor,as.numeric)
# InputDODeprNumY <- InputDODeprNum$Y
# InputDODeprNumY[InputDODeprNumY==2] <- 0
# InputDODeprNumX <- InputDODeprNum %>% select(-starts_with('Y'))
# 
# 
# #RF tuning
# tune.RF =list(mtry_seq = c(1:10))
# #XGB tuning
# tune.xgb = list(ntrees = c(10, 25, 50, 100,250),
#                 max_depth = c(1,2,5),
#                 shrinkage = c(0.0001, 0.001, 0.01, 0.015),
#                 minobspernode = c(1,2,4,8))
# 
# 
# learners.RF.DODepr  = create.Learner(c("SL.ranger"), tune = tune.RF)
# learners.XGB.DODepr = create.Learner(c("SL.xgboost"), tune = tune.xgb, detailed_names = TRUE, name_prefix = "xgb" )
# 
# cv_sl.DODepr = CV.SuperLearner(Y = InputDODeprNumY, X = InputDODeprNumX, family=binomial(), 
#                            V = 5,
#                            method = "method.AUC",
#                            SL.library = c("SL.mean", "SL.glmnet", learners.RF.DODepr$names, learners.XGB.DODepr$names,"SL.ranger"))
# summary(cv_sl.DODepr)
# plot(cv_sl.DODepr) + theme_bw()
# 
# # calculate positive predictive value from superlearner
# pred.sl.DODepr = predict.SL.xgboost(cv_sl.DODepr, InputDODeprNumX, X=NULL, Y=nULL, onlySL = FALSE)
# str(pred.sl.DODepr)
# qplot(pred.sl.DODepr[,1])+theme_minimal()
# qplot(InputDODeprNumY,pred.sl.DODepr[,1])+theme_minimal()
# pred_rocr = ROCR::prediction(pred.sl.DODepr$pred, InputDODeprNumY)
# auc = ROCR::perfomance(pred_rocr, measure="AUC", x.measure = "cutoff")@y.values[[1]]
# auc
# 
# # assess best models pivottable
# table(cv_sl.DODepr$library.predict[,3]<.5, InputDODeprNumY)
# table(cv_sl.DODepr$SL.predict<.5, InputDODeprNumY)
# 


# save image
save.image("Z:/inbox/WorkInProgress/image070221PR.RData")

Freq(InputDODepr$Y)
Freq(InputDepr$Y)
Freq(InputDO$Y)


evalm(model_list.Depr)
evalm(model_list.DO)
evalm(model_list.DODepr)

#  #  #  #  #  #   #  #  #   #  #  #   #  #  #  

## apply on test_data --------------------------------------------------------

#  #  #   #  #  #   #  #  #   #  #  #   #  #  #   

# load testdata
# From datasplit file write.csv(df_testDO,"Z:/inbox/TestSet/df_testDO")
df_test <- read.csv("Z:/inbox/TestSet/df_testDO")


# A. Prepare TEST data ---------------------------------------------------


## Impute data #################################################################
select <- dplyr::select
# for demographics add NA group --> No NAs anymore at this point (group for missing)

# for questionnaires impute rounded median
df_test %>% select(starts_with("Q.")) %>% summarise_all(funs(sum(is.na(.))))

# df_train <- df_train %>% transmute_if(is.integer, funs(as.numeric(as.integer(.))))
# str(df_train)
# medians.Q <- df_train%>% select(starts_with("Q.")) %>% summarise(across(everything(),~ median(., na.rm =TRUE)))

#loop over Q's and replace missings with medians from training dataset
for(i in 1:length(medians.Q)){
  df_test <- df_test %>% mutate_at(vars(paste0(colnames(medians.Q)[i])), ~ifelse(is.na(.), medians.Q[[i]], .))
}
#df_test <- df_test %>% mutate_at(vars(starts_with("Q.")), ~ifelse(is.na(.), medians.Q, .))

df_test %>% select(starts_with("Q.")) %>% summarise_all(funs(sum(is.na(.))))

# for DSM imputation NA --> no / 0 
df_test %>% select(starts_with("DSM.")) %>% summarise_all(funs(sum(is.na(.))))
df_test <- df_test %>% mutate_at(vars(starts_with("DSM.")), ~ifelse(is.na(.),0,.))

# for Dem. imputation NA --> no / 0 
df_test %>% select(starts_with("Dem.")) %>% summarise_all(funs(sum(is.na(.))))
df_test <- df_test %>% mutate_at(vars(starts_with("Dem.")), ~ifelse(is.na(.),0,.))

# for Clin
# df_train %>% select(starts_with("Clin")) %>% str()
df_test %>% select(starts_with("Clin.")) %>% summarise_all(funs(sum(is.na(.))))
df_test <- df_test %>% mutate_at(vars(starts_with("Clin."),-starts_with("Clin.Intox.Alcohol")), ~ifelse(is.na(.),0,.))
df_test <- df_test %>% mutate_at(vars(starts_with("Clin.Intox.Alcohol")), ~ifelse(is.na(.), 1, .))
df_test %>% select(starts_with("Clin.")) %>% summarise_all(funs(sum(is.na(.))))

# Check near zero variance
## Identify near-zero variances predictors
# nzvar <- nearZeroVar(Input, saveMetrics = TRUE)
# nzvarnm <- row.names(nzvar[nzvar$nzv,])

# remove near-zero variancd variables --> DSM.ASD and Clin.Med.lithium
# for (i in nzvarnm){
#   Input <- Input%>%select(-c(i))}

# Select predictors
InputTest <- df_test%>%select((starts_with("DSM.")|starts_with("Clin.")|starts_with("Q.")|starts_with("Dem."))&
                             -starts_with("ID")&
                             -starts_with("DSM.ASD")&
                             -starts_with("Clin.Med.Lithium")&
                             -starts_with("Y") )

# make sure everything in input is offered in same dataform

InputTest <- InputTest%>% mutate_at(vars(starts_with("Dem.work")|starts_with("Dem.maritalstatus")|starts_with("Dem.matitalstatus")), as.numeric)

#check for nas
colnames(InputTest)[colSums(is.na(InputTest))>0]

## a. make appropriate outcome vector Y ========================================

# We need 3 outcome vectors for the three different models
#   1: predicting whether subjects deteriorate in terms of depressive symptoms
#   2: predicting whether subjects will drop out
#   3: predicting whether subjects will either deteriorate and/or will drop out
#           I. print pivottable to determine base-rate / positive predictive value
#       b. select appropriate case.names

# Select output and cases for 1.Depr ############################################

# construct output
InputDeprbTest <- InputTest
YAbsTest <- df_test$Q.BDI.total.pre - df_test$Qout.BDI.total.post
YRelTest <- YAbsTest/df_test$Q.BDI.total.pre
#YRel0Test <- vector(mode="logical", length = length(YRelTest))
#YRel0[df_test$Q.BDI.total.pre<13& !is.na(YAbs)] <- YAbs[df_test$Q.BDI.total.pre<13 & !is.na(YAbs)]<(-3) # everyone who does not score 'depressed', 9 point in symptom as relevant, arbitrary. 
YRelAllTest <- YRelTest<(-.14)
YRelAllTest[df_test$Q.BDI.total.pre<13] <- YAbsTest[df_test$Q.BDI.total.pre<13]<(-3)# based on Button et al, psych med, 2016 MCI difference on the BDI-II - according to the patient's perspective
InputDeprbTest$Y <- as.factor(YRelAllTest)
levels(InputDeprbTest$Y) <- c("worse","SameBetter")



# select data: only those with pre and post measurement
InputDeprTest <- InputDeprbTest[  !is.na(df_test$Q.BDI.total.pre) & 
                                    !is.na(df_test$Qout.BDI.total.post) & !is.na(InputDeprbTest$Y),]
YAbssemitest <- YAbsTest[  !is.na(df_test$Q.BDI.total.pre) & 
                             !is.na(df_test$Qout.BDI.total.post) & !is.na(InputDeprbTest$Y)]
YRelAllTest <- YRelAllTest[  !is.na(df_test$Q.BDI.total.pre) & 
                               !is.na(df_test$Qout.BDI.total.post) & !is.na(InputDeprbTest$Y)]

ggplot(data = InputDeprTest,aes(x=Q.BDI.total.pre, y=YAbssemitest, group = YRelAllTest)) +
  geom_point(aes(color = YRelAllTest))

print('base PPV =')
Freq(InputDepr$Y)

#check for nas
colnames(InputDeprTest)[colSums(is.na(InputDeprTest))>0]

print('base PPV =')
Freq(InputDeprTest$Y)

# Select output andcases for 2. DO ######################################################
# select data: all participants for whom numbses is known ###############################
InputDOTest <- Input[!is.na(df_test$Numbses),]

# construct output
InputDOTest$Y <- as.factor(df_test$Numbses<5)
levels(InputDOTest$Y) <- c("In","worse")

print('base PPV =')
Freq(InputDOTest$Y)

# Select output andcases for 3.DoDepr ###################################################

# construct output
InputDODeprTest <- InputTest
InputDODeprTest$Y <- as.factor(YRelAll | df_test$Numbses<5)
levels(InputDODeprTest$Y) <- c("AdviseIn","worse")

# select data: all participants for whom numbses <5 OR those for whom pre and post BDI is known
InputDODeprTest <- InputDODeprTest[df_test$Numbses<5|
                             (!is.na(df_test$Q.BDI.total.pre) & !is.na(df_test$Qout.BDI.total.post))&
                             !is.na(InputDODeprTest$Y),]

print('base PPV =')
Freq(InputDODeprTest$Y)

## Exploratory datavisualisation ##############################################


# Check for outliers with correlation matrix for numeric predictors
InputDODeprTest %>% select(Q.OQ.symptomdistress, Q.OQ.socialrole, Q.OQ.interpersonalrelations,
                       Q.BDI.cognitive, Q.BDI.somatic, Q.BDI.affective,
                       Q.PSWQ.worry, Q.PSWQ.AbsenceOfWorry,
                       Q.FFMQ.observe,Q.FFMQ.describe,Q.FFMQ.actawareness,Q.FFMQ.nonjudging,Q.FFMQ.nonreactivity,
                       Q.SCS.selfjudgement,Q.SCS.commonhumanity,Q.SCS.mindfulness,
                       Q.SCS.overidentified,Q.SCS.selfkindness,Q.SCS.isolation,
                       Dem.age, Clin.Intox.Alcohol)%>%
  chart.Correlation(.,histogram=TRUE, method = 'pearson', pch=19)


# assess whether there are obvious differences between the outcome groups

## create a vecor specifying the variable names:
myvars <- names(InputDeprTest)
myvars <- myvars[myvars!="Y"]
## select all except Y and paste them together with the + sign
IndVars <- paste(myvars, collapse="+")

tab1.Depr <- tableby(as.formula(paste('Y~',IndVars)),data=InputDeprTest)
df <- as.data.frame(tab1.DeprTest)
print(df[df$p.value<.05,])

tab1.DO <- tableby(as.formula(paste('Y~',IndVars)),data=InputDOTest)
df <- as.data.frame(tab1.DOTest)
print(df[df$p.value<.05,])

tab1.DODepr <- tableby(as.formula(paste('Y~',IndVars)),data=InputDODeprTest)
df <- as.data.frame(tab1.DODeprTest)
print(df[df$p.value<.05,])

set.seed(12345)
NetFit.Depr.test <- predict(NetFit.Depr, newdata = InputDeprTest, type ="prob")
confusionMatrix(as.factor(NetFit.Depr.test$worse<(threshold.NetFit.Depr) ), as.factor(InputDeprTest$Y=='worse'), positive = 'TRUE')

plotdata = cbind(YAbsTest,NetFit.Depr.test,InputDeprTest)
ggplot(data = plotdata,aes(x=Q.BDI.total.pre, y=YAbsTest, group = worse<(threshold.NetFit.Depr))) +
  geom_point(aes(color = worse<(threshold.NetFit.Depr)))

set.seed(12345)
XGB.Depr.test <- predict(XGB.Depr, newdata = InputDeprTest, type ="prob")
confusionMatrix(as.factor(XGB.Depr.test$worse<(threshold.XGB.Depr)), as.factor(InputDeprTest$Y=='worse'), positive = 'TRUE')

hist(YAbsTest[XGB.Depr.test$worse<(threshold.XGB.Depr)])
hist(YAbsTest[XGB.Depr.test$worse>=(threshold.XGB.Depr)])

plotdata = cbind(YAbsTest,XGB.Depr.test,InputDeprTest)
ggplot(data = plotdata,aes(x=Q.BDI.total.pre, y=YAbsTest, group = worse<(threshold.XGB.Depr))) +
  geom_point(aes(color = worse<(threshold.XGB.Depr), size = 1 ))


set.seed(12345)
XGB.DODepr.test <- predict(XGB.DODepr, newdata = InputDODeprTest, type ="prob")
confusionMatrix(as.factor(XGB.DODepr.test$worse<(threshold.XGB.DODepr)), as.factor(InputDODeprTest$Y=='worse'), positive = 'TRUE')

plotdata = cbind(XGB.DODepr.test,InputDODeprTest)
ggplot(data = plotdata,aes(x=Q.BDI.total.pre, y=Y, group = worse<(threshold.XGB.DODepr))) +
  geom_point(aes(color = worse<(threshold.XGB.DODepr)))
