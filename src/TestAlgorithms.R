### TestAlgorithms.R --- 
## 
## Filename: TestAlgorithms.R
## Description: 
## Author: Sergio-Feliciano Mendoza-Barrera
## Maintainer: 
## Created: Mon Aug  3 08:50:53 2015 (-0500)
## Version: 
## Package-Requires: ()
## Last-Updated: Mon Aug  3 09:25:41 2015 (-0500)
##           By: Sergio-Feliciano Mendoza-Barrera
##     Update #: 41
## URL: 
## Doc URL: 
## Keywords: 
## Compatibility: 
## 
######################################################################
## 
### Commentary: 
## 
## 
## 
######################################################################
## 
### Change Log:
## 
## 
######################################################################
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or (at
## your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.
## 
######################################################################
## 
### Code:

library(caret)
library(mboost)
library(pROC)
library(AER)
library(party)
library(earth)

data(CreditCard)

CreditCard$Class <- CreditCard$card
CreditCard <- subset(CreditCard, select=-c(card, expenditure, share))
set.seed(1984)
training <- createDataPartition(CreditCard$Class, p = 0.6, list=FALSE)

trainData <- CreditCard[training,]
testData <- CreditCard[-training,]

## GLM model
glmModel <- glm(Class~ . , data=trainData, family=binomial)
pred.glmModel <- predict(glmModel, newdata=testData, type="response")

roc.glmModel <- pROC::roc(testData$Class, pred.glmModel)
auc.glmModel <- pROC::auc(roc.glmModel)
auc.glmModel[1]

## glmboost

fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(2014)

glmBoostModel <- train(Class ~ ., data = trainData, method = "glmboost",
                       metric = "ROC", trControl = fitControl,
                       tuneLength = 5, center = TRUE,
                       family = Binomial(link = c("logit")))

pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=testData, type="prob")[,"yes"])
roc.glmBoostModel <- pROC::roc(testData$Class, pred.glmBoostModel)
auc.glmBoostModel <- pROC::auc(roc.glmBoostModel)
auc.glmBoostModel[1]

## CART
set.seed(2014)
cartModel <- train(Class ~ ., data = trainData, method = "rpart",
                   metric = "ROC", trControl = fitControl, tuneLength
                   = 5)

pred.cartModel <- as.vector(predict(cartModel, newdata=testData, type="prob")[,"yes"])
roc.cartModel <- pROC::roc(testData$Class, pred.cartModel)
auc.cartModel <- pROC::auc(roc.cartModel)
auc.cartModel[1]

## Conditional Inference Tree
set.seed(2014)
partyModel <- train(Class ~ ., data = trainData, method = "ctree",
                    metric = "ROC", trControl = fitControl, tuneLength
                    = 5)

pred.partyModel <- as.vector(predict(partyModel, newdata=testData, type="prob")[,"yes"])
roc.partyModel <- pROC::roc(testData$Class, pred.partyModel)
auc.partyModel <- pROC::auc(roc.partyModel)
auc.partyModel[1]

## Elastic Net
set.seed(2014)
eNetModel <- train(Class ~ ., data = trainData, method = "glmnet",
                   metric = "ROC", trControl = fitControl,
                   family = "binomial", tuneLength = 5)

pred.eNetModel <- as.vector(predict(eNetModel, newdata=testData, type="prob")[,"yes"])
roc.eNetModel <- pROC::roc(testData$Class, pred.eNetModel)
auc.eNetModel <- pROC::auc(roc.eNetModel)
auc.eNetModel[1]

## Earth
set.seed(2014)
earthModel <- train(Class ~ ., data = trainData, method = "earth",
                    glm = list(family=binomial), metric = "ROC",
                    trControl = fitControl, tuneLength = 5)

pred.earthModel <- as.vector(predict(earthModel, newdata=testData, type="prob")[,"yes"])
roc.earthModel <- pROC::roc(testData$Class, pred.earthModel)
auc.earthModel <- pROC::auc(roc.earthModel)
auc.earthModel[1]

## Boosted Trees
set.seed(2014)
gbmModel <- train(Class ~ ., data = trainData, method = "gbm",
                  metric = "ROC", trControl = fitControl,
                  verbose = FALSE, tuneLength = 5)

pred.gbmModel <- as.vector(predict(gbmModel, newdata=testData, type="prob")[,"yes"])
roc.gbmModel <- pROC::roc(testData$Class, pred.gbmModel)
auc.gbmModel <- pROC::auc(roc.gbmModel)
auc.gbmModel[1]

## Random Forest
set.seed(2014)
rfModel <- train(Class ~ ., data = trainData, method = "rf", metric = "ROC",
                 trControl = fitControl, verbose=FALSE, tuneLength = 5)

pred.rfModel <- as.vector(predict(rfModel, newdata=testData, type="prob")[,"yes"])
roc.rfModel <- pROC::roc(testData$Class, pred.rfModel)
auc.rfModel <- pROC::auc(roc.rfModel)
auc.rfModel[1]



######################################################################
### TestAlgorithms.R ends here
