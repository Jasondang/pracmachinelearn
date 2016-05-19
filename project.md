# Practical Machine Learning Course Project

#### By Jason Dang
#### 20 May 2016, Melbourne Australia

##Executive Summary
Devices such as the Fitbit and Nike FuelBand are able to collect a large amount of data about personal activity. In this project the data provided were collected from accelerometers on the belt, forearm, arm and dumbell of 6 participant. They were asked to perform the barbell lifts correctly and incorrectly in 5 differnt ways. 

The goal of this project is to predict the manner in which the exercise was performed. This is indicated by the "classe" variable in the training set. 

##Initial Setup
The following packages are required for the completion of this project and hence they were loaded via the following code:
```{r echo = TRUE}
library(caret)
library(AppliedPredictiveModeling)
library(rpart)
library(rattle)
library(randomForest)
```

##Data Preprocessing
The training and testing dataset can be collected from the following links:
```{r echo=TRUE}
#Training Dataset:
## https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

#Testing Dataset:
## https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
```

In this project, the data was previously downloaded and read in the directory of a local computer. 
```{r echo=TRUE}
training <- read.csv("~/RProgramming/Practical Machine Learning/ProjectData/pml-training.csv")
testing <- read.csv("~/RProgramming/Practical Machine Learning/ProjectData/pml-testing.csv")

```

For reproducibility, the seed was set to 10
```{r echo=TRUE}
set.seed(10)
```

###Cleaning the data
Numerous steps were performed to clean the data appropriately before data analysis could be performed. The first step was to remove the first 7 columns of the data as they were irrelevent information for the analysis. The reason for this was because the data contained data related to time as our analysis will not be dependent on this variable. 
```{r echo=TRUE}
#Remove the first 7 columns of the dataset as they are irrelevent to the data analysis
training <- training[c(-1:-7)]
```

The next step is to remove columns with an excessive number of NA values, in this case, the threshold was set at 60%
```{r echo=TRUE}
#Remove columns with NA values
training_1 <- training
index_na <- c()
for(i in 1:length(training_1)) {
  if (sum(is.na(training_1[,i]))/nrow(training_1) > 0.60) {
    index_na <- c(index_na, i)
  }
}
training <- training[-index_na]
```

The final step is to remove Near Zero Variance varibles:
```{r echo=TRUE}
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[, nzv$nzv==FALSE]
```

Once the data has been cleaned, it can now be split into 2 subsets, a training subset and a testing subset. 60% of the original data will be subsetted into the training subset with the remaining 40% in the testing subset. 
```{r echo=TRUE}
#Split each of the subset of data into training and testing groups
set.seed(10)
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
training1 <- training[inTrain,]
testing1 <- training[-inTrain,]
```


##Data Analysis

###Classification Tree
The first model chosen for the analysis of the data was the classification tree using the Rpart method. The following code was executed for this model. 
```{r}
#First method Rpart
set.seed(10)
modRpart <- rpart(classe~., data=training1, method="class")
fancyRpartPlot(modRpart)
```

To analyse the accuracy of this model, it was tested against the testing subset of the original data using the following code:
```{r}
predRpart <- predict(modRpart, testing1, type="class")
confusionMatrix(predRpart, testing1$classe)
```

From this it can be seen that this model was approximately 73% accurate on the testing dataset. Ideally we would like this to be much higher. 

###Random Forest
The second modelling method chosen was the random forest method. The following code was executed for this model:
```{r echo=TRUE}
set.seed(10)
modRfor <- randomForest(classe ~., data=training1)
```

The following results were achieved on the testing dataset:
```{r echo=TRUE}
predRfor <- predict(modRfor, testing1, type="class")
confusionMatrix(predRfor, testing1$classe)
```

It is clear that the random forest method improve the accuracy by over 25% to over 99% compared to the classifiction tree model. However there is still one more model to consider before making a final decision

###Boosting
The third and final modelling method chosen was the Boosting. The following code was executed for this model: 
```{r echo=TRUE}
#Third method boosting
modBoo <- train(classe~., data=training1, method="gbm", trControl=trainControl(method = "repeatedcv",number = 5),verbose=FALSE)
```

The following results were achieved:
```{r echo= TRUE}
predBoo <- predict(modBoo, testing1)
confusionMatrix(predBoo, testing1$classe)
```

While the Boosting method was significantly better then the classification tree, it was slightly less accurate than the random forest method.

###Conclusion
To recap, the accuracy of the 3 models chosen were as follows
```{r echo=TRUE}
confusionMatrix(predRpart, testing1$classe)$overall[1]
confusionMatrix(predRfor, testing1$classe)$overall[1]
confusionMatrix(predBoo, testing1$classe)$overall[1]
```

For the purpose of predicting the testing set given for this project, the random forest was the chosen model. 

###Final Prediction
The results for the final prediction were as follows:
```{r echo=TRUE}
finalPred <- predict(modRfor, testing, type="class")
finalPred
```

The predictions were written to a test file when the following code was executed:
```{r echo=TRUE}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(finalPred)
```

