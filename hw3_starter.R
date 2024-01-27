rm(list = ls())

## You should set the working directory to the folder of hw3_starter by
## uncommenting the following and replacing YourDirectory by what you have
## in your local computer / labtop

# setwd("YourDirectory/hw3_starter")

## Load utils.R and penalized_logistic_regression.R

source("utils.R")
source("penalized_logistic_regression.R")



## load data sets

train <- Load_data("./data/train.csv")
valid <- Load_data("./data/valid.csv")
test <- Load_data("./data/test.csv")

x_train <- train$x
y_train <- train$y

x_valid <- valid$x
y_valid <- valid$y

x_test <- test$x
y_test <- test$y


### Visualization 
## uncomment the following command to visualize the first five and 301th-305th 
##  digits in the training data. 
# Plot_digits(c(1:5, 301:305), x_train)




#####################################################################
#                           Part a.                                 #
# TODO: Find the best choice of the hyperparameters:                #
#     - stepsize (i.e. the learning rate)                           #
#     - max_iter (the maximal number of iterations)                 #
#   The regularization parameter, lbd, should be set to 0           #
#   Draw plot of training losses and training 0-1 errors            #
#####################################################################

par(mfrow=c(2,2))
lbd = 0
stepsize <- c(0.001, 0.03)
max_iter <- c(500, 2500)
for (i in 1:length(stepsize)) for (j in 1:length(max_iter))  { 
  output <- Penalized_Logistic_Reg(x_train = x_train, y_train = y_train, lbd = lbd, stepsize = stepsize[i], max_iter = max_iter[j])
  plot(x = 1 : max_iter[j], y = output$loss, main = paste0("stepsize=", stepsize[i], ", max_iter=", max_iter[j]))
  plot(x = 1 : max_iter[j], y = output$error, main = paste0("stepsize=", stepsize[i], ", max_iter=", max_iter[j]))
}

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################






#####################################################################
#                           Part b.                                 #
# TODO: Identify the best stepsize and max_iter for each lambda     #
#       from the given grid. Draw the plots of training and         #
#       validation 0-1 errors versus different values of lambda     #
#####################################################################


stepsize <- 0.03  # this should be replaced by your answer in Part a
max_iter <- 2500  # this should be replaced by your answer in Part a

lbd_grid <- c(0, 0.01, 0.05, 0.1, 0.5, 1)
train_error <- valid_error <- numeric(length(lbd_grid))
for (i in 1:length(lbd_grid)) {
  output_train <- Penalized_Logistic_Reg(x_train = x_train, y_train = y_train, lbd = lbd_grid[i], stepsize = stepsize, max_iter = max_iter)
  output_valid <- Penalized_Logistic_Reg(x_train = x_valid, y_train = y_valid, lbd = lbd_grid[i], stepsize = stepsize, max_iter = max_iter)
  train_error[i] <- output_train$error[max_iter]
  valid_error[i] <- output_valid$error[max_iter]
}
par(mfrow = c(1,2))
  plot(x = lbd_grid, y = train_error, pch = 20)
  plot(x = lbd_grid, y = valid_error, pch = 20)




#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################







#####################################################################
#                           Part c.                                 #
# TODO: using the best stepsize,  max_iter and lbd you found, fit   # 
#  the penalized logistic regression and compute its test 0-1 error #
#####################################################################


stepsize <- 0.3  # this should be replaced by your answer in Part a
max_iter <- 2500  # this should be replaced by your answer in Part a
lbd <- 0       # this should be replaced by your answer in Part b

output1 <- Penalized_Logistic_Reg(x_train = x_train, y_train = y_train, lbd = lbd, stepsize = stepsize, max_iter = max_iter)
beta1 <- output1$beta1
beta0 <- output1$beta0
pred_class <- Predict_logis(data_feature = x_test, beta1 = beta1, beta0 = beta0, type = "class")
error <- Evaluate(true_label = y_test, pred_label = pred_class)
error
library(glmnet)
ridge.mod <- glmnet(x_train, y_train, alpha = 0, lambda = lbd/2)
pred.ridge <- predict(ridge.mod, s = lbd, newx = x_test)
pred.class <- ifelse(pred.ridge > 0.5, 1, 0)
mean(pred.class != y_test)

#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################




