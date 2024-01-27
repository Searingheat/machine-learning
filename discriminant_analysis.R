

Comp_priors <- function(train_labels) {
  #' Compute the priors of each class label 
  #' 
  #' @param train_labels a vector of labels with length equal to n
  #' @return a probability vector of length K = 10
  
  
  K <- 10
  pi_vec <- rep(0, K)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  label1 <- 0:9
  n <- length(train_labels)
  for (i in 1:length(label1)) {
    pi_vec[i] <- sum(train_labels == label1[i])/n

  }
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(pi_vec)
}
  


Comp_cond_means <- function(train_data, train_labels) {
  #' Compute the conditional means of each class 
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' 
  #' @return a p by 10 matrix, each column represents the conditional mean given
  #'   each class.
  
  K <- 10
  p <- ncol(train_data)
  mean_mat <- matrix(0, p, K)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
 
  label1 <- 0:9
  for (i in (1:K)) {
    n_k <- sum(train_labels == label1[i])
    x <- rep(0, p)
    for (k in 1:length(train_labels)) {
      if (train_labels[k] == label1[i]) {
        x <- x + train_data[k, ]
      }
    }
    mean_mat[, i] <- (1/ n_k) * x
  }
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(mean_mat)
}



Comp_cond_covs <- function(train_data, train_labels, cov_equal = FALSE) {
  #' Compute the conditional covariance matrix of each class
  #' 
  #' @param train_data a n by p matrix containing p features of n training points
  #' @param train_labels a vector of labels with length equal to n
  #' @param cov_equal TRUE if all conditional covariance matrices are equal, 
  #'   otherwise, FALSE 
  #' 
  #' @return 
  #'  if \code{cov_equal} is FALSE, return an array with dimension (p, p, K),
  #'    containing p by p covariance matrices of each class;
  #'  else, return a p by p covariance matrix. 
  
  K <- 10
  p <- ncol(train_data)
  
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  
  mean_mat <- Comp_cond_means(train_data, train_labels)
  n <- length(train_labels)
  label1 <- 0:9
  
  if (cov_equal == TRUE) {
    cov_arr <- matrix(0, p, p)
    for (l in 1:K) {
      cov_tem <- matrix(0, p, p)
      for (j in 1:length(train_labels)) {
        if (train_labels[j] == label1[l]) {
          cov_tem <- cov_tem + (train_data[j, ] - mean_mat[, l]) %*% t(train_data[j, ] - mean_mat[, l])
        }
      }
      cov_arr <- cov_arr + cov_tem
    }
    cov_arr <- (1/ (n-K)) * cov_arr
  }
  
  if (cov_equal == FALSE) {
    cov_arr <- array(0, dim = c(p, p, K))
    for (l in 1:K) {
      n_k <- sum(train_labels == label1[l])
      cov_tem <- matrix(0, p, p)
      for (j in 1:length(train_labels)) {
        if (train_labels[j] == label1[l]) {
          cov_tem <- cov_tem + (train_data[j, ] - mean_mat[, l]) %*% t(train_data[j, ] - mean_mat[, l])
        }
      }
      cov_arr[, , l] <- (1/(n_k - 1)) * cov_tem
    }
  }
  
  return(cov_arr)
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
}




Predict_posterior <- function(test_data, priors, means, covs, cov_equal) {
  
  #' Predict the posterior probabilities of each class 
  #'
  #' @param test_data a n_test by p feature matrix 
  #' @param priors a vector of prior probabilities with length equal to K
  #' @param means a p by K matrix containing conditional means given each class
  #' @param covs covariance matrices of each class, depending on \code{cov_equal}
  #' @param cov_equal TRUE if all conditional covariance matrices are equal; 
  #'   otherwise FALSE.
  #'   
  #' @return a n_test by K matrix: each row contains the posterior probabilities 
  #'   of each class.
  
  n_test <- nrow(test_data)
  K <- length(priors)
  posteriors <- matrix(0, n_test, K)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  library(mvtnorm)
  if (cov_equal == TRUE) {
    den <- rep(0, n_test)
    for (i in 1:n_test) {
      num <- 0
      for (j in 1:K) {
        num <- num + priors[j] * dmvnorm(x = test_data[i, ], mean = means[, j], sigma = covs)
      }
      den[i] <- num
    }
    for (i in 1: n_test) {
      for (j in 1:K) {
        posteriors[i, j] <- priors[j] * dmvnorm(x = test_data[i, ], mean = means[, j], sigma = covs) / den[i]
      }
    }
  }
  
  if (cov_equal == FALSE) {
    den <- rep(0, n_test)
    for (i in 1:n_test) {
      num <- 0
      for (j in 1:K) {
        num <- num + priors[j] * dmvnorm(x = test_data[i, ], mean = means[, j], sigma = covs[, , j])
      }
      den[i] <- num
    }
    for (i in 1: n_test) {
      for (j in 1:K) {
        posteriors[i, j] <- priors[j] * dmvnorm(x = test_data[i, ], mean = means[, j], sigma = covs[, , j]) / den[i]
      }
    }
  }
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(posteriors)
}


Predict_labels <- function(posteriors) {
  
  #' Predict labels based on the posterior probabilities over K classes
  #' 
  #' @param posteriors A n by K posterior probabilities
  #' 
  #' @return A vector of predicted labels with length equal to n
  
  n_test <- nrow(posteriors)
  pred_labels <- rep(NA, n_test)
  
  #####################################################################
  #  TODO                                                             #
  #####################################################################
  
  pred_labels <- apply(posteriors, 1, which.max) - 1
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  
  return(pred_labels)
}




