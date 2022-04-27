library(tidyverse)
library(rjags)

set.seed(31618)  # For reproducibility

# Set the number of iterations for the MCMC sampler
burn    <- 10000
iters   <- 50000
thin    <- 10

# Load the data
Y <- log(df$aav)
X <- cbind(df$contractLength, df$draftRound, df$draftOverall, df$draftYear,
           df$X3P., df$X2P., df$FT., df$AST, df$STL, df$BLK, df$TOV, df$PTS)
# Notes about predictor variables:
# Removed age, height, and weight as their coefficients would not converge
# Removed FG. and TS. as coefficients would not converge in Models 3 and 4

# Standardize the data
X <- scale(X)
X <- cbind(1, X)

# Assign a numeric id to the players in each team
team <- unique(df$Tm)
id <- rep(NA, length(Y))
for(j in 1:48){
  id[df$Tm == team[j]] <- j
}

n <- length(Y)       # number of observations
p <- ncol(X)        # number of covariates (including the intercept)
N <- length(team)    # number of teams

##############################################################################
#
# Model 1: Constant slopes using Gaussian priors
#
##############################################################################

# Specify the FIRST model
model_string1 <- textConnection("model{

  # Likelihood
  for(i in 1:n){
    Y[i] ~ dnorm(inprod(X[i,], beta[]), taue)
  }

  # Priors
  beta[1] ~ dnorm(0, 0.001)
  for(j in 2:p){
    beta[j] ~ dnorm(0, taub*taue)
  }
  taue ~ dgamma(0.1, 0.1)
  taub ~ dgamma(0.1, 0.1)
}")

data    <- list(Y=Y, X=X, n=n, p=p)
model1   <- jags.model(model_string1, data = data, quiet=TRUE, n.chains=2)
update(model1, burn, progress.bar="none")

samples1 <- coda.samples(model1,
                         variable.names = c("beta", "taue", "taub"),
                         n.iter = iters,
                         #thin=thin,
                         progress.bar = "none")

summary1 <- summary(samples1)

##############################################################################
#
# Model 2: Constant slopes using double exponential priors
#
##############################################################################

# Specify the SECOND model
model_string2 <- textConnection("model{

  # Likelihood
  for(i in 1:n){
    Y[i] ~ dnorm(inprod(X[i,], beta[]), taue)
  }

  # Priors
  beta[1] ~ dnorm(0, 0.001)
  for(j in 2:p){
    beta[j] ~ ddexp(0, taub*taue)
  }
  taue ~ dgamma(0.1, 0.1)
  taub ~ dgamma(0.1, 0.1)
}")

data    <- list(Y=Y, X=X, n=n, p=p)
model2   <- jags.model(model_string2, data = data, quiet=TRUE, n.chains=2)
update(model2, burn, progress.bar="none")

samples2 <- coda.samples(model2,
                         variable.names=c("beta", "taue", "taub"),
                         n.iter = iters,
                         #thin=thin,
                         progress.bar = "none")

summary2 <- summary(samples2)

##############################################################################
#
# Model 3: Slopes as fixed effects
#
##############################################################################

model_string3 <- textConnection("model{

  # Likelihood
  for(i in 1:n){
    Y[i] ~ dnorm(mnY[i],taue)
    mnY[i] <- inprod(X[i,],beta[id[i],])
   }

  # Slopes
  for(j in 1:p){
    for(i in 1:N){
      beta[i,j] ~ dnorm(0,0.01)
    }
  }

  # Priors
  taue ~ dgamma(0.1,0.1)


  # WAIC calculations
  for(i in 1:n){
    like[i] <- dnorm(Y[i],mnY[i],taue)
  }
}")

data <- list(Y=Y,n=n,N=N,X=X,p=p,id=id)
model3   <- jags.model(model_string3, data = data, quiet=TRUE, n.chains=2)
update(model3, burn, progress.bar="none")

samples3 <- coda.samples(model3,
                         variable.names=c("beta","taue"),
                         n.iter=iters,
                         #thin=thin,
                         progress.bar="none")

summary3 <- summary(samples3)

##############################################################################
#
# Model 4: Slopes as random effects
#
##############################################################################

model_string4 <- textConnection("model{

  # Likelihood
  for(i in 1:n){
    Y[i] ~ dnorm(mnY[i],taue)
    mnY[i] <- inprod(X[i,],beta[id[i],])
   }

  # Random slopes
  for(j in 1:p){
    for(i in 1:N){
      beta[i,j] ~ dnorm(mu[j],taub[j])
    }
    mu[j]    ~ dnorm(0,0.01)
    taub[j]  ~ dgamma(0.1,0.1)
  }

  # Priors
  taue ~ dgamma(0.1,0.1)


  # WAIC calculations
  for(i in 1:n){
    like[i] <- dnorm(Y[i],mnY[i],taue)
  }
}")

data <- list(Y=Y,n=n,N=N,X=X,p=p,id=id)
model4   <- jags.model(model_string4,data = data,quiet=TRUE, n.chains=2)
update(model4, burn, progress.bar="none")

samples4 <- coda.samples(model4,
                         variable.names=c("beta","taue"),
                         n.iter=iters,
                         #thin=thin,
                         progress.bar="none")

summary4 <- summary(samples4)

# Effective Sample Size > 1000 indicates convergence
ESS1 <- effectiveSize(samples1)
ESS2 <- effectiveSize(samples2)
ESS3 <- effectiveSize(samples3)
ESS4 <- effectiveSize(samples4)

# Geweke statistic < 2 indicates convergence
geweke1 <- geweke.diag(samples1[[1]])
geweke2 <- geweke.diag(samples2[[1]])
geweke3 <- geweke.diag(samples3[[1]])
geweke4 <- geweke.diag(samples4[[1]])

# Compare DIC for all models
dic1 <- dic.samples(model1, n.iter = iters, progress.bar = "none")
dic2 <- dic.samples(model2, n.iter = iters, progress.bar = "none")
dic3 <- dic.samples(model3, n.iter = iters, progress.bar = "none")
dic4 <- dic.samples(model4, n.iter = iters, progress.bar = "none")
