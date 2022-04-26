library(tidyverse)
library(rjags)

set.seed(31618)  # For reproducibility

# Set the number of iterations for the MCMC sampler
burn    <- 10000
iters   <- 50000
thin    <- 10

# Load the data
Y <- log(df$totalValue)
X <- cbind(1, df$contractLength, df$draftRound, df$draftOverall, df$draftYear,
           df$FG., df$X3P., df$X2P., df$FT., df$AST, df$STL, df$BLK, df$TOV,
           df$PTS, df$TS.)
# Removed age, height, and weight as their coefficients would not converge
n <- length(Y)
p <- ncol(X)
data    <- list(Y=Y,X=X,n=n,p=p)

# Specify the FIRST model
model_string1 <- textConnection("model{

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

model1   <- jags.model(model_string1,data = data,quiet=TRUE, n.chains=2)
update(model1, burn, progress.bar="none")

samples1 <- coda.samples(model1,
                         variable.names=c("beta","taue","taub"),
                         n.iter=iters,
                         #thin=thin,
                         progress.bar="none")

summ1 <- summary(samples1)

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

data    <- list(Y=Y,X=X,n=n,p=p)
model2   <- jags.model(model_string2,data = data,quiet=TRUE, n.chains=2)
update(model2, burn, progress.bar="none")

samples2 <- coda.samples(model2,
                         variable.names=c("beta","taue","taub"),
                         n.iter=iters,
                         #thin=thin,
                         progress.bar="none")

summary2 <- summary(samples2)

# Effective Sample Size should be > 1000
ESS1 <- effectiveSize(samples1)
ESS2 <- effectiveSize(samples2)

# Geweke statistic should be < 2
geweke1 <- geweke.diag(samples1[[1]])
geweke2 <- geweke.diag(samples2[[1]])

# Compare DIC for both models
dic1 <- dic.samples(model1, n.iter = iters, progress.bar = "none")
dic2 <- dic.samples(model2, n.iter = iters, progress.bar = "none")
