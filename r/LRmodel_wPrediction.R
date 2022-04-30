library(tidyverse)
library(rjags)

set.seed(31618)  # For reproducibility

# Download the data
df        <- read.csv('./data/full_nonrookie.csv')
df_rookie <- read.csv('./data/full_rookie.csv')

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

# Fit the model to a training set of size 1000 and make prediction for the 
# remaining observations
test  <- order(runif(length(Y)))>1000

Yo    <- Y[!test]    # Observed data
Xo    <- X[!test,]
ido   <- id[!test]

Yp    <- Y[test]     # Data set aside for prediction
Xp    <- X[test,]
idp   <- id[test]

no    <- length(Yo)  # number of observations in observed data
np    <- length(Yp)  # number of observations in data set aside for prediction
p     <- ncol(Xo)    # number of covariates (including the intercept)

#n <- length(Y)       # number of observations
#p <- ncol(X)        # number of covariates (including the intercept)
N <- length(team)    # number of teams

##############################################################################
#
# Model 1: Constant slopes using Gaussian priors
#
##############################################################################

# Specify the FIRST model
model_string1 <- textConnection("model{

  # Likelihood
  for(i in 1:no){
    Yo[i] ~ dnorm(inprod(Xo[i,], beta[]), taue)
  }

  # Priors
  beta[1] ~ dnorm(0, 0.001)
  for(j in 2:p){
    beta[j] ~ dnorm(0, taub*taue)
  }
  taue ~ dgamma(0.1, 0.1)
  taub ~ dgamma(0.1, 0.1)
  
  # Prediction
  for(i in 1:np){
    Yp[i] ~ dnorm(inprod(Xp[i,], beta[]), taue)
  }
  
  # Posterior preditive checks
  for(i in 1:no){
    Y2[i] ~ dnorm(inprod(Xo[i,], beta[]), taue)
  }
  z[1] <- mean(Y2[])
  z[2] <- min(Y2[])
  z[3] <- max(Y2[])
  z[4] <- max(Y2[])-min(Y2[])
  z[5] <- sd(Y2[])
  
}")

data <- list(Yo=Yo, Xo=Xo, Xp=Xp, no=no, np=np, p=p)  # Yp is not sent to JAGS!
model1 <- jags.model(model_string1, data = data, quiet = TRUE, n.chains = 2)
update(model1, burn, progress.bar="none")

samples1 <- coda.samples(model1,
                         variable.names = c("Yp", "beta", "taub", "taue", "z"),
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
  for(i in 1:no){
    Yo[i] ~ dnorm(inprod(Xo[i,], beta[]), taue)
  }

  # Priors
  beta[1] ~ dnorm(0, 0.001)
  for(j in 2:p){
    beta[j] ~ ddexp(0, taub*taue)
  }
  taue ~ dgamma(0.1, 0.1)
  taub ~ dgamma(0.1, 0.1)
  
  # Prediction
  for(i in 1:np){
    Yp[i] ~ dnorm(inprod(Xp[i,], beta[]), taue)
  }
  
  # Posterior preditive checks
  for(i in 1:no){
    Y2[i] ~ dnorm(inprod(Xo[i,], beta[]), taue)
  }
  z[1] <- mean(Y2[])
  z[2] <- min(Y2[])
  z[3] <- max(Y2[])
  z[4] <- max(Y2[])-min(Y2[])
  z[5] <- sd(Y2[])
  
}")

data <- list(Yo=Yo, Xo=Xo, Xp=Xp, no=no, np=np, p=p)  # Yp is not sent to JAGS!
model2   <- jags.model(model_string2, data = data, quiet=TRUE, n.chains=2)
update(model2, burn, progress.bar="none")

samples2 <- coda.samples(model2,
                         variable.names=c("Yp", "beta", "taub", "taue", "z"),
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
  for(i in 1:no){
    Yo[i] ~ dnorm(inprod(Xo[i,], beta[ido[i],]), taue)
   }

  # Slopes
  for(j in 1:p){
    for(i in 1:N){
      beta[i,j] ~ dnorm(0,0.01)
    }
  }

  # Priors
  taue ~ dgamma(0.1,0.1)

  # Prediction
  for(i in 1:np){
    Yp[i] ~ dnorm(inprod(Xp[i,], beta[idp[i],]), taue)
  }
  
  # Posterior preditive checks
  for(i in 1:no){
    Y2[i] ~ dnorm(inprod(Xo[i,], beta[ido[i],]), taue)
  }
  z[1] <- mean(Y2[])
  z[2] <- min(Y2[])
  z[3] <- max(Y2[])
  z[4] <- max(Y2[])-min(Y2[])
  z[5] <- sd(Y2[])
}")

data <- list(Yo=Yo, Xo=Xo, Xp=Xp, no=no, np=np, ido=ido, idp=idp, N=N,
             p=p)  # Yp is not sent to JAGS!
model3   <- jags.model(model_string3, data = data, quiet=TRUE, n.chains=2)
update(model3, burn, progress.bar="none")

samples3 <- coda.samples(model3,
                         variable.names=c("Yp", "beta", "taue", "z"),
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
  for(i in 1:no){
    Yo[i] ~ dnorm(inprod(Xo[i,], beta[ido[i],]), taue)
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

  # Prediction
  for(i in 1:np){
    Yp[i] ~ dnorm(inprod(Xp[i,], beta[idp[i],]), taue)
  }
  
  # Posterior preditive checks
  for(i in 1:no){
    Y2[i] ~ dnorm(inprod(Xo[i,], beta[ido[i],]), taue)
  }
  z[1] <- mean(Y2[])
  z[2] <- min(Y2[])
  z[3] <- max(Y2[])
  z[4] <- max(Y2[])-min(Y2[])
  z[5] <- sd(Y2[])
}")

data <- list(Yo=Yo, Xo=Xo, Xp=Xp, no=no, np=np, ido=ido, idp=idp, N=N,
             p=p)  # Yp is not sent to JAGS!
model4   <- jags.model(model_string4,data = data,quiet=TRUE, n.chains=2)
update(model4, burn, progress.bar="none")

samples4 <- coda.samples(model4,
                         variable.names=c("Yp", "beta", "taue", "z"),
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

##############################################################################
#
# Plot the PPD of each model for 5 observations in the test data set
#
##############################################################################

#Extract the samples for each parameter
samps1 <- samples1[[1]]
Yp.samps1 <- samps1[,1:np]
samps2 <- samples2[[1]]
Yp.samps2 <- samps2[,1:np]
samps3 <- samples3[[1]]
Yp.samps3 <- samps3[,1:np]
samps4 <- samples4[[1]]
Yp.samps4 <- samps4[,1:np]

# PPD 95% intervals
#low1   <- apply(Yp.samps1, 2, quantile, 0.025)
#high1  <- apply(Yp.samps1, 2, quantile, 0.975)
#cover1 <- mean(Yp>low1 & Yp<high1)
#mean(cover1)

# Plot the PPD
for(j in 1:5){
  
  # PPD
  plot(density(Yp.samps1[,j]), col="darkred", xlab="log(AAV)", main="PPD",
       lwd=2)
  lines(density(Yp.samps2[,j]), col="burlywood4", lwd=2)
  lines(density(Yp.samps3[,j]), col="darkorange1", lwd=2)
  lines(density(Yp.samps4[,j]), col="darkgoldenrod1", lwd=2)
  
  # Truth
  abline(v=Yp[j], col="chartreuse4", lwd=2)
  
  legend("topright",c("Model 1", "Model 2", "Model 3", "Model 4", "Truth"),
         col=c("darkred", "burlywood4", "darkorange1", "darkgoldenrod1", "chartreuse4"),
         lty=1, lwd=2, inset=0.05)
}

##############################################################################
#
# Compute the Bayesian p-values
#
##############################################################################

# Compute the test stats for the data
samps0   <- c(mean(Y), min(Y), max(Y), max(Y)-min(Y), sd(Y))
stat.names <- c("Mean Y", "Min Y", "Max Y", "Range Y", "Standard Deviation Y")

# Compute the test stats for the models
pval1 <- rep(0, 5)
names(pval1) <- stat.names
pval4 <- pval3 <- pval2 <- pval1

for(j in 1:5){
  plot(density(samps1[,ncol(samps1)-5+j]),
       xlim = range(c(samps0[j], 
                      samps1[,ncol(samps1)-5+j],
                      samps2[,ncol(samps2)-5+j],
                      samps3[,ncol(samps3)-5+j],
                      samps4[,ncol(samps4)-5+j])),
       xlab = "D",
       ylab = "Posterior probability",
       main = stat.names[j])
  lines(density(samps2[,ncol(samps2)-5+j]), col=2)
  lines(density(samps3[,ncol(samps3)-5+j]), col=3)
  lines(density(samps4[,ncol(samps4)-5+j]), col=4)
  abline(v=samps0[j], col=5)
  legend("topleft", c("Model 1","Model 2","Model 3", "Model 4"),
         lty=1, col=1:4, bty="n")
  
  pval1[j] <- mean(samps1[,ncol(samps1)-5+j] > samps0[j])
  pval2[j] <- mean(samps2[,ncol(samps1)-5+j] > samps0[j])
  pval3[j] <- mean(samps3[,ncol(samps1)-5+j] > samps0[j])
  pval4[j] <- mean(samps4[,ncol(samps1)-5+j] > samps0[j])
}

