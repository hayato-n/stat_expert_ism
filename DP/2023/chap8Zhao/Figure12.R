library(mvtnorm)
library(ggplot2)

set.seed(1)

mu <- c(1.1, 1.8)
sigma <- c(3, 4)
rho <- 0.6
sigmas <- matrix(c(sigma[1]^2, rho*sigma[1]*sigma[2], rho*sigma[1]*sigma[2], sigma[2]^2), nrow = 2)

x <- seq(-15.0, 15.0, 0.1)
y <- seq(-15.0, 15.0, 0.1)
grid <- expand.grid(x=x, y=y)
Z <- dmvnorm(grid, mean=mu, sigma=sigmas)

px_cond_y <- function(y) {
  px_mean <- mu[1] + rho*(sigma[1]/sigma[2])*(y-mu[2])
  px_scale <- sigma[1]*sqrt(1-rho^2)
  return(rnorm(1, mean=px_mean, sd=px_scale))    
}

py_cond_x <- function(x) {
  py_mean <- mu[2] + rho*(sigma[2]/sigma[1])*(x-mu[1])
  py_scale <- sigma[2]*sqrt(1-rho^2)
  return(rnorm(1, mean=py_mean, sd=py_scale))
}

gibbs_sampling <- function(steps=1000, x_init=c(0, 0)) {
  samples <- matrix(0, nrow = steps+1, ncol = 2)
  samples[1,] <- x_init
  x <- x_init[1]
  y <- x_init[2]
  
  for (i in 2:(steps+1)) {
    if (i %% 2 == 0) {
      x <- px_cond_y(y)
    } else {
      y <- py_cond_x(x)
    }
    samples[i, ] <- c(x, y)
  }
  
  return(samples)
}

x_init <- c(-7.0, -7.0)
samples <- gibbs_sampling(steps=1000, x_init=x_init)

df <- data.frame(x = grid$x, y = grid$y, z = Z)
samples_df <- data.frame(x = samples[, 1], y = samples[, 2])
p <- ggplot(df, aes(x = x, y = y)) +
  geom_contour(aes(z = z)) +
  geom_point(data = samples_df, aes(x = x, y = y), alpha = 0.15, color = 'green') +
  geom_path(data = samples_df[1:50, ], aes(x = x, y = y), color = 'red') +
  geom_point(aes(x = x_init[1], y = x_init[2]), color = 'black') +
  labs(title = "Gibbs Sampling", x = "X", y = "Y") +
  theme_minimal()

print(p)

library(mvnormtest)
mshapiro.test(t(samples_df[-1,]))
#平均値
colMeans(samples_df[-1,])
#標準偏差
sqrt(cov(samples_df[-1,]))
#相関係数
cor(samples_df[-1,])