import statsmodels.stats.proportion as smp

n = 60000  # Sample size
p = 0.7827  # Sample proportion
confidence_level = 0.95  # Confidence level

# Calculate the confidence interval
lower, upper = smp.proportion_confint(
    count=p * n,  # Number of successes
    nobs=n,  # Number of observations
    alpha=1 - confidence_level,  # Significance level
    method="normal",  # Method for interval calculation
)

# Calculate the width of the confidence interval
width = (upper - lower) / 2

print(f"{confidence_level*100}% confidence interval width: {width*100:.2f}%")
