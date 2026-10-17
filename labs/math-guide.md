# Math guide

Run `python src/math_portfolio.py` from the project root. It writes the numerical results and chart to `reports/`.

## 1. Linear algebra

Each row of `X` represents one example. Its first column contains ones for the intercept; the remaining columns contain feature values. Multiplication `X @ beta` creates the model predictions for every row at once. The least-squares solution chooses the coefficient vector `beta` that minimizes the Euclidean length squared of the residual vector `r = y - X @ beta`.

At the least-squares optimum, `X.T @ r` is approximately zero. Geometrically, the residual is perpendicular to every feature column in the design matrix. The matrix rank checks whether the columns contain independent information. The condition number warns when small changes to input data can produce large changes to estimated coefficients.

The program uses `numpy.linalg.lstsq`, rather than manually inverting `X.T @ X`, to calculate the solution robustly.

## 2. Probability

Bayes' rule is `P(A|B) = P(B|A) P(A) / P(B)`. The example assumes a 2% base rate, 90% sensitivity, and 95% specificity. It shows how a low base rate affects the meaning of a positive result. All numbers are hypothetical.

Monte Carlo estimation approximates a probability by repeating a random experiment and calculating the fraction of successes. Its approximate standard error is `sqrt(p_hat * (1-p_hat) / N)`, where `N` is the number of trials.

## 3. Statistics

The OLS standard error estimates the sampling variability of a coefficient under the model assumptions. The confidence interval is an interval-building procedure that, over many repeated samples under those assumptions, would contain the fixed coefficient about 95% of the time. It is not a 95% probability statement about this already-computed interval.

The p value measures how unusual an estimate at least this far from zero would be if the zero-coefficient model and assumptions were true. It does not measure effect size, practical importance, or the probability the null is true. R-squared describes the fraction of outcome variation explained in this sample; it is not proof of predictive quality.

The bootstrap draws new samples of rows with replacement and refits the model. Its interval reflects variation across those resamples, but independent row resampling is inappropriate for clustered or time-ordered data.

## 4. Calculus and optimization

For `J(theta) = ||X theta - y||^2 / (2n)`, differentiation gives `gradient = X.T @ (X @ theta - y) / n`. Gradient descent repeatedly applies `theta = theta - learning_rate * gradient`.

A learning rate that is too small makes progress slow. A learning rate that is too large can make the objective jump around or diverge. Compare the curve in the report after changing `learning_rate` in the script.

## Try these experiments

- Change the seed in `make_dataset` and observe how estimates and intervals vary.
- Increase the noise scale from `1.4` and observe wider uncertainty.
- Make the two feature columns correlated and inspect rank, condition number, and coefficient stability.
- Change the learning rate to `0.01`, `0.15`, and `0.8`; compare speed and stability.
- Remove one useful feature and consider how the remaining coefficients change.
- Increase `bootstrap_count` to see the bootstrap interval become less jagged from simulation noise.

Always describe the generated data as simulated. Do not present the output as a finding about any real community or country.
