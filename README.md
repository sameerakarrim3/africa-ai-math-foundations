# Africa AI Math Portfolio

A self-contained, reproducible Python project that demonstrates the mathematics commonly used in introductory machine learning. It uses a clearly labeled simulated example inspired by the kinds of development questions studied across Africa. The simulated values are for learning only; they are not claims about any real country or community.

## What it demonstrates

- **Linear algebra:** vectors, design matrices, matrix multiplication, least-squares estimation, matrix rank, condition number, and residual orthogonality.
- **Probability:** Bayes' rule, a base-rate example, Monte Carlo estimation, and simulation standard error.
- **Statistics:** descriptive estimates, ordinary least squares, standard errors, t-based confidence intervals, a coefficient hypothesis test, R², and bootstrap intervals.
- **Calculus and optimization:** the gradient of mean squared error and gradient descent compared with the least-squares solution.
- **Good analytical practice:** reproducible random seeds, explicit assumptions, uncertainty interpretation, and a clear boundary between simulated data and real-world evidence.

## Run the program

Install Python 3.10 or newer. From this project folder, open a terminal and run:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Then install the packages and run the program:

```bash
pip install -r requirements.txt
python src/math_portfolio.py
```

The program creates a `reports` folder containing a Markdown report and a chart. It does not need internet access.

## The experiment

The program creates a deterministic synthetic dataset with two standardized features and a response generated from known coefficients plus random noise. Because the generating values are known, the fitted model can be compared against a known answer. The dataset is intentionally synthetic so the mathematical demonstration does not imply findings about people, countries, or policy.

The model uses a design matrix `X` with an intercept column and estimates `β` by minimizing the squared residual length:

\[
\hat\beta = \arg\min_\beta \|X\beta-y\|_2^2.
\]

It checks that the residuals are approximately orthogonal to the columns of `X`, then computes classical OLS standard errors and confidence intervals. Gradient descent minimizes `J(θ)=||Xθ-y||²/(2n)` using `∇J(θ)=Xᵀ(Xθ-y)/n`. A row bootstrap provides a second, assumption-sensitive view of coefficient uncertainty.

## Suggested walkthrough

Start with [the math guide](labs/math-guide.md). It explains the equations, how to interpret the output, and small experiments you can try by changing the seed, noise, feature dependence, and learning rate.

When presenting the project, explain the model in your own words, distinguish a confidence interval from a prediction interval, interpret the Bayes example, and describe what can make the statistical assumptions fail. The point is to show your reasoning, not just that the script runs.

## Limitations

The generated data are not collected observations. The screening example is hypothetical and is not medical advice or a real diagnostic tool. Classical OLS intervals assume a correctly specified linear model with independent, constant-variance errors. The bootstrap resamples rows independently and would need a different design for clustered or time-dependent observations. A low p value does not prove an effect is important or causal.
