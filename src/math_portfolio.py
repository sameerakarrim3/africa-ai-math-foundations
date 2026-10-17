"""Self-contained AI mathematics portfolio demonstration."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def make_dataset(seed: int = 17):
    """Return seeded simulated features and outcomes with known coefficients."""
    rng = np.random.default_rng(seed)
    n = 180
    features = rng.normal(size=(n, 2))
    true_intercept = 18.0
    true_slopes = np.array([2.5, -1.2])
    noise = rng.normal(loc=0.0, scale=1.4, size=n)
    outcomes = true_intercept + features @ true_slopes + noise
    return features, outcomes, np.r_[true_intercept, true_slopes]


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    features, y, truth = make_dataset()
    n = len(y)
    X = np.column_stack((np.ones(n), features))
    names = ["intercept", "feature_1", "feature_2"]

    # Least-squares solution and linear algebra diagnostics.
    beta, _, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    orthogonality_error = float(np.linalg.norm(X.T @ residuals))
    condition_number = float(np.linalg.cond(X))
    dof = n - X.shape[1]

    # Classical OLS uncertainty and two-sided tests against coefficient = 0.
    residual_variance = float(residuals @ residuals / dof)
    covariance = residual_variance * np.linalg.inv(X.T @ X)
    standard_errors = np.sqrt(np.diag(covariance))
    critical = float(t.ppf(0.975, dof))
    ci_low = beta - critical * standard_errors
    ci_high = beta + critical * standard_errors
    t_statistics = beta / standard_errors
    p_values = 2 * t.sf(np.abs(t_statistics), dof)
    r_squared = float(1 - residuals @ residuals / np.sum((y - y.mean()) ** 2))

    # Bayes' rule: a hypothetical positive-result example where base rate matters.
    prevalence, sensitivity, specificity = 0.02, 0.90, 0.95
    positive_probability = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    posterior_probability = sensitivity * prevalence / positive_probability

    # Monte Carlo estimate of a fair-coin probability and its simulation standard error.
    rng = np.random.default_rng(2026)
    coin_trials = rng.binomial(1, 0.5, size=100_000)
    estimated_probability = float(coin_trials.mean())
    monte_carlo_se = float(np.sqrt(estimated_probability * (1 - estimated_probability) / len(coin_trials)))

    # Batch gradient descent on J(theta) = ||X theta - y||^2 / (2n).
    def objective(theta: np.ndarray) -> float:
        error = X @ theta - y
        return float(error @ error / (2 * n))

    def gradient(theta: np.ndarray) -> np.ndarray:
        return X.T @ (X @ theta - y) / n

    theta = np.zeros(X.shape[1])
    learning_rate = 0.15
    losses = []
    for _ in range(500):
        losses.append(objective(theta))
        theta -= learning_rate * gradient(theta)
    gradient_descent_gap = float(np.linalg.norm(theta - beta))

    # Percentile bootstrap: resample complete rows and refit the coefficients.
    bootstrap_rng = np.random.default_rng(81)
    bootstrap_count = 600
    bootstrap_coefficients = np.empty((bootstrap_count, X.shape[1]))
    for index in range(bootstrap_count):
        sample = bootstrap_rng.integers(0, n, size=n)
        bootstrap_coefficients[index] = np.linalg.lstsq(X[sample], y[sample], rcond=None)[0]
    bootstrap_low, bootstrap_high = np.quantile(bootstrap_coefficients[:, 1:], [0.025, 0.975], axis=0)

    # Save a convergence curve and compare two ways of expressing slope uncertainty.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(losses, color="#176b87")
    axes[0].set(title="Gradient descent minimizes squared error", xlabel="Update", ylabel="Objective J(θ)")
    slope_positions = np.arange(len(beta) - 1)
    axes[1].errorbar(beta[1:], slope_positions - 0.06,
                     xerr=np.vstack((beta[1:] - ci_low[1:], ci_high[1:] - beta[1:])),
                     fmt="o", capsize=4, label="OLS t interval")
    axes[1].errorbar((bootstrap_low + bootstrap_high) / 2, slope_positions + 0.06,
                     xerr=np.vstack(((bootstrap_high - bootstrap_low) / 2,) * 2),
                     fmt="s", capsize=4, label="Bootstrap interval")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(yticks=slope_positions, yticklabels=names[1:],
                title="Two estimates of slope uncertainty", xlabel="Coefficient estimate")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "math_portfolio_results.png", dpi=160)
    plt.close(fig)

    table = "\n".join(
        f"| {name} | {actual:.3f} | {estimate:.3f} | {error:.3f} | [{low:.3f}, {high:.3f}] | {pvalue:.4g} |"
        for name, actual, estimate, error, low, high, pvalue
        in zip(names, truth, beta, standard_errors, ci_low, ci_high, p_values)
    )
    report = f"""# Math portfolio results

This reproducible experiment uses **simulated data**. It is not evidence about any real place or population. The response was generated from known coefficients with independent Gaussian noise so estimates can be compared with the values used to create the example.

## Linear algebra and regression

The design matrix has shape `{X.shape}`, rank `{rank}`, and condition number `{condition_number:.3f}`. The smallest singular value is `{singular_values[-1]:.4f}`. The residual orthogonality check `||Xᵀr||₂` is `{orthogonality_error:.2e}`, close to zero at floating-point precision. Training R² is `{r_squared:.3f}`.

| Term | Generating value | OLS estimate | Standard error | Approx. 95% confidence interval | p value for zero coefficient |
|---|---:|---:|---:|---:|---:|
{table}

The p value tests a zero-coefficient null under the model assumptions. It is not the probability that the null hypothesis is true.

## Probability

In the hypothetical base-rate example, the assumed prevalence is 2%, sensitivity is 90%, and specificity is 95%. Bayes' rule gives a probability of **{posterior_probability:.1%}** for the condition after a positive result. This illustrates why the base rate matters; it is not a real screening test.

A Monte Carlo experiment estimated the chance of heads as `{estimated_probability:.4f}` across 100,000 simulated tosses. Its estimated Monte Carlo standard error is `{monte_carlo_se:.4f}`. That standard error describes simulation noise under the assumed fair-coin model.

## Calculus and optimization

For `J(θ) = ||Xθ-y||²/(2n)`, the gradient is `∇J(θ) = Xᵀ(Xθ-y)/n`. Gradient descent ran for 500 updates with learning rate `{learning_rate}`. Its coefficient vector is `{gradient_descent_gap:.3g}` in Euclidean distance from the least-squares solution.

![Gradient descent and coefficient uncertainty](math_portfolio_results.png)

## Uncertainty and limitations

The bootstrap interval resamples rows independently. Both it and the classical t interval rely on assumptions; neither fixes biased sampling, omitted variables, or dependence between observations. The generated dataset is a teaching example only.
"""
    (REPORTS / "math_portfolio_results.md").write_text(report, encoding="utf-8")
    print(f"Created {REPORTS / 'math_portfolio_results.md'}")
    print(f"Created {REPORTS / 'math_portfolio_results.png'}")


if __name__ == "__main__":
    main()
