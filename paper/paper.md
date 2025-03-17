---
title: 'GBNet: Gradient Boosting packages integrated into PyTorch'
tags:
  - Python
  - PyTorch
  - Gradient Boosting
  - XGBoost
  - LightGBM
authors:
  - name: Michael Horrell
    orcid: 0009-0001-3091-0342
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Independent Researcher, USA
   index: 1
date: 16 March 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

GBNet is a Python software package integrating the powerful Gradient Boosting Machines (GBMs) packages XGBoost and LightGBM with PyTorch, a widely-used deep learning library. Gradient boosting is a popular machine learning technique known for accuracy in predictive modeling. XGBoost and LightGBM are industry standard implementations of GBMs recognized for their speed and strong performance across numerous applications. However, these libraries primarily handle standard machine learning tasks and are more difficult to use when faced with complex or non-standard modeling scenarios. For example, using non-standard loss functions with either XGBoost or LightGBM requires the user to manually compute gradients and hessians, a prohibitively difficult requirement for even moderately complex losses.

PyTorch is popular for its ease of defining and training neural networks. Its computational graph offers automatic differentiation capabilities. GBNet leverages these capabilities acting as the go-between that links gradients/hessian calculations from PyTorch to XGBoost or LightGBM models. This integration allows users to easily construct and train complex hybrid models that combine gradient boosting with neural network architectures. GBNet significantly broadens the scope of problems that can be solved with the world leading gradient boosting software packages.

# Statement of need

While XGBoost and LightGBM are industry-standard solutions for tabular data machine learning problems, they have limited flexibility in defining complex model architectures tailored to specific problem types. Users wishing to define custom loss functions, novel architectures, or other advanced modeling scenarios face substantial difficulty to do so due to complex gradient and hessian calulations that are requirements to use either XGBoost or LightGBM.

As a simple motivating example, consider a forecasting model that has a linear trend and a periodic component. A natural specification of this model might be Forecast $(t) = t \beta +$ PeriodicFn $(t)$, where $\beta$ is a constant defining the trend and $PeriodicFn$ is modeled using a GBM. Despite its relative simplicity, this model cannot be easily fit using XGBoost or LightGBM alone.

GBNet addresses this limitation by providing PyTorch Modules that wrap XGBoost and LightGBM. These Modules are model building blocks like any other PyTorch Module. Valid code defining a PyTorch module accomplishing a version of Forecast $(t)$ is given in just a few lines:

```python
import torch
from gbnet.xgbmodule import XGBModule

class ForecastModule(torch.nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)
        self.xgb = XGBModule(n, d, 1)

    def forward(self, t):
        return self.linear(t) + self.xgb(t)

    def gb_step(self):
        self.xgb.gb_step()
```
The new parts of this code (mainly in comparison to PyTorch code) are `XGBModule`, the wrapper on XGBoost and `gb_step`, a method used to update the underlying XGBoost model.

As seen in this example, once an instance of `XGBModule` has been defined, it can be combined with any other model logic supported by PyTorch. This straightforward example demonstrates GBNet’s ease-of-use in defining complex models.

As of this writing, there do not appear to be any alternatives that directly integrate XGBoost and LightGBM with an easy-to-use auto-differentiation framework like PyTorch. There are related projects that attempt to combine aspects of trees and neural networks, such as Deep Neural Decision Forests (DNDF), DeepGBM, or NODE; however, these typically involve complex neural network structures or stacking methods. GBNet provides a simpler, direct integration. GBNet bridges the simplicity and performance of gradient boosting with the flexibility of neural network modeling.

## Research Applications

Several research areas stand to benefit from GBNet. GBNet itself contains a forecasting application (`gbnet.models.forecasting`) that has improved performance over Meta's Prophet algorithm on a set of benchmarks seen in the notebook linked [here](https://github.com/mthorrell/gbnet/blob/main/examples/simple_forecast_example.ipynb). The package also provides an ordinal regression implementation (`gbnet.models.ordinal_regression`) featuring the ordinal loss which itself is complex, has fittable parameters and is not included in either XGBoost or LightGBM. A notebook [here](https://github.com/mthorrell/gbnet/blob/main/examples/ordinal_regression_comparison.ipynb) demonstrates the ordinal regression application.

More broadly, GBNet may benefit any researcher looking to leverage non-parametric methods while still retaining structural control over their model. In particular, researchers using PyTorch mainly for its ability to produce outputs suited for their application may prefer GBNet at times because XGBoost and LightGBM themselves are incredibly robust. Neural networks can be finicky, requiring many small adjustments and normalizations, while GBMs often just work.

Research into network architectures specifically tailored for GBMs may also hold intrinsic value. Several classic architectures previously explored exclusively with pure neural network methods are now accessible for GBMs through GBNet. Important concepts and methods such as embeddings, autoencoders, variational methods, and contrastive learning may exhibit novel and interesting properties when integrated with GBMs.

# Software Description and Examples

GBNet comprises two primary submodules:

- `gbnet.xgbmodule`, `gbnet.lgbmodule`, `gbnet.gblinear`: Contain PyTorch Module classes (`XGBModule`, `LGBModule` and `GBLinear`) that integrate XGBoost, LightGBM and a linear booster respectively.
- `gbnet.models`: Includes practical implementations of models using either `XGBModule` or `LGBModule`. Currently there are two implementations. `gbnet.models.forecasting` provides a Sci-kit Learn interface for an optimized version of Forecast $(t)$ seen above.  `gbnet.models.ordinal_regression` provides a Sci-kit Learn interface for Ordinal Regression.

## Forecasting Example

`gbnet.models.forecasting.Forecast` is compared to the Meta Prophet algorithm over 500 independent trials as reported in the following table. Each trial consists of selecting a dataset uniformly at random, selecting a training cutoff uniformly at random, selecting a test period cutoff uniformly at random, and finally training a model and testing performance.  The default `gbnet.models.forecasting.Forecast` beat Prophet in 74\% of trials and had a higher than 50% win rate on 8 out of 9 datasets when comparing RMSE values. In addition, `gbnet.models.forecasting.Forecast`, when it did have the losing RMSE, tended to lose by less in comparison to Prophet.

| Dataset                   | N trials | GBNet win Rate (%) | Avg. GBNet Losing RMSE Ratio | Avg. Prophet Losing RMSE Ratio |
|---------------------------|----------|--------------------|------------------------------|--------------------------------|
| Air Passengers            | 50       | **74%**            | **1.42**                         | 1.64                           |
| Pedestrians Covid         | 56       | **66%**            | **1.21**                         | 1.73                           |
| Pedestrians Multivariate  | 54       | **70%**            | **1.34**                         | 1.35                           |
| Retail Sales              | 75       | **81%**            | **1.26**                         | 1.97                           |
| WP Log R                  | 59       | **90%**            | **2.19**                         | 2.60                           |
| WP Log R Outliers1        | 60       | **77%**            | **1.40**                         | 2.56                           |
| WP Log R Outliers2        | 49       | **71%**            | **1.85**                         | 2.47                           |
| WP Log Peyton Manning     | 45       | 44%                | **1.36**                         | 2.22                           |
| Yosemite Temps            | 52       | **85%**            | **2.16**                         | 2.93                           |

Code for these results is [here](https://github.com/mthorrell/gbnet/blob/main/examples/simple_forecast_example.ipynb).

## Ordinal Regression Example

Ordinal regression requires fitting a cumulative logit model with breakpoints `[@mccullagh1980regression]`

<img src="ordinal_probs.png" alt="Fitted Ordinal Probabilities" style="width:50%;"/>

# Acknowledgements

The author gratefully acknowledges insightful feedback from Joe Guinness.

# References




# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

The author gratefully acknowledges insightful feedback from Joe Guinness.

# References
