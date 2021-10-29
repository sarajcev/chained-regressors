## Time-series forecasting with chained regressors

[![GitHub](https://img.shields.io/github/license/sarajcev/Seminar)](./LICENSE)
![Flake8 workflow](https://github.com/sarajcev/Seminar/actions/workflows/python-app.yml/badge.svg) 
[<img src="https://deepnote.com/buttons/launch-in-deepnote-small.svg">](https://github.com/sarajcev/Seminar)

#### EIT postgraduate course seminar in Machine Learning for Power System Analysis.

Multi-step time-series forecasting of PV production using chained regressors from the `scikit-learn` Python library on the Liege microgrid data from Kaggle. 

Demonstrating, among other things, the use of **pipelines** and **regressor chaining** for the multi-step time-series forecasting. Regressor chaining allows implementing multi-output regression for all `scikit-learn` regressors that do not support it natively. Project also includes features engineering for time-series forecasting, data preprocessing, principal component analysis, hyper-parameters optimization with cross-validation, etc.

Project is also concerned with testing of the stability of the newly introduced `HalvingRandomSearchCV` randomized search strategy for the optimization of model hyper-parameters. This new strategy is still experimental at the time of this writing. Comparison of the novel `HalvingRandomSearchCV` with the traditional `RandomizedSearchCV` is carried out, in terms of execution speed, stability and performance.

More information can be found in this [paper](https://www.zenodo.org/record/5254588).
