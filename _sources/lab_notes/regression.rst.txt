Regression
==========

A command to run supervised regression analysis on the SPARK dataset.

Introduction
------------

Many instruments in SPARK are behavioural questionnaires that have a set of question features and a single score feature.
We investigate whether the score feature of one instrument can be predicted from the question features of another instrument.
Strong predictability would suggest common latent traits, while poor predictability would indicate instrument-specific variance and provide evidence for ASD heterogeneity.

Methods
-------

The instruments under analysis include:

- Repetitive Behavior Scale - Revised (:attr:`~spark.inst.Inst.RBSR`)
- Developmental Coordination Disorder Questionnaire (:attr:`~spark.inst.Inst.DCDQ`)
- Child Behavior Checklist for ages 6 to 18 years (:attr:`~spark.inst.CBCL_6_18`)
- Social Communication Questionnaire (:attr:`~spark.inst.Inst.SCQ`)

The inclusion criteria for these instruments include:

- Each instrument must have a 1 score feature used as the target for regression.
- Each instrument must have a n question features used as the input for regression.

The regression implementations used include:

- `Linear regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_
- `Lasso regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_
- `Ridge regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_
- `Histogram-based gradient boosting regression tree <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor>`_
- `Random forest regression tree <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor>`_

The sample size for each regression is determined by the number of subjects that have both the question features and the score feature available.
The number of subjects used in each regression is shown below:

.. figure:: /_static/images/subject_counts.png

   A heatmap showing the number of subjects used in each regression.
   The x-axis corresponds to the instrument providing question features and the y-axis corresponds to the instrument whose score feature is being predicted.

K-fold `cross-validation <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_ with :math:`k = 5` is used to evaluate :math:`R^2` values.
95% confidence intervals are calculated using the `Student's t-distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html>`_ with degrees of freedom equal to :math:`k - 1 = 4`.

Results
-------

The results are plotted below:

.. figure:: /_static/images/regression_results.png

   A grid of heatmaps showing regression performance. Each heatmap displays the mean :math:`R^2` values for a specific model, where the x-axis corresponds to the instrument providing question features and the y-axis corresponds to the instrument whose score feature is being predicted.

The largest mean :math:`R^2` values are observed when:

- predicting the SCQ final score using the questions from RBS-R (:math:`R^2=0.39`, 95\% CI [0.38, 0.40] using histogram-based gradient boosting regression tree); and
- predicting the CBCL/6-18 final score using the questions from RBS-R (:math:`R^2=0.34`, 95\% CI [0.33, 0.35] using linear regression).

Discussion
----------

The regression results indicate that instrument questions struggle to predict scores of other instruments. Non-linear regression models do not perform better than linear regression models. The low mean :math:`R^2` values supports the view that ASD is a heterogeneous condition, which motivates our unsupervised analysis.


Module contents
---------------


.. automodule:: asd_strat.commands.regression
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

