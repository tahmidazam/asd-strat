Initial phenotypic cluster analysis
===================================

We perform k-means clustering on the final scores of the following instruments:

- Repetitive Behavior Scale - Revised (:attr:`~spark.inst.Inst.RBSR`)
- Developmental Coordination Disorder Questionnaire (:attr:`~spark.inst.Inst.DCDQ`)
- Child Behavior Checklist for ages 6 to 18 years (:attr:`~spark.inst.CBCL_6_18`)
- Social Communication Questionnaire (:attr:`~spark.inst.Inst.SCQ`)

These instruments are selected because they have a final score feature.
There are 17251 subjects after joining these features and removing subjects with missing values.

K-means on final scores
-----------------------

K-means is run for :math:`k=2` to :math:`k=15`, repeated 5 times each to calculate means and standard deviations for each metric.

.. figure:: /_static/images/k_means_metrics.png

   Metrics (inertia, silhouette score, Davies-Bouldin index, Calinski-Harabasz index) for k-means clustering on the final scores of the instruments against k.

Interpretation:

#. In the inertia plot, a sharp drop from :math:`k=2` to :math:`k=6` then a gradual taper is observed, showing the classic elbow shape. This suggests that adding clusters beyond 6 yields diminishing returns in reducing within-cluster variance.
#. Tn the silhouette score plot, values are around 0.15–0.20 across :math:`k`, which suggest weak cluster separation and possible overlapping or continuous structure in the data. The little variation show there is no strongly preferred :math:`k`.
#. The lowest Davies-Bouldin index (lower is better) is observed at :math:`k=4`, suggesting this is the optimal :math:`k`.
#. In the Calinski-Harabasz index plot (higher is better), there is a steep decline through all :math:`k`.

The overall impression is that cluster structure is weak. The best candidate is :math:`k=4`, but a lack of distinct peaks in metrics do not strongly support any particular :math:`k`. There may be continuous structure or noise in the data.

Gaussian mixture model (GMM) on final scores
--------------------------------------------

We also run Gaussian mixture model (GMM) clustering for :math:`k=2` to :math:`k=15`, repeated 5 times each to calculate means and standard deviations for each metric.

.. figure:: /_static/images/gmm_metrics.png

   Metrics (log likelihood, AIC, BIC, silhouette score, Davies-Bouldin index, Calinski-Harabasz index) for GMM clustering on the final scores of the instruments against k.


K-means on question features
----------------------------

.. figure:: /_static/images/questions_k_means_metrics.png

   Metrics (inertia, silhouette score, Davies-Bouldin index, Calinski-Harabasz index) for k-means clustering on the question features of the instruments against k.

GMM on question features
------------------------

.. figure:: /_static/images/questions_gmm_metrics.png

   Metrics (log likelihood, AIC, BIC, silhouette score, Davies-Bouldin index, Calinski-Harabasz index) for GMM clustering on the question features of the instruments against k.

Module contents
---------------


.. automodule:: asd_strat.commands.initial_pheno
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: