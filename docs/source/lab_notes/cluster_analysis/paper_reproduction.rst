Paper reproduction
==================

We attempt to reproduce the results of the Litman et al. paper titled *Decomposition of phenotypic heterogeneity
in autism reveals underlying genetic programs* [#paper]_. The code and script used in this paper was accessed
from `GitHub <https://github.com/FunctionLab/asd-pheno-classes>`_.

Feature selection
-----------------

The list of features used in the paper was obtained from using a code generation function injected into the preprocessing script [#script]_ at line 127, before dataframes were integrated.

.. code-block:: python

   print("\n".join(["SCQ_FEATURES = ["] + [f"    Feat.SCQ_{col.upper()}," for col in scqdf.columns] + ["]"]))
   print("\n".join(["BHC_FEATURES = ["] + [f"    Feat.BHC_{col.upper()}," for col in bhcdf.columns] + ["]"]))
   print("\n".join(["BHS_FEATURES = ["] + [f"    Feat.BHS_{col.upper()}," for col in bhsdf.columns] + ["]"]))
   print("\n".join(["RBSR FEATURES = ["] + [f"    Feat.RBSR_{col.upper()}," for col in rbsr.columns] + ["]"]))
   print("\n".join(["CBCL_6_18_FEATURES = ["] + [f"    Feat.CBCL_6_18_{col.upper()}," for col in cbcl_2.columns] + ["]"]))

These features were joined using the :attr:`~spark.spark.SPARK.init_and_join` method.

.. note::

   The ASD column was omitted from :attr:`~spark.inst.Inst.BHC`, as it was already included by :attr:`~spark.inst.Inst.SCQ`, mirroring how Duplicated columns are removed in the preprocessing script [#script]_.

Preprocessing
-------------

Inclusion criteria ascertained from the preprocessing script [#script]_ involve:

- An :attr:`~spark.inst.Inst.SCQ` age of evaluation range of 4–18 years.
- An :attr:`~spark.inst.Inst.BHC` age of evaluation range of 4–18 years.

Data transformations include:

- Mapping of male sex assigned at birth to 1, female sex assigned at birth to 0.
- Serialising categorical variables into numerical values.

Exclusion criteria include:

- Columns where more than 10% of the values are missing
- Rows with any missing values.

The preprocessing metrics are shown below, including the male to female split:

+---------------------+------------------+-------------------+
|                     | Reproduction     | Original Paper    |
+=====================+==================+===================+
| Number of subjects  | 27688            | 9094              |
+---------------------+------------------+-------------------+
| Number of features  | 108              | 247               |
+---------------------+------------------+-------------------+
| Male subjects       | 21416            | 6818              |
+---------------------+------------------+-------------------+
| Female subjects     | 6272             | 2276              |
+---------------------+------------------+-------------------+

Observations include a 3-fold increase in subject count, but a loss of 139 features. The male to female proportions have stayed roughly the same. These observed differences are likely due to the use of a more recent release of SPARK (``2025-03-31`` oppose to the original paper's ``2022-12-12``).
Raising the maximum missing value threshold from 10% to 30% only yields 6 more features, which indicates that there are many more subjects who have joined the study, but have not completed all the instruments.

When comparing the subject identifiers between the original and the reproduction, we find that 96.4% of the subjects in the original paper are also present in the reproduction. The individuals who are not present in the reproduction may have requested their data be removed, may have found to be erroneous.

Fit results
-----------

+--------------------------------+--------------------+----------------------+--------------------+
|                                | Original paper     | Original identifiers | All identifiers    |
+================================+====================+======================+====================+
| Sample size                    | 5392               | 8767                 | 27688              |
+--------------------------------+--------------------+----------------------+--------------------+
| Case weights                   | 37%, 34%, 19%, 10% | 31%, 27%, 25%, 16%   | 52%, 27%, 14%, 6%  |
+--------------------------------+--------------------+----------------------+--------------------+
| Number of estimated parameters |                    | 10229                | 9049               |
+--------------------------------+--------------------+----------------------+--------------------+
| Sacled relative entropy        |                    | 0.9780               | 0.9783             |
+--------------------------------+--------------------+----------------------+--------------------+



.. rubric:: Footnotes

.. [#paper] Litman A, Sauerwald N, Green Snyder L, Foss-Feig J, Park CY, Hao Y, et al. Decomposition of phenotypic heterogeneity in autism reveals underlying genetic programs. Nat Genet. 2025 Jul 9;1–9.
.. [#script] `asd-pheno-classes/PreprocessingScripts/process_integrate_phenotype_data.py <https://github.com/FunctionLab/asd-pheno-classes/blob/main/PreprocessingScripts/process_integrate_phenotype_data.py>`_

Module contents
---------------

.. automodule:: asd_strat.commands.paper_reproduction
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: