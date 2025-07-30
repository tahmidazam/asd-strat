The Stratification of Autism Spectrum Disorder Using Multimodal, Unsupervised Cluster Analysis of the SPARK Cohort
=====================================================================================================================

Tahmid Azam [#ta]_, Chirag J. Patel [#cjp]_

Autism Spectrum Disorder (ASD) is a highly heterogenous, neurodevelopmental condition with a diverse clinical presentation involving many genetic factors. However, the *Fifth Edition of the Diagnostic and Statistical Manual of Mental Disorders, Text Revision* (DSM-5-TR) consolidated several conditions into a single ASD diagnosis to improve diagnostic consistency. While previous studies have found evidence for subgroups within individual modalities, such as behavior or genetics, we hypothesize that integrating multiple modalities into cluster analysis will reveal meaningful, robust subgroups hidden within the DSM-5-TR ASD diagnosis. Stratifying ASD in this way could enable precise, individualized support strategies for autistic individuals in educational, home, and clinical environments.

We propose a multi-modal, unsupervised cluster analysis to identify subgroups within the Simons Foundation Powering Autism Research for Knowledge (SPARK) cohort, the largest ASD-focused study that aggregates behavioral questionnaires, genetic samples, and environmental information from over 150,000 autistic individuals across the United States. We integrate questionnaire scores, genetic single nucleotide polymorphisms, and markers of socioeconomic status such as deprivation indices to form a multi-modal representation of each participant, clustering using k-means and Gaussian mixture models.

As an initial step, we assessed predictability between behavioral instruments using supervised linear and non-linear models. Low peak cross-instrument predictability (:math:`R2` = 0.38, 95% CI [0.37, 0.39]) supports mutual exclusivity between measured behavioral traits which motivates our proposed stratification technique.

If our hypothesis is supported, the resulting strata would be strengthened by the statistical power of the SPARK cohort and the multi-modal nature of the analysis. These subgroups could help focus research on the mechanisms behind ASD onset and guide precision care, ensuring autistic individuals are neither overlooked nor overwhelmed, but truly understood.

.. rubric:: Footnotes

.. [#ta] University of Cambridge, United Kingdom
.. [#cjp] Department of Biomedical Informatics, Harvard Medical School, United States

Project contents
----------------

.. toctree::
   :maxdepth: 3

   packages
   lab_notes
