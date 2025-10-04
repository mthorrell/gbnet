gbnet.models package
====================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   gbnet.models.tests

Submodules
----------

gbnet.models.forecasting module
-------------------------------

.. automodule:: gbnet.models.forecasting
   :members:
   :show-inheritance:
   :undoc-members:

gbnet.models.ordinal\_regression module
---------------------------------------

.. automodule:: gbnet.models.ordinal_regression
   :members:
   :show-inheritance:
   :undoc-members:

gbnet.models.survival package
-----------------------------

.. toctree::
   :maxdepth: 4

gbnet.models.survival.hazard_integrator module
---------------------------------------------

.. automodule:: gbnet.models.survival.hazard_integrator
   :members:
   :show-inheritance:
   :undoc-members:

gbnet.models.survival.hazard_survival module
-------------------------------------------

.. automodule:: gbnet.models.survival.hazard_survival
   :members:
   :show-inheritance:
   :undoc-members:

gbnet.models.survival.discrete\_survival module
-----------------------------------------------

.. automodule:: gbnet.models.survival.discrete_survival
   :members:
   :show-inheritance:
   :undoc-members:

The survival submodule currently exposes three estimators:
- ``HazardSurvivalModel``: Continuous-time survival using gradient-boosted hazards integrated over time.
- ``BetaSurvivalModel``: Discrete-time survival using a Beta distribution to model a mixture of geometric distributions with GBMs.
- ``ThetaSurvivalModel``: Discrete-time survival using a geometric distribution with a single parameter, theta, produced by a GBM.

Module contents
---------------

.. automodule:: gbnet.models
   :members:
   :show-inheritance:
   :undoc-members:
