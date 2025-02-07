======================
Constraints
======================

Constraints are a fundamental component in non-trivial fitting operations. They can also be used to affirm the minimum/maximum of a parameter or tie parameters together in a model.

Anatomy of a constraint
-----------------------

A constraint is a rule which is applied to a **dependent** variable. This rule can consist of a logical operation, relation to one or more **independent** variables or an arbitrary function.


Constraints on Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`easyscience.Objects.Base.Parameter` has the properties `builtin_constraints` and `user_constraints`. These are dictionaries which correspond to constraints which are intrinsic and extrinsic to the Parameter. This means that on the value change of the Parameter firstly the `builtin_constraints` are evaluated, followed by the `user_constraints`.


Constraints on Fitting
^^^^^^^^^^^^^^^^^^^^^^

:class:`easyscience.fitting.Fitter` has the ability to evaluate user supplied constraints which effect the value of both fixed and non-fixed parameters. A good example of one such use case would be the ratio between two parameters, where you would create a :class:`easyscience.fitting.Constraints.ObjConstraint`.

Using constraints
-----------------

A constraint can be used in one of three ways; Assignment to a parameter, assignment to fitting or on demand. The first two are covered and on demand is shown below.

.. code-block:: python

     from easyscience.fitting.Constraints import NumericConstraint
     from easyscience.Objects.Base import Parameter
     # Create an `a < 1` constraint
     a = Parameter('a', 0.5)
     constraint = NumericConstraint(a, '<=', 1)
     # Evaluate the constraint on demand
     a.value = 5.0
     constraint()
     # A will now equal 1

Constraint Reference
--------------------

.. minigallery:: easyscience.fitting.Constraints.NumericConstraint
    :add-heading: Examples using `Constraints`

Built-in constraints
^^^^^^^^^^^^^^^^^^^^

These are the built in constraints which you can use

.. autoclass:: easyscience.fitting.Constraints.SelfConstraint
   :members: +enabled

.. autoclass:: easyscience.fitting.Constraints.NumericConstraint
  :members: +enabled

.. autoclass:: easyscience.fitting.Constraints.ObjConstraint
  :members: +enabled

.. autoclass:: easyscience.fitting.Constraints.FunctionalConstraint
  :members: +enabled

.. autoclass:: easyscience.fitting.Constraints.MultiObjConstraint
  :members: +enabled

User created constraints
^^^^^^^^^^^^^^^^^^^^^^^^

You can also make your own constraints by subclassing the :class:`easyscience.fitting.Constraints.ConstraintBase` class. For this at a minimum the abstract methods ``_parse_operator`` and ``__repr__`` need to be written.

.. autoclass:: easyscience.fitting.Constraints.ConstraintBase
  :members:
  :private-members:
  :special-members: __repr__