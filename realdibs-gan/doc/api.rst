:orphan:

.. _api_reference:

====================
Python API Reference
====================

This is the place where you can find all about our classes (with
``CamelCase`` names) and our functions (with ``snake_case`` names).

.. contents::
   :local:
   :depth: 2

Factory methods
===============

.. currentmodule:: noice

.. autosummary::
   :toctree: generated/

   make_dataset
   make_model
   make_optimizer
   make_loss


Utility Classes
===============

.. currentmodule:: noice.utils

:py:mod:`noice.utils`:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Bunch
   BunchConst
   BunchConstNamed


Losses
======

:py:mod:`noice.losses`

.. currentmodule:: noice.losses

.. autosummary::
   :toctree: generated/

   norm_crossentropy_mean
   softmax_crossentropy_mean
   binary_crossentropy
   sigmoid_crossentropy
   norm_crossentropy
   softmax_crossentropy
   focalloss
   binary_bitempered_crossentropy
   bitempered_crossentropy


Logging and Configuration
=========================

.. currentmodule:: noice.utils

.. autosummary::
   :toctree: generated/

   get_variable