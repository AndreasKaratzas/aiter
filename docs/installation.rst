Install a matching build
==========================

The distribution is named ``amd-aiter``; the Python import is ``aiter``. Select the ROCm, Python, Torch and DSL environment required by your application and the wheel's support record. Package names and GPU family names alone do not establish binary compatibility.

This site documents the ``akaratza_aiter_implementation`` branch in ``AndreasKaratzas/aiter``. Select that branch to use the prepared interfaces, package layout and CI commands shown here.

Develop from source
---------------------

.. code-block:: bash

   git clone --recursive --branch akaratza_aiter_implementation https://github.com/AndreasKaratzas/aiter.git
   cd aiter
   python -m pip install -e .
   python -m aiter doctor --gpu

For an existing clone, run ``git submodule update --init --recursive`` to obtain its pinned native dependencies. Native compilation needs the matching ROCm compiler. Package metadata and ``python -m aiter doctor`` without ``--gpu`` do not initialize Torch or a device.

Install a qualified wheel
---------------------------

Download the wheel for the exact approved environment and verify its recorded SHA-256 digest. Install into that environment without replacing its Torch and other protected dependencies:

.. code-block:: bash

   python -m pip install --no-deps /path/to/approved/amd_aiter-VERSION.whl
   python -m aiter doctor --gpu

``VERSION`` represents the complete downloaded wheel filename, including its Python and platform tags. It is not a published version or a literal install command. Do not infer a package index or container tag from this example; the qualified release record provides the artifact locations.

Build a wheel
---------------

.. code-block:: bash

   python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/aiter-wheels

The :doc:`build guide </extend/build>` explains explicit build dependencies, native SDK bundling, prebuild selections, target architectures and source distributions. The build stages its output outside the runtime source package. Metadata queries do not install GPU dependencies or rewrite source version markers.

Compilation policy
--------------------

A bundled prepared HIP/CK provider can run with ``ExecutionPolicy(allow_compile=False)``. Triton/Gluon preparation needs its compiler environment. Historical wrappers may still JIT a variant that is absent from an AOT wheel; that behavior must be declared in the consumer profile. See the :doc:`container guide </deliver/containers>` for wheel copying and base-image inheritance.

Optional Iris communication dependencies are declared in ``requirements/runtime/iris.txt``. They are needed only for those communication implementations.
