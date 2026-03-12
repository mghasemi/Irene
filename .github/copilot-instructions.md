# Copilot Instructions for the `projects` repository

This repository contains two tightly coupled Python packages:

* `GMM/` – a small toolbox for Gaussian mixture models.  The main entry
  point is `GMM/GMM.py` which implements a subclass of `scipy.stats.rv_continuous`
  and provides sampling, several fitting routines (moment matching, least
  squares, EM, variational Bayes, etc.), and a wrapper around a
  moment–matching discretization (`TMM`) that uses the `Irene` library.
  The script `GMM/exm01.py` is a minimal demo showing how to create a
  `GMM` instance, generate synthetic data, call `TMM` and plot results.

* `Irene/` – a general-purpose toolkit for algebraic global optimisation
  problems based on Lasserre relaxations and semidefinite programming.
  It defines classes such as `SDPRelaxations`, `sdp`, `Mom` and friends.
  The `GMM` package imports from `Irene` when it needs to formulate and
  solve SDP relaxations in `TMM`.

There is no overarching build system; both packages are normal Python
modules with a `setup.py` in `Irene` and plain `.py` files elsewhere.

## Dependencies and setup

The code is intended to run with Python 3.  The most common dependencies
can be installed with pip:

```bash
python -m pip install numpy scipy sympy scikit-learn matplotlib pandas
# optional (required for TMM):
python -m pip install cvxopt  # or ensure system solvers such as csdp/sdpa/dsdp
```

`Irene` also needs one or more SDP solvers on the PATH; the `base.py`
helper will search for `csdp`, `sdpa`, `dsdp5`, etc.  On Linux these are
usually installed via your package manager or compiled from source.

To install the `Irene` package into the current environment (for
`import Irene` to work) run

```bash
cd projects/Irene && sudo python setup.py install
``` 

(or use a virtual environment).

Most development work happens by editing the Python files directly and
running the small example scripts.  There are no tests provided – if you
add one, put it beside the relevant module and run `python -m unittest`
from the repository root.

## Typical developer workflows

* **Exploring / running an experiment:** open `GMM/exm01.py`; modify the
  true mixture parameters or `morder`/`solver` for `TMM`.  Run
  `python GMM/exm01.py` to see PNG plots saved in the working directory.

* **Adding a new fitting method:** implement a new method on the `GMM`
  class (e.g. `def myFit(self, X, Y): ...`) following the pattern of
  `MmntMtch`, `EM`, etc.  The method should return a triple `[w, m, v]`
  and a callable pdf function.  Use `self.initparam` and `self.SetInit`
  if you need to provide starting guesses.

* **Using the Irene SDP API:** look at `GMM.TMM` and the `Irene/`
  modules as examples of constructing a relaxation.  Relaxations are built
  by setting an objective with `Rlx.SetObjective`, adding polynomial
  moment constraints with `Rlx.MomentConstraint`, specifying the solver
  via `Rlx.SetSDPSolver`, then calling `Rlx.InitSDP()` and `Rlx.Minimize()`.
  The solution object exposes `Info['moments']` and `Solution.NumericalRank()`.

* **Debugging numerical issues:** most algorithms print the optimizer
  results (`print(res)`) or debug information.  Use a Python debugger
  (`python -m pdb`) or add print statements in `GMM.py` as needed.

* **Plotting and saving results:** the `Plot` method in `GMM` uses
  `matplotlib` and always saves to `{self.name}_{name}.png`.  It colours
  the true mixture, the fitted mixture and individual components.

## Project-specific conventions

* **Parameter ordering:** mixture parameters are stored as flat lists in
  the order `[weights, means, variances]`.  Functions that accept a
  mixture typically expect three separate lists, e.g. `Plot(w, m, v, name)`.

* **Use of `self.a`/`self.b`:** bounds for the distribution inherited from
  `rv_continuous`.  Many methods generate points or random samples via
  these bounds; when adding new sampling routines, respect these fields.

* **`initparam` array:** used by various fitting routines.  Callers can
  fill it with `GMM.SetInit` or directly assign before invoking a
  fitting method.  Its length is 3 × `n` (components) but may be shorter.

* **Irene solver detection:** `base.which` checks the system PATH.  When
  editing Irene, be mindful that unavailable solvers raise `ImportError`.

* **Data flow in sample generation:** `PointsOnCurve` returns evenly
  spaced `X`, `RandomPoints` returns an empirical density `Z` over bins.
  These are typically passed to fitting methods which expect `X` as
  abscissae and `Y` or `Z` as function values.

## Architecture notes

There is no HTTP service or GUI – it's a library and a handful of scripts
for research experimentation.  The two main components communicate via
standard Python imports.  `GMM` is lightweight and depends only on
scientific Python packages, while `Irene` has a deeper algebraic
structure and optional external dependencies.

Major functions of the codebase are:

* mixture modelling (`GMM.py`) – building and fitting mixture models
* moment technology (`AGD-MoM.py`, `GMM.TMM`) – computing and matching
  moments, sometimes using TensorFlow or SDP relaxations
* global optimisation (`Irene/`) – low–level support for Lasserre
  relaxations, moment matrices, and interfacing with SDP solvers

Most users will work in the `GMM` folder; you only need to dive into
`Irene` when extending the TMM or solving custom polynomial problems.

## Example snippets

```python
from GMM import GMM

# create a 3-component mixture on [0,1]
gmm = GMM(n=3, a=0, b=1, Ws=[.3,.3,.4], Ms=[.2,.5,.8], Vs=[.01,.02,.03])
X = gmm.PointsOnCurve(100)
Y = gmm.pdf(X)
sol, pdf_fun = gmm.TMM(X, Y, morder=4, solver='csdp')
gmm.SetInit(*sol)
gmm.Plot(*sol, name='tmm_example')
```

Any time you ask Copilot to make a change, mention these key files and
patterns so it understands the domain: `GMM/GMM.py`, `GMM/exm01.py`,
`Irene/`, `AGD-MoM.py`.

---

Please review and let me know if any important workflows or details are
missing or unclear.