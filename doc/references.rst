=================================
References
=================================

This page is the consolidated bibliography for the Irene documentation. It lists
the works the method chapters rest on, grouped by the construction each one
supports. Page-level reading lists in individual chapters remain for local
context; when the two disagree, this page is authoritative.

Bibliographic data here was checked against Crossref or the arXiv API
(2026-10); volume, issue, page range and author names are the published ones,
not preprint-stage metadata.

----------------------------------------
The truncated moment problem on algebras
----------------------------------------

The results the generalization from polynomials to arbitrary unital commutative
algebras rests on (see :doc:`optim` and :doc:`theory`).

.. [CGIK] R. E. Curto, M. Ghasemi, M. Infusino, and S. Kuhlmann,
   *The truncated moment problem for unital commutative* :math:`\mathbb{R}`-*algebras*,
   Journal of Operator Theory **90** (1) (2023), 223--261.
   DOI: `10.7900/jot.2021nov26.2392 <https://doi.org/10.7900/jot.2021nov26.2392>`__.

.. [JNie] J. Nie, *The A-truncated K-moment problem*, Foundations of
   Computational Mathematics **14** (6) (2014), 1243--1276.

- J. Stochel, *Solving the truncated moment problem solves the full moment
  problem*, Glasgow Mathematical Journal **43** (3) (2001), 335--341.
  DOI: `10.1017/S0017089501030130 <https://doi.org/10.1017/S0017089501030130>`__.

------------------------------------------------
Moment/SOS hierarchy and polynomial optimization
------------------------------------------------

.. [JBL] J.-B. Lasserre, *Global optimization with polynomials and the problem
   of moments*, SIAM Journal on Optimization **11** (3) (2001), 796--817.
   DOI: `10.1137/S1052623400366802 <https://doi.org/10.1137/S1052623400366802>`__.

.. [HL] D. Henrion and J.-B. Lasserre, *Detecting global optimality and
   extracting solutions in GloptiPoly*, in Positive Polynomials in Control,
   Lecture Notes in Control and Information Sciences **312**, Springer (2005),
   293--310.

.. [JH16] C. Josz and D. Henrion, *Strong duality in Lasserre's hierarchy for
   polynomial optimization*, Optimization Letters **10** (1) (2016), 3--10.

- P. A. Parrilo, *Structured semidefinite programs and semialgebraic geometry
  methods in robustness and optimization*, PhD thesis, California Institute of
  Technology, 2000.
- M. Laurent, *Sums of squares, moment matrices and optimization over
  polynomials*, in Emerging Applications of Algebraic Geometry, Springer
  (2009), 157--270.
- M. Marshall, *Positive Polynomials and Sums of Squares*, Mathematical Surveys
  and Monographs **146**, American Mathematical Society, 2008.
- J.-B. Lasserre, *An Introduction to Polynomial and Semi-Algebraic
  Optimization*, Cambridge Texts in Applied Mathematics **52**, Cambridge
  University Press, 2015.

----------------------------------------
Geometric-programming bounds
----------------------------------------

The GP path in :doc:`geometric` descends from the lower-bound construction of
Ghasemi and Marshall, extended with Lasserre's moment relaxation and then to
constrained geometries.

- M. Ghasemi and M. Marshall, *Lower bounds for polynomials using geometric
  programming*, SIAM Journal on Optimization **22** (2) (2012), 460--473.
  DOI: `10.1137/110836869 <https://doi.org/10.1137/110836869>`__.
- M. Ghasemi, J.-B. Lasserre, and M. Marshall, *Lower bounds on the global
  minimum of a polynomial*, Computational Optimization and Applications
  **57** (2) (2013), 387--402.
  DOI: `10.1007/s10589-013-9596-x <https://doi.org/10.1007/s10589-013-9596-x>`__.
- M. Ghasemi and M. Marshall, *Lower bound for a polynomial on a product of
  hyperellipsoids using geometric programming*, preprint
  `arXiv:2510.03105 <https://arxiv.org/abs/2510.03105>`__ (2025).
- R. J. Duffin, E. L. Peterson, and C. Zener, *Geometric Programming: Theory and
  Application*, Wiley, 1967.
- S. Iliman and T. de Wolff, *Lower bounds for polynomials with simplex Newton
  polytopes based on geometric programming*, SIAM Journal on Optimization
  **26** (2) (2016), 1128--1146.
  DOI: `10.1137/140962425 <https://doi.org/10.1137/140962425>`__.
- S. Iliman and T. de Wolff, *Amoebas, nonnegative polynomials and sums of
  squares supported on circuits*, Research in the Mathematical Sciences **3**
  (2016), article 9.
  DOI: `10.1186/s40687-016-0052-2 <https://doi.org/10.1186/s40687-016-0052-2>`__.

----------------------------------------
SONC relaxations
----------------------------------------

For the circuit-based family in :doc:`sonc`: the characterization of nonnegative
circuit polynomials, the Positivstellensatz and its constrained extension that
turn the problem into a geometric program, and the duality theory behind
optimal SONC bounds.

- M. Dressler, S. Iliman, and T. de Wolff, *A Positivstellensatz for sums of
  nonnegative circuit polynomials*, SIAM Journal on Applied Algebra and Geometry
  **1** (1) (2017), 536--555.
  DOI: `10.1137/16M1086303 <https://doi.org/10.1137/16M1086303>`__.
- M. Dressler, S. Iliman, and T. de Wolff, *An approach to constrained
  polynomial optimization via nonnegative circuit polynomials and geometric
  programming*, Journal of Symbolic Computation **91** (2019), 149--172.
  DOI: `10.1016/j.jsc.2018.06.018 <https://doi.org/10.1016/j.jsc.2018.06.018>`__.
- D. Papp, *Duality of sum of nonnegative circuit polynomials and optimal SONC
  bounds*, Journal of Symbolic Computation **114** (2023), 246--266.
  DOI: `10.1016/j.jsc.2022.04.015 <https://doi.org/10.1016/j.jsc.2022.04.015>`__.
- L. Katthän, H. Naumann, and T. Theobald, *A unified framework of SAGE and
  SONC polynomials and its duality theory*, Mathematics of Computation
  **90** (329) (2021), 1297--1322.
  DOI: `10.1090/mcom/3607 <https://doi.org/10.1090/mcom/3607>`__.
- J. Heuer and T. de Wolff, *The duality of SONC: advances in circuit-based
  certificates*, Journal of Symbolic Computation **132** (2026), 102479.
  DOI: `10.1016/j.jsc.2025.102479 <https://doi.org/10.1016/j.jsc.2025.102479>`__.
- M. Dressler, D. Heuer, D. Naumann, and T. de Wolff, *Global optimization via
  the dual SONC cone and linear programming*, preprint
  `arXiv:2002.09368 <https://arxiv.org/abs/2002.09368>`__ (2020).

----------------------------------------
SOS+SONC hybrid certificates
----------------------------------------

For the two-step framework and the cone geometry in :doc:`sosonc`:

- M. Schick, *Sums of squares plus sums of nonnegative circuit polynomials*,
  PhD dissertation, Universität Konstanz, 2025. Toolbox:
  `SOS_plus_SONC_toolbox <https://github.com/schick-moritz/SOS_plus_SONC_toolbox>`__.
- M. Dressler, S. Kuhlmann, and M. Schick, *Geometrical study of the cone of
  sums of squares plus sums of nonnegative circuits*, preprint
  `arXiv:2305.14848 <https://arxiv.org/abs/2305.14848>`__ (2023).
- M. Ghasemi and S. Kuhlmann, *The cone generated by positive semidefinite mean
  polynomials*, preprint `arXiv:2608.22739 <https://arxiv.org/abs/2608.22739>`__
  (2026).

----------------------------------------
Mean forms and differential SDP
----------------------------------------

For the mean-polynomial certificate family and the differential hierarchy in
:doc:`dsdp_mean`:

- M. Ghasemi, *Mean polynomials: generalizing SOS and SONC via power mean
  forms*, companion manuscript, 2026 (Chapters 1--7).
- J. F. Ritt, *Differential Algebra*, American Mathematical Society Colloquium
  Publications **33**, 1950.
- E. R. Kolchin, *Differential Algebra and Algebraic Groups*, Academic Press,
  1973.

----------------------------------------
Structure exploitation and numerics
----------------------------------------

For the reductions in :doc:`sparsity`, :doc:`newton_polytope`,
:doc:`border_basis` and :doc:`sparse_moment`:

- J.-B. Lasserre, *Convergent SDP-relaxations in polynomial optimization with
  sparsity*, SIAM Journal on Optimization **17** (3) (2006), 822--843.
  DOI: `10.1137/05064504x <https://doi.org/10.1137/05064504x>`__.
- J. Wang, V. Magron, and J.-B. Lasserre, *TSSOS: a moment-SOS hierarchy that
  exploits term sparsity*, SIAM Journal on Optimization **31** (1) (2021),
  30--58.
  DOI: `10.1137/19M1307871 <https://doi.org/10.1137/19M1307871>`__.
- T. Baumbach and M. Bender, *The moment method for computing real points of
  sparse polynomial systems*, preprint
  `arXiv:2609.33313 <https://arxiv.org/abs/2609.33313>`__ (2026) -- the source of
  the toric construction in :doc:`sparse_moment`.

----------------------------------------
Archival versions
----------------------------------------

Released versions of Irene are archived on Zenodo. Cite the software itself
through ``CITATION.cff`` in the repository root, which carries the DOI of the
current release.
