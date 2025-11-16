################################################################################
### PYQUANT: PYthon package for the simulation of QUANTum systems            ###
###                                                                          ###
### Author of this file: Simon Euchner                                       ###
################################################################################


Description.

    While working on my master's thesis in the year 2025 it became clear to me
that for the quick-and-dirty (basic) simulations I run on a daily basis I always
need the same numerical quantities. Therefore I descided to write a small
software library to avoid writing the same computer code over and over.
    To implement this library I decided to use the scipting language Python,
since it offers an easy to use interface to well-tested and fast FORTRAN, etc.
code. Unfortunately I had problems with ready-to-use packages like QuTiP
[https://qutip.org] for several reasons:

    1. I find it difficult to figure out how exactly routines work, as QuTiP is
       rather large.

    2. QuTiP has build-in help mechanisms which complicate optimization.

    3. I want to be sure that all routines use 128 bit complex numbers.

    4. I want to use sparse matrices at ALL times. I personally beliefe that
       (except for special cases) there is no good reason to use dense matrices.
       That is because even if the dense routine is faster, I can only use it in
       case the matrices are small anyway, i.e., also the sparse routines will
       typically be fast enough.

    I will place all functions in a single file in order to make the package as
see-through as possible. As soon as I need new functionality I implement it.
This no real timeloss in this, as I need to implement each function exactly one
time --- so no reason to use ready-to-use packages. This way I keep full control
over my simulations.


Dependencies.

 > scipy
 > numpy


Documentation.

    All documentation is directly built into the library. Just call Python's
'help' method on the object you need help with.


QuTiP compatibility.

    If you have a quantum object defined by PYQUANT, you can convert it to a
QuTiP one by running the QuTiP contructor Qobj on its data, i.e. do the
following:

    QuTiP_QOBJ = qutip.Qobj(PYQUANT_QOBJ.data)

Now you can use the QuTiP object as if PYQUANT never existed, i.e., there is no
harm in using PYQUANT if youre collaborator uses QuTiP.
