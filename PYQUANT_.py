################################################################################
### Source code of PYQUANT                                                   ###
###                                                                          ###
### Author of this file: Simon Euchner                                       ###
################################################################################

# ---------------------------------------------------------------------------- #
### Packages
import numpy
import scipy
# ---------------------------------------------------------------------------- #

### Constants
EPSILON = 1e-14 # Assumed machine precision for 64-bit floating-point numbers
MAXSIZE = 100 # Print matrices with less or this many entries as dense

class QOBJ():
    """
    QOBJ: Quantum mechanical OBJects

    Class of which all operators and states are instances.

    Attributes
    ----------
    data: scipy.sparse.csr_matrix
        Sparse matrix in CSR format defining entries of state or operator.
    shape: tuple
        Tuple (m, n), where m is the number of rows and n is the number of
        columns.
    """

    def __init__(self: "QOBJ",
                 data: scipy.sparse.csr_matrix,
                 shape: tuple) -> None:
        """
        Initialise a quantum object.

        Parameters
        ----------
        self: QOBJ
            Quantum object.
        data: scipy.sparse.csr_matrix
            Sparse matrix in CSR format defining entries of state or operator.
        shape: tuple
            Tuple (m, n), where m is the number of rows and n is the number of
            columns.

        Returns
        ----------
        """
        self.__check_validity(data, shape)
        self.shape = shape
        self.data = data

    def copy(self: "QOBJ"):
        """
        Copy a quantum object.

        Parameters
        ----------
        self: QOBJ
            Quantum object.

        Returns
        ----------
        _: QOBJ
            Copy of input.
        """
        return __class__.FROM_CSR(self.data)

    def __str__(self: "QOBJ") -> str:
        """
        Text representation of a quantum object.

        Parameters
        ----------
        self: QOBJ
            Quantum object.

        Returns
        ----------
        txt: str
            String representing input.
        """

        # Format string
        txt = ( "\n"
                "INSTANCE OF CLASS QOBJ\n"
                "SHAPE: {:d} x {:d}\n"
                "DATA:\n"
                "{}\n" )

        # Prepare information
        if self.shape[0]*self.shape[1] <= MAXSIZE:
            data = self.data.toarray()
        else:
            data = self.data

        txt = txt.format(self.shape[0], self.shape[1], data)

        return txt

    @staticmethod
    def __check_validity(data: any,
                         shape: any) -> None:
        """
        Check input validity.

        Parameters
        ----------
        data: any
            Candidate for attribute 'data'.
        shape: any
            Candidate for attribute 'shape'.

        Returns
        ----------
        """

        # Check if data is a CSR matrix
        if not isinstance(data, scipy.sparse.csr_matrix):
            message = "PYQUANT: FORMAT IS NOT 'scipy.sparse.csr_matrix'"
            raise ValueError(message)

        # Check if shape is two-dimensional
        if len(shape) != 2:
            message = "PYQUANT: ONLY TWO-DIMENSIONAL ARRAYS ARE ALLOWED"
            raise TypeError(message)

    @classmethod
    def from_csr(cls: "class",
                 data: scipy.sparse.csr_matrix) -> "QOBJ":
        """
        Construct QOBJ from a sparse matrix in CSR format.

        Parameters
        ----------
        cls: class
            Class.
        data: scipy.sparse.csr_matrix
            Sparse matrix to convert into a quantum object.

        Returns
        ----------
        _: QOBJ
            Resulting quantum object.
        """
        cls.__check_validity(data, data.shape)
        return cls(data.astype(numpy.complex128,
                               casting="safe",
                               copy=False),
                   data.shape)

    @classmethod
    def from_array(cls: "class",
                   data: numpy.array) -> "QOBJ":
        """
        Construct QOBJ from a numpy array.

        Parameters
        ----------
        cls: class
            Class.
        data: numpy.array
            Numpy array to convert into a quantum object.

        Returns
        ----------
        _: QOBJ
            Resulting quantum object.
        """
        dt = scipy.sparse.coo_matrix(data).tocsr()
        return cls.from_csr(dt)

    def __add__(self: "QOBJ",
                O: any) -> "QOBJ":
        """
        Addition for quantum objects.

        Parameters
        ----------
        self: QOBJ
            First addend.
        O: any
            Second addend.

        Returns
        ----------
        _: QOBJ
            Sum.
        """
        if isinstance(O, __class__):
            return __class__.from_csr(self.data + O.data)
        else:
            return NotImplemented

    def __sub__(self: "QOBJ",
                O: any) -> "QOBJ":
        """
        Addition for quantum objects.

        Parameters
        ----------
        self: QOBJ
            Minuend.
        O: any
            Subtrahend.

        Returns
        ----------
        _: QOBJ
            Difference.
        """
        if isinstance(O, __class__):
            return __class__.from_csr(self.data - O.data)
        else:
            return NotImplemented

    def __mul__(self: "QOBJ",
                X: any) -> "QOBJ":
        """
        Multiplication and scalar multiplication from the right.

        Parameters
        ----------
        self: QOBJ
            First factor.
        X: any
            Second factor.

        Returns
        ----------
        _: QOBJ
            Product.
        """
        if isinstance(X, __class__):
            return __class__.from_csr(self.data @ X.data)
        else:
            try:
                return __class__.from_csr(self.data * numpy.complex128(X))
            except:
                return NotImplemented

    def __rmul__(self: "QOBJ",
                 alpha: any) -> "QOBJ":
        """
        Scalar multiplication from the left.

        Parameters
        ----------
        self: QOBJ
            First factor.
        O: any
            Second factor.

        Returns
        ----------
        _: QOBJ
            Product.
        """
        try:
            return __class__.from_csr(numpy.complex128(alpha) * self.data)
        except:
            return NotImplemented

    def __pow__(self: "QOBJ",
                n: int) -> "QOBJ":
        """
        Integer power.

        Parameters
        ----------
        self: QOBJ
            Base.
        n: int
            Power.

        Returns
        ----------
        _: QOBJ
            Input quantum object to the power of n.
        """

        # Check if matrix is square
        dim = self.shape[0]
        if dim != self.shape[1]:
            message = "PYQUANT: INVERSE ONLY ALLOWED FOR SQUARE MATRICES"
            raise TypeError(message)

        # Compute power
        if not n:
            return __class__.from_csr(
                    scipy.sparse.identity(dim, dtype=numpy.complex128).tocsr()
                    )
        elif n < 0:
            return __class__.from_csr(scipy.sparse.linalg.inv(self.data)**n)
        else:
            return __class__.from_csr(self.data**n)

class CQOBJ:
    """
    CQOBJ: Constructors for Quantum OBJects

    Collection of constructors for quantum objects.

    Attributes
    ----------
    """

    def zero(nrows: int,
             ncols: int) -> "QOBJ":
        """
        Zero operator.

        Parameters
        ----------
            nrows: int
                Number of rows.
            ncols: int
                Number of columns.

        Returns
        ----------
            _: QOBJ
                Zero operator.
        """
        if not nrows*ncols:
            message = "PYQUANT: DIMENSION CANNOT BE ZERO"
            raise ValueError(message)
        Z = scipy.sparse.csr_matrix((nrows, ncols), dtype=numpy.complex128)
        return QOBJ(Z, (nrows, ncols))

    def one(dim: int) -> "QOBJ":
        """
        Identity operator.

        Parameters
        ----------
            dim: int
                Dimension of identity.

        Returns
        ----------
            _: QOBJ
                Identity operator.
        """
        E = scipy.sparse.identity(dim,
                                  dtype=numpy.complex128,
                                  format="csr")
        return QOBJ(E, (dim, dim))

    def annihilator(dim: int) -> "QOBJ":
        """
        Bosonic annihilation operator.

        Parameters
        ----------
        dim: int
            Dimension of annihilation operator.

        Returns
        ----------
        _: QOBJ
            Annihilation operator.
        """

        # Get positions and components
        row_indices = numpy.arange(0, dim-1, 1, dtype=int)
        col_indices = numpy.arange(1, dim  , 1, dtype=int)
        components  = numpy.sqrt(col_indices)

        # Place numbers in matrix
        entries = (components, (row_indices, col_indices))
        result  = scipy.sparse.coo_matrix(entries,
                                          shape=(dim, dim),
                                          dtype=numpy.complex128)

        return QOBJ(result.tocsr(), (dim, dim))

    def creator(dim: int) -> "QOBJ":
        """
        Bosonic creation operator.

        Parameters
        ----------
        dim: int
            Dimension of creation operator.

        Returns
        ----------
        _: QOBJ
            Creation operator.
        """

        # Get positions and components
        row_indices = numpy.arange(1, dim  , 1, dtype=int)
        col_indices = numpy.arange(0, dim-1, 1, dtype=int)
        components  = numpy.sqrt(row_indices)

        # Place numbers in matrix
        entries = (components, (row_indices, col_indices))
        result  = scipy.sparse.coo_matrix(entries, shape=(dim, dim),
                                          dtype=numpy.complex128)

        return QOBJ(result.tocsr(), (dim, dim))

    def number(dim: int) -> "QOBJ":
        """
        Bosonic number operator.

        Parameters
        ----------
        dim: int
            Dimension of number operator.

        Returns
        ----------
        _: QOBJ
            Number operator.
        """

        # Get positions and components
        row_indices = numpy.arange(1, dim, 1, dtype=int)
        col_indices = row_indices
        components  = row_indices

        # Place numbers in matrix
        entries = (components, (row_indices, col_indices))
        result  = scipy.sparse.coo_matrix(entries,
                                          shape=(dim, dim),
                                          dtype=numpy.complex128)

        return QOBJ(result.tocsr(), (dim, dim))

    def pauli(which: str) -> "QOBJ":
        """
        Pauli matrix.

        Parameters
        ----------
        which: str
            Either 'x', 'y' or 'z'.

        Returns
        ----------
        _: QOBJ
            Pauli matrix.
        """

        # Define Pauli matrix as array
        sigma = None
        if which=='x':
            sigma = numpy.array([ [  0,   1]   ,
                                  [  1,   0]  ],
                                dtype=numpy.complex128)
        elif which=='y':
            sigma = numpy.array([ [  0, -1j]   ,
                                  [ 1j,   0]  ],
                                dtype=numpy.complex128)
        elif which=='z':
            sigma = numpy.array([ [  1,   0]   ,
                                  [  0,  -1]  ],
                                dtype=numpy.complex128)

        return QOBJ(scipy.sparse.csr_matrix(sigma), (2, 2))

    def proj(dim: int,
             n: int) -> "QOBJ":
        """
        Parameters
        ----------
        dim: int
            Dimension of projector.
        n: int
            Subspace n = 1, ..., dim.

        Returns
        ----------
        _: QOBJ
                                 1 0 0
            dim = 3 and n = 1 => 0 0 0
                                 0 0 0
        """

        # Get entry and shape
        entry = ([1], ([n-1], [n-1]))
        shape = (dim, dim)

        # Define matrix
        result = scipy.sparse.coo_matrix(entry,
                                         shape=shape,
                                         dtype=numpy.complex128)

        return QOBJ(result.tocsr(), (dim, dim))

    def smfock(dim: int,
               n: int,
               which: str) -> "QOBJ":
        """
        Single mode Fock state.

        Parameters
        ----------
        dim: int
            Truncation dimension of Fock space.
        n: int
            Exitation number, n = 0, ..., dim-1.
        which: str
            Either "bra" or "ket".

        Returns.
        _: QOBJ
            Truncated Fock state <n| or |n>.
        """

        # Check which type of state is selected
        entry = None
        shape = None
        if which == "bra":
            entry = ([1], ([0], [n]))
            shape = (1, dim)
        elif which == "ket":
            entry = ([1], ([n], [0]))
            shape = (dim, 1)
        else:
            message = "PYQUANT: ARGUMENT MUST BE EITHER \"bra\" or \"ket\""
            raise ValueError(message)

        # Define matrix
        result = scipy.sparse.coo_matrix(entry,
                                         shape=shape,
                                         dtype=numpy.complex128)

        return QOBJ(result.tocsr(), shape)

    def fock(dim: int,
             ns: list,
             which: str) -> "QOBJ":
        """
        Fock state

        Parameters
        ----------
        dim: int
            Truncation dimension of each mode.
        ns: list
            ns = [n1, ..., nN], ni = 0, ..., dim-1; N: number of modes.
        which: str
            Either "bra" or "ket".
        """
        result = CQOBJ.smfock(dim, ns[0], which).data
        for n in ns[1:]:
            result = scipy.sparse.kron(result, CQOBJ.smfock(dim, n, which).data)
        return QOBJ.from_csr(result.tocsr())

    def cs(dim: int,
           alpha: numpy.complex128,
           which: str) -> "QOBJ":
        """
        Coherent state.

        Parameters
        ----------
        dim: int
            Truncation dimension of Fock space.
        alpha: numpy.complex128
            Coherent state amplitude.
        which: str
            Either "bra" or "ket".

        Returns
        ----------
        _: QOBJ
            Coherent state in basis of Fock states.
        """
        result = CQOBJ.zero(dim, 1)
        for n in range(dim):
            f = CQOBJ.smfock(dim, n, "ket")
            s = alpha**n/numpy.sqrt(scipy.special.gamma(n+1))*f
            result = result + s
        result = numpy.exp(-numpy.abs(alpha)**2/2)*result
        return result if which == "ket" else DAGGER(result)

    def stdbv(dim: int,
              n: int,
              which: str) -> "QOBJ":
        """
        Standard basis vector.

        Parameters
        ----------
        dim: int
            Dimension.
        n: int
            Position of entry, n = 1, ..., dim.
        which: str
            Either "bra" or "ket".

        Result
        ----------
        _: QOBJ
            Standard basis vector (ket), or the dual one (bra).
        """
        return CQOBJ.smfock(dim, n-1, which)

    def annihilators_list(dim: int,
                          number_of_modes: int) -> None:
        """
        Bosonic multimode annihilation operators.

        Parameters
        ----------
        dim: int
            Truncation dimension of each mode.
        number_of_modes: int
            Number of modes.

        Returns
        ----------
        _: list
            List of annihilation operators.
            Example: number_of_modes = 2 returns [a x 1, 1 x a]
        """

        # Single mode annihilator
        a = CQOBJ.annihilator(dim)

        # Check if something is to be done
        if not number_of_modes:
            return []
        elif number_of_modes == 1:
            return [ a ]

        # Push modes to subspaces
        result = [ OQOBJ.to_subspace(0, a, dim**(number_of_modes-1)) ]
        for i in range(1, number_of_modes-1):
            result += [ OQOBJ.to_subspace(dim**i,
                                          a,
                                          dim**(number_of_modes-i-1)) ]
        result += [ OQOBJ.to_subspace(dim**(number_of_modes-1), a, 0) ]

        return result

    def creators_list(dim: int,
                      number_of_modes: int) -> None:
        """
        Bosonic multimode creation operators.

        Parameters
        ----------
        dim: int
            Truncation dimension of each mode.
        number_of_modes: int
            Number of modes.

        Returns
        ----------
        _: list
            List of creation operators.
            Example: number_of_modes = 2 returns [a x 1, 1 x a]
        """

        # Single mode creator
        ad = CQOBJ.creator(dim)

        # Check if something is to be done
        if not number_of_modes:
            return []
        elif number_of_modes == 1:
            return [ a ]

        # Push modes to subspaces
        result = [ OQOBJ.to_subspace(0, a, dim**(number_of_modes-1)) ]
        for i in range(1, number_of_modes-1):
            result += [ OQOBJ.to_subspace(dim**i,
                                          a,
                                          dim**(number_of_modes-i-1)) ]
        result += [ OQOBJ.to_subspace(dim**(number_of_modes-1), a, 0) ]

        return result

    def numbers_list(dim: int,
                     number_of_modes: int) -> None:
        """
        Bosonic multimode number operators.

        Parameters
        ----------
        dim: int
            Truncation dimension of each mode.
        number_of_modes: int
            Number of modes.

        Returns
        ----------
        _: list
            List of number operators.
            Example: number_of_modes = 2 returns [a x 1, 1 x a]
        """

        # Single mode number operator
        ad = CQOBJ.number(dim)

        # Check if something is to be done
        if not number_of_modes:
            return []
        elif number_of_modes == 1:
            return [ a ]

        # Push modes to subspaces
        result = [ OQOBJ.to_subspace(0, a, dim**(number_of_modes-1)) ]
        for i in range(1, number_of_modes-1):
            result += [ OQOBJ.to_subspace(dim**i,
                                          a,
                                          dim**(number_of_modes-i-1)) ]
        result += [ OQOBJ.to_subspace(dim**(number_of_modes-1), a, 0) ]

        return result

    def basis(dim: int) -> list:
        """
        Standard basis.

        Parameters
        ----------
        dim: int
            Number of basis vectors.

        Returns
        ----------
        _: list
            List of the dim-dimensional statndard basis vectors e1, ..., edim.
        """

        # Generate list of basis vectors
        shape = (dim, 1)
        entry = None
        basis = list()
        for i in range(dim):
            entry = ([1], ([i], [0]))
            state = scipy.sparse.coo_matrix(entry,
                                            shape=shape,
                                            dtype=numpy.complex128)
            state = QOBJ(state.tocsr(), shape)
            basis.append(state)

        return basis

class OQOBJ:
    """
    OQOBJ: Operations on Quantum OBJects

    Collection of operations on quantum objects.

    Attributes
    ----------
    """

    def dagger(O: "QOBJ") -> "QOBJ":
        """
        Hermitian conjugate.

        Parameters
        ----------
        O: QOBJ
            Quantum object.

        Returns
        ----------
            Hermitian conjugate of input.
        """
        return QOBJ.from_csr(O.data.getH().tocsr())

    def tp(Os: list) -> "QOBJ":
        """
        Tensor product

        Parameters
        ----------
        Os: list
            List of quantum objects.

        Returns
        ----------
        _: QOBJ
            Tensor product of the input's elements from left to right.
        """
        result = Os[0].data
        for O in Os[1:]:
            result = scipy.sparse.kron(result, O.data)
        return QOBJ.from_csr(result.tocsr())

    def dyad(ket: "QOBJ") -> "QOBJ":
        """
        Dyadic product.

        Parameters
        ----------
        ket: QOBJ
            Ket state.

        _: QOBJ
            Dyad |ket><ket|.
        """
        if ket.shape[1] != 1:
            message = "PYQUANT: DYADIC PRODUCT IS ONLY DEFINED FOR A KET STATE"
            raise TypeError(message)
        else:
            dyad = scipy.sparse.kron(ket.data, ket.data.getH())
            return QOBJ.from_csr(dyad.tocsr())

    def sp(phi: "QOBJ",
           psi: "QOBJ") -> numpy.complex128:
        """
        Scalar product.

        Input:
        phi: QOBJ
            Ket state.
        psi: QOBJ
            Ket state.

        Output:
            numpy.complex128: <phi|psi>
                Scalar product between input states.
        """
        if phi.shape == psi.shape and phi.shape[1] == 1:
            return (OQOBJ.dagger(phi) * psi).data.toarray()[0][0]
        else:
            message = "PYQUANT: INVALID INPUT GIVEN TO SCALAR PRODUCT"
            raise TypeError(message)

    def to_subspace(dim1: int,
                    O: "QOBJ",
                    dim2: int) -> "QOBJ":
        """
        Push quadratic operator to a subspace.

        Parameters
        ----------
        dim1: int
            Dimension of identity before O.
        O: QOBJ
            Operator.
        dim2: int
            Dimension of identity after O.

        Returns
        ----------
        _: QOBJ:
            Tensor product of Edim1, A, and Edim2 in that order.
        """

        ### Check of operator is quadratic
        if O.shape[0] != O.shape[1]:
            message = "PYQUANT: OPERATION ONLY DEFINED FOR QUADRATIC OPERATORS"
            raise TypeError(message)

        ### Compute tensor products
        if not dim1 and not dim2:
            return O
        elif not dim1:
            return CQOBJ.tp([O, ONE(dim2)])
        elif not dim2:
            return CQOBJ.tp([ONE(dim1), O])
        else:
            return CQOBJ.tp([ONE(dim1), O, ONE(dim2)])

class SPEC:
    """
    SPEC: SPECtra

    Class of methods to compute spectra.

    Attributes
    ----------
    """

    def __check_square_compute_all(O: "QOBJ",
                                   howmany: (int, str)) -> bool:
        """
        Check if matrix is square and all eigenvalues and eigenvectors ought to
        be computed.

        Parameters
        ----------
        O: QOBJ
            Quantum object.
        howmany: int, str
            Number of eigenvalues and eigenvectors to compute; can be "all".

        Returns
        ----------
        _: bool
            'True' if all eigenvalues and eigenvectors ought to be cmputed, and
            'False' otherwise.
        """

        # Make sure O is square
        if O.shape[0] != O.shape[1]:
            message = "PYQUANT: CANNOT COMPUTE EIGENVALUES OF NON-SQUARE MATRIX"
            raise TypeError(message)

        # Check if all eigenvalues and eigenvectors shall be computed
        if howmany == "all" or howmany > O.shape[0]-2:
            return True
        else:
            return False

    def eigen_symm(O: "QOBJ",
                   get_states: bool,
                   which: str,
                   sigma: numpy.float64,
                   howmany: (int, str)) -> list:
        """
        Compute eigenvalues and eigenvectors of a real symmetric matrix.

        Parameters
        ----------
        O: QOBJ
            Symmetric operator. It is assumed that the imaginary part is zero,
            but there is NO INTERNAL CHECK, i.e., it is the caller's
            responsibility.
        get_states: bool
            Decide if eigenvectors are computed.
        which: str
            "LM": largest magnitude
            "SM": smallest magnitude
            "LA": largest algebraic
            "SA": smallest algebraic
            "BE": k/2 from high and k/2 from low end
        sigma: numpy.float64
            Shift for shift-and-invert mode of ARPACK. If 'None', no shift is
            used. Typically, one needs to use shift-and-invert mode to speed up
            the calculation of smallest magnitude eigenvalues, because "SM"
            itself is very slow for large matrices.
        howmany: int, str
            Number of eigenvalues and eigenvectors ought to be computed; can be
            "all".

        Returns
        ----------
        _: list
            List of the form
                [ numpy_array_of_eigenvalues, numpy_array_of_eigenstates ] .
            Eigenstates are (dim,) numpy arrays, not sparse matrices.
            numpy_array_of_eigenstates[i] is the eigenvector to the eigenvalue
            numpy_array_of_eigenvalues[i]. If get_states is 'False',
            list_of_eigenstates = [].
        """

        # Check for "all" eigenvalues and eigenvectors and check if O is square
        get_all = SPEC.__check_square_compute_all(O, howmany)

        # Get the real part, because it is assumed that Im(O)=0!
        Oreal = O.data.real

        # Compute eigenvalues and eigenvectors
        if get_all:
            if get_states:
                energies, states = numpy.linalg.eigh(Oreal.toarray())
                return [ energies, states.T ]
            else:
                energies = numpy.linalg.eigvalsh(Oreal.toarray())
                return [ energies, [] ]
        else:
            result = scipy.sparse.linalg.eigsh(Oreal,
                                               k=howmany,
                                               which=which,
                                               sigma=sigma,
                                               return_eigenvectors=get_states,
                                               tol=EPSILON)
            if get_states:
                return [ result[0], result[1].T ]
            else:
                return [ result, [] ]

    def eigen_herm(O: "QOBJ",
                   get_states: bool,
                   which: str,
                   sigma: numpy.float64,
                   howmany: (int, str)) -> list:
        """
        Compute eigenvalues and eigenvectors of a hermitian matrix.

        Parameters
        ----------
        O: QOBJ
            Hermitian operator.
        get_states: bool
            Decide if eigenvectors are computed.
        which: str
            "LM": largest magnitude
            "SM": smallest magnitude
            "LA": largest algebraic
            "SA": smallest algebraic
            "BE": k/2 from high and k/2 from low end
        sigma: numpy.float64
            Shift for shift-and-invert mode of ARPACK. If 'None', no shift is
            used. Typically, one needs to use shift-and-invert mode to speed up
            the calculation of smallest magnitude eigenvalues, because "SM"
            itself is very slow for large matrices.
        howmany: int, str
            Number of eigenvalues and eigenvectors ought to be computed; can be
            "all".

        Returns
        ----------
        _: list
            List of the form
                [ numpy_array_of_eigenvalues, numpy_array_of_eigenstates ] .
            Eigenstates are (dim,) numpy arrays, not sparse matrices.
            numpy_array_of_eigenstates[i] is the eigenvector to the eigenvalue
            numpy_array_of_eigenvalues[i]. If get_states is 'False',
            list_of_eigenstates = [].
        """

        # Check for "all" eigenvalues and eigenvectors and check if O is square
        get_all = SPEC.__check_square_compute_all(O, howmany)

        # Compute eigenvalues and eigenvectors
        if get_all:
            if get_states:
                energies, states = numpy.linalg.eigh(O.data.toarray())
                return [ energies, states.T ]
            else:
                energies = numpy.linalg.eigvalsh(O.data.toarray())
                return [ energies, [] ]
        else:
            result = scipy.sparse.linalg.eigsh(O.data,
                                               k=howmany,
                                               which=which,
                                               sigma=sigma,
                                               return_eigenvectors=get_states,
                                               tol=EPSILON)
            if get_states:
                return [ result[0], result[1].T ]
            else:
                return [ result, [] ]

    def eigen_gen_real(O: "QOBJ",
                       get_states: bool,
                       which: str,
                       sigma: numpy.float64,
                       howmany: (int, str)) -> list:
        """
        Compute eigenvalues and eigenvectors of a general real matrix.

        Parameters
        ----------
        O: QOBJ
            Real operator. It is assumed that the imaginary part is zero, but
            there is NO INTERNAL CHECK, i.e., it is the caller's responsibility.
        get_states: bool
            Decide if eigenvectors are computed.
        which: str
            "LM": largest magnitude
            "SM": smallest magnitude
            "LR": largest real part
            "SR": smallest real part
            "LI": largest imaginary part
            "SI": smallest imaginary part
        sigma: numpy.float64
            Shift for shift-and-invert mode of ARPACK. If 'None', no shift is
            used. Typically, one needs to use shift-and-invert mode to speed up
            the calculation of smallest magnitude eigenvalues, because "SM"
            itself is very slow for large matrices.
        howmany: int, str
            Number of eigenvalues and eigenvectors ought to be computed; can be
            "all".

        Returns
        ----------
        _: list
            List of the form
                [ numpy_array_of_eigenvalues, numpy_array_of_eigenstates ] .
            Eigenstates are (dim,) numpy arrays, not sparse matrices.
            numpy_array_of_eigenstates[i] is the eigenvector to the eigenvalue
            numpy_array_of_eigenvalues[i]. If get_states is 'False',
            list_of_eigenstates = [].
        """

        # Check for "all" eigenvalues and eigenvectors and check if O is square
        get_all = SPEC.__check_square_compute_all(O, howmany)

        # Get the real part, because it is assumed that Im(O)=0!
        Oreal = O.data.real

        # Compute eigenvalues and eigenvectors
        if get_all:
            if get_states:
                energies, states = numpy.linalg.eig(Oreal.toarray())
                return [ energies, states.T ]
            else:
                energies = numpy.linalg.eigvals(Oreal.toarray())
                return [ energies, [] ]
        else:
            result = scipy.sparse.linalg.eigs(Oreal,
                                              k=howmany,
                                              which=which,
                                              sigma=sigma,
                                              return_eigenvectors=get_states,
                                              tol=EPSILON)
            if get_states:
                return [ result[0], result[1].T ]
            else:
                return [ result, [] ]

    def eigen_gen(O: "QOBJ",
                  get_states: bool,
                  which: str,
                  sigma: numpy.float64,
                  howmany: (int, str)) -> list:
        """
        Compute eigenvalues and eigenvectors of a general matrix.

        Parameters
        ----------
        O: QOBJ
            Operator.
        get_states: bool
            Decide if eigenvectors are computed.
        which: str
            "LM": largest magnitude
            "SM": smallest magnitude
            "LR": largest real part
            "SR": smallest real part
            "LI": largest imaginary part
            "SI": smallest imaginary part
        sigma: numpy.float64
            Shift for shift-and-invert mode of ARPACK. If 'None', no shift is
            used. Typically, one needs to use shift-and-invert mode to speed up
            the calculation of smallest magnitude eigenvalues, because "SM"
            itself is very slow for large matrices.
        howmany: int, str
            Number of eigenvalues and eigenvectors ought to be computed; can be
            "all".

        Returns
        ----------
        _: list
            List of the form
                [ numpy_array_of_eigenvalues, numpy_array_of_eigenstates ] .
            Eigenstates are (dim,) numpy arrays, not sparse matrices.
            numpy_array_of_eigenstates[i] is the eigenvector to the eigenvalue
            numpy_array_of_eigenvalues[i]. If get_states is 'False',
            list_of_eigenstates = [].
        """

        # Check for "all" eigenvalues and eigenvectors and check if O is square
        get_all = SPEC.__check_square_compute_all(O, howmany)

        # Compute eigenvalues and eigenvectors
        if get_all:
            if get_states:
                energies, states = numpy.linalg.eig(O.data.toarray())
                return [ energies, states.T ]
            else:
                energies = numpy.linalg.eigvals(O.data.toarray())
                return [ energies, [] ]
        else:
            result = scipy.sparse.linalg.eigs(O.data,
                                              k=howmany,
                                              which=which,
                                              sigma=sigma,
                                              return_eigenvectors=get_states,
                                              tol=EPSILON)
            if get_states:
                return [ result[0], result[1].T ]
            else:
                return [ result, [] ]

class TEVOL:
    """
    TEVOL: Time EVOLution

    Class of methods to compute time evolution.

    Attributes
    ----------
    """

    def __action_of_hamiltonian(h0: scipy.sparse.csr_matrix,
                                hs: list,
                                fs: list,
                                t: numpy.float64,
                                psi: numpy.array) -> numpy.array:
        """
        Compute action of Hamiltonian.

        Parameters
        ----------
        h0: scipy.sparse.csr_matrix
            Hamiltonian as sparse CSR matrix.
        hs: list
            List of sparse hermitian CSR matrices.
        fs: list
            List of scalar functions depending on time.

        Returns
        ----------
        _: numpy.array
            Resulting state.
        """
        H = h0
        for f, h in zip(fs, hs):
            H = H + f(t) * h
        return numpy.complex128(-1j) * H.dot(psi)

    def __action_of_lindbladdian(h0: scipy.sparse.csr_matrix,
                                 hs: list,
                                 fs: list,
                                 Llst: list,
                                 t: numpy.float64,
                                 rho: numpy.array,
                                 shape: tuple) -> numpy.array:
        """
        Compute action of Lindbladdian.

        Parameters
        ----------
        h0: scipy.sparse.csr_matrix
            Hamiltonian as sparse CSR matrix.
        hs: list
            List of sparse hermitian CSR matrices.
        fs: list
            List of scalar functions depending on time.
        Llst: list
            List of scaled jump operators: sqrt(gamma_i)Li, with the rate gammai
            associated to the jump oprator Li. sqrt(gamma_1)L_1,
            sqrt(gamma_2)L_2, ... must be quantum objects.
        t: numpy.float64
            Time point.
        rho: numpy.array
            Current state.
        shape: tuple
            Shape of current state.

        Returns
        ----------
        _: numpy.array
            Resulting state.
        """

        # Bring rho into quadratic form
        rho_quad = rho.reshape(shape, order='C', copy=False)

        # Compute Hamiltonian at time t
        H = h0
        for f, h in zip(fs, hs):
            H = H + f(t) * h

        # Compute action of Hamiltonian
        result = H.dot(rho_quad)
        result -= numpy.conj(result.T) # Do not copy
        result *= numpy.complex128(-1j) # Do not copy

        # Add action of dissipator
        for L in Llst:
            tmp = L.data.dot(rho_quad)
            result += L.data.dot(numpy.conj(tmp.T))
            tmp = L.data.getH().dot(tmp)
            tmp += numpy.conj(tmp.T)
            result += numpy.complex128(-.5) * tmp

        return result.reshape((shape[0]*shape[1], 1), order='C', copy=False)

    def uevol_expm(H: "QOBJ",
                   psi0: "QOBJ",
                   times: numpy.array,
                   get_states: (bool, str),
                   observables: list,
                   print_info: bool) -> list:
        """
        H: QOBJ
            Quantum object representing the time-independent Hamiltonian.
        psi0: QOBJ
            Initial state.
        times: numpy.array
            Time points of interest.
        get_states: bool, str
            Decide if states are saved and returned.
            If equal to "final", only the final state is saved.
        observables: list
            Observables to compute expectation values from, at each specified
            time point. Observables can be time-dependent using the same
            formatting as for Hlst in the method 'uevol_ode'.
        print_info: bool
            If 'True', print information on integration process.

        Returns
        ----------
        list: [ times, states, <observable 1>, ..., <observable last> ]
              States are numpy arrays of shape (dim,) not sparse.
              If get_states if 'False', states = [].
              Formatting: <observable i> = [ <observable t0>, ... ]
        """

        # Check if only final state is desired
        if get_states == "final":
            get_states_ = False
        else:
            get_states_ = get_states

        # Convert 'psi0' to dense
        psi0_ = psi0.data.toarray()

        # Find out which observables carry explicit time dependence
        is_timedep = []
        for observable in observables:
            is_timedep += [ isinstance(observable, list) ]

        # Lists for states and expectation values of observables
        states = []
        expect_observables = []

        # Add elements for smallest time
        print("\n\nPYQUANT: STARTING INTEGRATION USING EXPONETIAL MULTIPLY\n")
        if get_states_: states += [ psi0_.T[0] ]
        ev = None
        expects = []
        for i, observable in enumerate(observables):
            if is_timedep[i]:
                ct = numpy.complex128(observable[0](times[0]))
                ev = observable[1].data.dot(psi0_)
                ev = ct * numpy.conj(psi0_.T).dot(ev)
            else:
                ev = observable.data.dot(psi0_)
                ev = numpy.conj(psi0_.T).dot(ev)
            expects += [ ev[0][0] ]
        expect_observables += [ expects ]
        if print_info:
            norm = numpy.real(numpy.sqrt(numpy.conj(psi0_.T).dot(psi0_))[0][0])
            print("TIME: {:1.3E} - NORM: {:1.3E}".format(times[0], norm))

        # Loop through remaining time points
        psit = psi0_
        t0 = times[0]
        for t in times[1:]:

            # Get state at time t
            psit = scipy.sparse.linalg.expm_multiply(-1j*H.data*(t-t0), psit)
            t0 = t

            # Collect state
            if get_states_: states += [ psit.T[0] ]

            # Compute expectation values
            expects = []
            for i, observable in enumerate(observables):
                if is_timedep[i]:
                    ct = numpy.complex128(observable[0](t))
                    ev = observable[1].data.dot(psit)
                    ev = ct * numpy.conj(psit.T).dot(ev)
                else:
                    ev = observable.data.dot(psit)
                    ev = numpy.conj(psit.T).dot(ev)
                expects += [ ev[0][0] ]
            expect_observables += [ expects ]
            if print_info:
                norm = numpy.sqrt(numpy.conj(psit.T).dot(psit))[0][0]
                norm = numpy.real(norm)
                print("TIME: {:1.3E} - NORM: {:1.3E}".format(t, norm))

        # Prepare lists for return
        if get_states == "final": states += [ psit.T[0] ]
        result = [ times, states ]
        expect_observables = numpy.array(expect_observables).T
        for expects in expect_observables:
            result += [ expects ]
        print("\n\nPYQUANT: INTEGRATION SUCCESSFUL\n\n")

        return result

    def uevol_ode(Hlst: list,
                  psi0: "QOBJ",
                  times: numpy.array,
                  get_states: bool,
                  is_stiff: bool,
                  observables: list,
                  print_info: bool) -> list:
        """
        Hls: list
            List of quantum objects representing the (potentially time-dependent
            potential).
            The format is

                Hlst = [ H0, [f1, H1], [f2, H2], ... ] ,

            where H0, H1, H2 are hermitian operators and f1, f2, ... are scalar
            functions of time. The Hamiltonin is constructed as follows:

                   Htot(t) = H0 + f(1)*H1 + f2(t)*H2 + ... .

            Note that Hlst = [ H0 ] is valid and yields a time-independent
            Hamiltonian.
        psi0: QOBJ
            Initial state.
        times: numpy.array
            Time points of interest.
        get_states: bool, str
            Decide if states are saved and returned.
            If equal to "final", only the final state is saved.
        is_stiff: bool
            Decide if ODE system is stiff, i.e., if 'BDF' is used, instead of
            'adams'.
        observables: list
            Observables to compute expectation values from, at each specified
            time point. Observables can be time-dependent using the same
            formatting as for Hlst.
        print_info: bool
            If 'True', print information on integration process.

        Returns
        ----------
        list: [ times, states, <observable 1>, ..., <observable last> ]
              States are numpy arrays of shape (dim,) not sparse.
              If get_states if 'False', states = [].
              Formatting: <observable i> = [ <observable t0>, ... ]
        """

        # Check if only final state is desired
        if get_states == "final":
            get_states_ = False
        else:
            get_states_ = get_states

        # Check if there is a time-independent part to the Hamiltonian
        h0_exsits = not isinstance(Hlst[0], list)

        # See how many contributions to the Hamiltonian there are
        l = len(Hlst)

        # Extract data of Hamiltonians
        h0 = Hlst[0].data if h0_exsits else ZERO(*Hlst[0][1].shape).data
        i = 1 if h0_exsits else 0
        hs = []; fs = []
        while i < l:
            fs += [ Hlst[i][0] ]
            hs += [ Hlst[i][1].data ]
            i += 1

        # Convert 'psi0' to dense
        psi0_ = psi0.data.toarray()

        # Find out which observables carry explicit time dependence
        is_timedep = []
        for observable in observables:
            is_timedep += [ isinstance(observable, list) ]

        # Define ODE system without initial condition
        F = lambda t, psi: TEVOL.__action_of_hamiltonian(h0, hs, fs, t, psi)
        system = scipy.integrate.ode(F)

        # Set integrator
        method = "BDF" if is_stiff else "adams"
        system.set_integrator("zvode", method=method)

        # Set initial condition
        system.set_initial_value(psi0_, times[0])

        # List for states and expectation values of observables
        states = []
        expect_observables = []

        # Add elements for smallest time
        print("\n\nSTARTING INTEGRATION USING METHOD *{:s}*\n".format(method))
        if get_states_: states += [ psi0_.T[0] ]
        ev = None
        expects = []
        for i, observable in enumerate(observables):
            if is_timedep[i]:
                ct = numpy.complex128(observable[0](times[0]))
                ev = observable[1].data.dot(psi0_)
                ev = ct * numpy.conj(psi0_.T).dot(ev)
            else:
                ev = observable.data.dot(psi0_)
                ev = numpy.conj(psi0_.T).dot(ev)
            expects += [ ev[0][0] ]
        expect_observables += [ expects ]
        if print_info:
            norm = numpy.real(numpy.sqrt(numpy.conj(psi0_.T).dot(psi0_))[0][0])
            print("TIME: {:1.3E} - NORM: {:1.3E}".format(times[0], norm))

        # Loop through remaining time points
        psit = psi0_
        t0 = times[0]
        for t in times[1:]:

            # Get state at time t
            psit = system.integrate(system.t+t-t0)
            t0 = t

            # Check if integration step was successful
            if not system.successful():
                print("\n\nINTEGRATION NOT SUCCESSFUL -> EXITING\n\n")
                exit(1)

            # Gather states
            if get_states_: states += [ psit.T[0] ]

            # Compute expectation values
            expects = []
            for i, observable in enumerate(observables):
                if is_timedep[i]:
                    ct = numpy.complex128(observable[0](t))
                    ev = observable[1].data.dot(psit)
                    ev = ct * numpy.conj(psit.T).dot(ev)
                else:
                    ev = observable.data.dot(psit)
                    ev = numpy.conj(psit.T).dot(ev)
                expects += [ ev[0][0] ]
            expect_observables += [ expects ]
            if print_info:
                norm = numpy.sqrt(numpy.conj(psit.T).dot(psit))[0][0]
                norm = numpy.real(norm)
                print("TIME: {:1.3E} - NORM: {:1.3E}".format(t, norm))

        # Prepare list to return
        if get_states == "final": states += [ psit.T[0] ]
        result = [ times, states ]
        expect_observables = numpy.array(expect_observables).T
        for expects in expect_observables:
            result += [ expects ]
        print("\n\nINTEGRATION SUCCESSFUL\n\n")

        return result

    def devol_lindblad(Hlst: list,
                       Llst: list,
                       rho0: "QOBJ",
                       times: numpy.array,
                       get_states: bool,
                       is_stiff: bool,
                       observables: list,
                       print_info: bool) -> list:
        """
        Hls: list
            List of quantum objects representing the (potentially time-dependent
            potential).
            The format is

                Hlst = [ H0, [f1, H1], [f2, H2], ... ] ,

            where H0, H1, H2 are hermitian operators and f1, f2, ... are scalar
            functions of time. The Hamiltonin is constructed as follows:

                   Htot(t) = H0 + f(1)*H1 + f2(t)*H2 + ... .

            Note that Hlst = [ H0 ] is valid and yields a time-independent
            Hamiltonian.
        Llst: list
            List of scaled jump operators: sqrt(gamma_i)Li, with the rate gammai
            associated to the jump oprator Li. sqrt(gamma_1)L_1,
            sqrt(gamma_2)L_2, ... must be quantum objects.
        rho0: QOBJ
            Initial state.
        times: numpy.array
            Time points of interest.
        get_states: bool, str
            Decide if states are saved and returned.
            If equal to "final", only the final state is saved.
        is_stiff: bool
            Decide if ODE system is stiff, i.e., if 'BDF' is used, instead of
            'adams'.
        observables: list
            Observables to compute expectation values from, at each specified
            time point. Observables can be time-dependent using the same
            formatting as for Hlst.
        print_info: bool
            If 'True', print information on integration process.

        Returns
        ----------
        list: [ times, states, <observable 1>, ..., <observable last> ]
              States are numpy arrays of shape (dim,) not sparse.
              If get_states if 'False', states = [].
              Formatting: <observable i> = [ <observable t0>, ... ]
        """

        # Check if only final state is desired
        if get_states == "final":
            get_states_ = False
        else:
            get_states_ = get_states

        # Check if there is a time-independent part to the Hamiltonian
        try:
            h0_exsits = not isinstance(Hlst[0], list)
        except:
            h0_exsits = False

        # See how many contributions to the Hamiltonian there are
        l = len(Hlst)

        # Extract data of Hamiltonians
        h0 = Hlst[0].data if h0_exsits else ZERO(*Llst[0].shape).data
        i = 1 if h0_exsits else 0
        hs = []; fs = []
        while (i<l):
            fs += [ Hlst[i][0] ]
            hs += [ Hlst[i][1].data ]
            i += 1

        # Convert 'rho0' to dense
        rho0_ = rho0.data.toarray()

        # Find out which observables carry explicit time dependence
        is_timedep = []
        for observable in observables:
            is_timedep += [ isinstance(observable, list) ]

        # Define ODE system without initial condition
        F = lambda t, rho: TEVOL.__action_of_lindbladdian(h0,
                                                          hs,
                                                          fs,
                                                          Llst,
                                                          t,
                                                          rho,
                                                          rho0.shape)
        system = scipy.integrate.ode(F)

        # Set integrator
        method = "BDF" if is_stiff else "adams"
        system.set_integrator("zvode", method=method)

        # List for states and expectation values of observables
        states = []
        expect_observables = []

        # Add elements for smallest time
        print("\n\nSTARTING INTEGRATION USING METHOD *{:s}*\n".format(method))
        if get_states_: states += [ rho0_ ]
        ev = None
        expects = []
        for i, observable in enumerate(observables):
            if is_timedep[i]:
                ct = numpy.complex128(observable[0](times[0]))
                ev = observable[1].data.dot(rho0_).trace()
                ev = ct * ev
            else:
                ev = observable.data.dot(rho0_).trace()
            expects += [ ev ]
        expect_observables += [ expects ]
        if print_info:
            trace = numpy.abs(numpy.trace(rho0_))
            print("TIME: {:1.3E} - TRACE: {:1.3E}".format(times[0], trace))

        # Set initial condition (data is not copied by reshape)
        rho0_ = rho0_.reshape((rho0.shape[0]*rho0.shape[0], 1),
                              order='C',
                              copy=False)
        system.set_initial_value(rho0_, times[0])

        # Loop through remaining time points
        rhot = rho0_
        t0 = times[0]
        for t in times[1:]:

            # Get state at time t
            rhot = system.integrate(system.t+t-t0)
            rhot_quad = rhot.reshape(rho0.shape, order='C', copy=False)
            t0 = t

            # Check if integration step was successful
            if not system.successful():
                print("\n\nINTEGRATION NOT SUCCESSFUL -> EXITING\n\n")
                exit(1)

            # Gather states
            if get_states_: states += [ rhot_quad ]

            # Compute expectation values
            expects = []
            for i, observable in enumerate(observables):
                if is_timedep[i]:
                    ct = numpy.complex128(observable[0](t))
                    ev = observable[1].data.dot(rhot_quad).trace()
                    ev = ct * ev
                else:
                    ev = observable.data.dot(rhot_quad).trace()
                expects += [ ev ]
            expect_observables += [ expects ]
            if print_info:
                trace = numpy.abs(numpy.trace(rhot_quad))
                print("TIME: {:1.3E} - TRACE: {:1.3E}".format(t, trace))

        # Prepare list to return
        rhot = rhot.reshape(rho0.shape, order='C', copy=False)
        if get_states == "final": states += [ rhot ]
        expect_observables = numpy.array(expect_observables).T
        result = [ times, states ]
        for expects in expect_observables:
            result += [ expects ]
        print("\n\nINTEGRATION SUCCESSFUL\n\n")

        return result

class QOPS:
    """
    QOPS: Quantum Optical Phase Space

    Class of methods to work with quantum optical phase space quantities.
    """

    def WIGNER_FOCK(rho: "QOBJ",
                    alpha: numpy.complex128) -> numpy.float64:
        """
        Wigner quasiprobability distribution for density matrix in Fock basis.

        Parameters
        ----------
        rho: QOBJ
            State in Fock basis.
        alpha:
            Argument of Wigner quasiprobability distribution.

        Returns
        ----------
        result: numpy.float64
            Value of Wigner quasiprobability distribution.
        """

        # Get non-zero entries and indices
        data = rho.data.toarray()

        # Compute contributions to Wigner-function
        offdiag = 0; diag = 0
        c1 = 2/numpy.pi
        c2 = 2*numpy.abs(alpha)**2
        c3 = numpy.exp(-c2)
        for m in range(rho.shape[0]):
            for n in range(rho.shape[1]):

                # Continue if contribution is zero
                if numpy.abs(data[m][n]) < EPSILON:
                    continue

                # Diagonal
                if m == n:
                    Wmm =   c1 \
                          * (-1)**n \
                          * c3 \
                          * scipy.special.genlaguerre(n, 0)(2*c2)
                    diag = diag + data[m][m] * Wmm
                    continue

                # Off-diagonal
                if m-n > -1:
                    Wmn =   c1 \
                          * (-1)**n \
                          * numpy.sqrt(   scipy.special.gamma(n+1)
                                        / scipy.special.gamma(m+1) ) \
                          * (2*numpy.conj(alpha))**(m-n) \
                          * c3 \
                          * scipy.special.genlaguerre(n, m-n)(2*c2)
                    offdiag = offdiag + data[m][n] * Wmn

        # Combine contributions to obtain Wigner-function
        result = diag + 2*numpy.real(offdiag)

        # Adjust format; using that the Wigner-function is real
        result = numpy.float64(numpy.real(result))

        return result
