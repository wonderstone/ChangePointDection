package cpd

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// ConstantHazard computes the hazard function for Bayesian online learning.
// Arguments:
//
//	lam - initial probability
//	r - R matrix
func ConstantHazard(lam float64, r *mat.Dense) *mat.Dense {
	rRows, rCols := r.Dims()
	result := mat.NewDense(rRows, rCols, nil)
	for i := 0; i < rRows; i++ {
		for j := 0; j < rCols; j++ {
			result.Set(i, j, 1/lam)
		}
	}
	return result
}

type StudentT struct {
	alpha, beta, kappa, mu     []float64
	alpha0, beta0, kappa0, mu0 []float64
}

func NewStudentT(t_alpha, t_beta, t_kappa, t_mu []float64) *StudentT {
	st := &StudentT{
		alpha:  make([]float64, len(t_alpha)),
		beta:   make([]float64, len(t_beta)),
		kappa:  make([]float64, len(t_kappa)),
		mu:     make([]float64, len(t_mu)),
		alpha0: make([]float64, len(t_alpha)),
		beta0:  make([]float64, len(t_beta)),
		kappa0: make([]float64, len(t_kappa)),
		mu0:    make([]float64, len(t_mu)),
	}
	copy(st.alpha, t_alpha)
	copy(st.beta, t_beta)
	copy(st.kappa, t_kappa)
	copy(st.mu, t_mu)
	copy(st.alpha0, t_alpha)
	copy(st.beta0, t_beta)
	copy(st.kappa0, t_kappa)
	copy(st.mu0, t_mu)
	return st

}

func (st *StudentT) PDF(data []float64) [][]float64 {
	// check all the parameters are the same length
	if len(st.alpha) != len(st.beta) || len(st.alpha) != len(st.kappa) || len(st.alpha) != len(st.mu) {
		panic("Parameters alpha, beta, kappa, and mu must have the same length")
	}
	res := make([][]float64, 0)
	for i := 0; i < len(st.alpha); i++ {
		scale := math.Sqrt(st.beta[i] * (st.kappa[i] + 1) / (st.alpha[i] * st.kappa[i]))
		pdfs := make([]float64, len(data))
		tDist := distuv.StudentsT{Mu: st.mu[i], Sigma: scale, Nu: 2 * st.alpha[i]}
		for j, x := range data {
			pdfs[j] = tDist.Prob(x)
		}
		res = append(res, pdfs)
	}

	return res
}



func (st *StudentT) UpdateTheta(data []float64) {

	// check all the parameters are the same length
	if len(st.alpha) != len(st.beta) || len(st.alpha) != len(st.kappa) || len(st.alpha) != len(st.mu) {
		panic("Parameters alpha, beta, kappa, and mu must have the same length")
	}
	
	// check the length of the data and make it equal to the length of the parameters
	// if the length of the data is less than the length of the parameters, then append the data with the last element of the data
	tmpdata := make([]float64,len(data))
	copy(tmpdata,data)
	
	if len(tmpdata) < len(st.alpha) {
		lastElement := tmpdata[len(tmpdata)-1]
		for len(tmpdata) < len(st.alpha) {
			tmpdata = append(tmpdata, lastElement)
		}
	}

	// change data into mat.vecDense format with the same length as the parameters
	tdata := mat.NewVecDense(len(st.alpha), tmpdata)
	talpha := mat.NewVecDense(len(st.alpha), st.alpha)
	tbeta := mat.NewVecDense(len(st.alpha), st.beta)
	tkappa := mat.NewVecDense(len(st.alpha), st.kappa)
	tmu := mat.NewVecDense(len(st.alpha), st.mu)
	talpha0 := mat.NewVecDense(len(st.alpha0), st.alpha0)
	tbeta0 := mat.NewVecDense(len(st.alpha0), st.beta0)
	tkappa0 := mat.NewVecDense(len(st.alpha0), st.kappa0)
	tmu0 := mat.NewVecDense(len(st.alpha0), st.mu0)

	muT0 := func(mu0, kappa, mu, data *mat.VecDense) *mat.VecDense {
		tmp := mat.NewVecDense(kappa.Len(), nil)
		tmp.MulElemVec(kappa, mu)
		tmp.AddVec(tmp, data)
		tmp0 := AddConstant(kappa, 1.0)
		tmp.DivElemVec(tmp, tmp0)
		return ConcatenateVertically(mu0, tmp)
	}(tmu0, tkappa, tmu, tdata)

	kappaT0 := ConcatenateVertically(tkappa0, AddConstant(tkappa, 1.0))
	alphaT0 := ConcatenateVertically(talpha0, AddConstant(talpha, 0.5))
	betaT0 := func(beta0, beta, kappa, mu, data *mat.VecDense) *mat.VecDense {
		tmp := mat.NewVecDense(beta.Len(), nil)
		tmp.SubVec(data, mu)
		tmp = ElementWisePower(tmp, 2)
		tmp.MulElemVec(kappa,tmp)

		tmp0 := AddConstant(kappa, 1.0)
		tmp0 = MulConstant(tmp0, 2.0)

		tmp.DivElemVec(tmp, tmp0)

		tmp.AddVec(beta, tmp)

		return ConcatenateVertically(beta0, tmp)
	} (tbeta0, tbeta, tkappa, tmu, tdata)

	st.mu =TransformVecDenseToSlice(muT0)
	st.kappa = TransformVecDenseToSlice(kappaT0)
	st.alpha = TransformVecDenseToSlice(alphaT0)
	st.beta = TransformVecDenseToSlice(betaT0)

}


// transform the data in [][]float64 form to mat.Dense with data[i] as the i-th row
func TransformToMatDense(data [][]float64) *mat.Dense {
	rows := len(data)
	cols := len(data[0])
	result := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, data[i][j])
		}
	}
	return result
}


// ConcatenateVertically concatenates two vectors vertically and returns a *mat.VecDense.
func ConcatenateVertically(a, b *mat.VecDense) *mat.VecDense {
	rowsA, _ := a.Dims()
	rowsB, _ := b.Dims()

	// Create a new vector with the length equal to the sum of the lengths of a and b
	result := mat.NewVecDense(rowsA+rowsB, nil)

	// Copy the elements of vector a into the result vector
	for i := 0; i < rowsA; i++ {
		result.SetVec(i, a.AtVec(i))
	}

	// Copy the elements of vector b into the result vector
	for i := 0; i < rowsB; i++ {
		result.SetVec(rowsA+i, b.AtVec(i))
	}

	return result
}


// AddConstant adds a constant value to each element of a *mat.VecDense vector.
func AddConstant(v *mat.VecDense, c float64) *mat.VecDense {
	// Get the number of elements in the vector
	n, _ := v.Dims()

	// Create a new vector to hold the result
	result := mat.NewVecDense(n, nil)

	// Add the constant to each element
	for i := 0; i < n; i++ {
		result.SetVec(i, v.AtVec(i)+c)
	}

	return result
}

func MulConstant(v *mat.VecDense, c float64) *mat.VecDense {
	n, _ := v.Dims()
	result := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		result.SetVec(i, v.AtVec(i)*c)
	}
	return result
}


// ElementWisePower raises each element of the input vector to the given power.
func ElementWisePower(vec *mat.VecDense, power float64) *mat.VecDense {
	n := vec.Len()
	result := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		val := vec.AtVec(i)
		result.SetVec(i, math.Pow(val, power))
	}
	return result
}


// func to change mat.VecDense to []float64
// TransformVecDenseToSlice converts a *mat.VecDense to a slice of float64 values.
func TransformVecDenseToSlice(vec *mat.VecDense) []float64 {
	n := vec.Len()
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = vec.AtVec(i)
	}
	return result
}




// SubDense returns a submatrix from a mat.Dense based on the specified row and column indices.
func SubDense(m *mat.Dense, rowIdx, colIdx []int) *mat.Dense {
	subRows := len(rowIdx)
	subCols := len(colIdx)

	// Create a new Dense matrix to hold the submatrix
	subMatrix := mat.NewDense(subRows, subCols, nil)

	// Copy the elements from the original matrix to the submatrix
	for i, r := range rowIdx {
		for j, c := range colIdx {
			subMatrix.Set(i, j, m.At(r, c))
		}
	}

	return subMatrix
}

// ReplaceSubMatrix replaces a part of the matrix `m` with the matrix `subMatrix`
// starting at the specified row and column indices.
func ReplaceSubMatrix(m *mat.Dense, subMatrix *mat.Dense, startRow, startCol int) {
	// Get the dimensions of the submatrix
	subRows, subCols := subMatrix.Dims()

	// Ensure the submatrix fits within the original matrix at the specified position
	rows, cols := m.Dims()
	if startRow+subRows > rows || startCol+subCols > cols {
		panic("Submatrix does not fit within the bounds of the original matrix")
	}

	// Copy elements from subMatrix into the corresponding positions in m
	for i := 0; i < subRows; i++ {
		for j := 0; j < subCols; j++ {
			m.Set(startRow+i, startCol+j, subMatrix.At(i, j))
		}
	}
}


// GetRowVector returns a specific row as a *mat.VecDense from a *mat.Dense matrix.
func GetRowVector(m *mat.Dense, rowIndex int, colStart, colEnd int) *mat.VecDense {
	if colStart < 0 || colEnd > m.RawMatrix().Cols || rowIndex >= m.RawMatrix().Rows {
		panic("Index out of bounds")
	}

	vec := mat.NewVecDense(colEnd-colStart, nil)
	for i := colStart; i < colEnd; i++ {
		vec.SetVec(i-colStart, m.At(rowIndex, i))
	}
	return vec
}

// GetColVector returns a specific column as a *mat.VecDense from a *mat.Dense matrix.
func GetColVector(m *mat.Dense, colIndex int, rowStart, rowEnd int) *mat.VecDense {
	if rowStart < 0 || rowEnd > m.RawMatrix().Rows || colIndex >= m.RawMatrix().Cols {
		panic("Index out of bounds")
	}

	// Slice the column from rowStart to rowEnd
	vec := mat.NewVecDense(rowEnd-rowStart, nil)
	for i := rowStart; i < rowEnd; i++ {
		vec.SetVec(i-rowStart, m.At(i, colIndex))
	}
	return vec
}

