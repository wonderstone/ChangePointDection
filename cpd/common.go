package cpd

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// + Compared to the Numpy package in Python, the Gonum.mat package in Go is really weak.
// + Some functions are not implemented in the Gonum.mat package.
// + The package can run for this moment, but not the best implementation.
// todo: Maybe improve the implementation in the future but no guarantee.

// * ConstantHazard computes the hazard function for Bayesian online learning.
// * Arguments:
//   - lam - initial probability
//   - r - R matrix
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

// * Define the StudentT_Bayesian_Update struct
type StudentT_Bayesian_Update struct {
	alpha, beta, kappa, mu     []float64
	alpha0, beta0, kappa0, mu0 []float64
}

// * NewStudentT_BU creates a new StudentT_Bayesian_Update struct with the given parameters.
// * Use copy to avoid interference between the two parameter sets.
func NewStudentT_BU(t_alpha, t_beta, t_kappa, t_mu []float64) *StudentT_Bayesian_Update {
	st := &StudentT_Bayesian_Update{
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

// * Method: PDF computes the probability density function
// *   of the Student's t-distribution for the given data.
// *  the output is a 2D slice
// * the first dimension corresponding to the parameters set slice, for alpha, beta, kappa, mu
// * the second dimension corresponding to the data slice，most of the time just one element
func (st *StudentT_Bayesian_Update) PDF(data []float64) [][]float64 {
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

// * Method: UpdateTheta updates the parameters of the Student's t-distribution
func (st *StudentT_Bayesian_Update) UpdateTheta(data []float64) {
	// @ 1. check all the parameters are the same length
	par_len := len(st.alpha)
	if par_len != len(st.beta) || par_len != len(st.kappa) || par_len != len(st.mu) {
		panic("Parameters alpha, beta, kappa, and mu must have the same length")
	}
	// @ 2. copy the data to avoid interference
	tmpdata := make([]float64, len(data))
	copy(tmpdata, data)

	// @ 3. make the data length the same as the parameters
	// @    to use the mat.VecDense operation
	if len(tmpdata) < len(st.alpha) {
		lastElement := tmpdata[len(tmpdata)-1]
		for len(tmpdata) < len(st.alpha) {
			tmpdata = append(tmpdata, lastElement)
		}
	}
	// @ 4. change data into mat.vecDense format
	// @  change data into mat.vecDense format with the same length as the parameters
	tdata := mat.NewVecDense(par_len, tmpdata)
	// @  change the parameters into mat.vecDense format
	talpha := mat.NewVecDense(par_len, st.alpha)
	tbeta := mat.NewVecDense(par_len, st.beta)
	tkappa := mat.NewVecDense(par_len, st.kappa)
	tmu := mat.NewVecDense(par_len, st.mu)
	// @  change the parameters(0 means init set) into mat.vecDense format
	talpha0 := mat.NewVecDense(len(st.alpha0), st.alpha0)
	tbeta0 := mat.NewVecDense(len(st.beta0), st.beta0)
	tkappa0 := mat.NewVecDense(len(st.kappa0), st.kappa0)
	tmu0 := mat.NewVecDense(len(st.mu0), st.mu0)
	// @ 5. update the parameters
	// @ update the mu0 part
	// @ tempmu0 = concatenate((mu0, (kappa * mu + data) / (kappa + 1)))
	muT0 := func(mu0, kappa, mu, data *mat.VecDense) *mat.VecDense {
		tmp_numerator := mat.NewVecDense(kappa.Len(), nil)
		tmp_numerator.MulElemVec(kappa, mu)
		tmp_numerator.AddVec(tmp_numerator, data)
		tmp_denominator := AddConstant(kappa, 1.0)
		tmp_numerator.DivElemVec(tmp_numerator, tmp_denominator)
		return ConcatenateVertically(mu0, tmp_numerator)
	}(tmu0, tkappa, tmu, tdata)
	// @ update the kappa0 part
	// @ tempkappa0 = concatenate((kappa0, kappa + 1))
	kappaT0 := ConcatenateVertically(tkappa0, AddConstant(tkappa, 1.0))
	// @ update the alpha0 part
	// @ tempalpha0 = concatenate((alpha0, alpha + 0.5))
	alphaT0 := ConcatenateVertically(talpha0, AddConstant(talpha, 0.5))
	// @ update the beta0 part
	// @ tempbeta0 = concatenate((beta0, beta + (kappa * (data - mu) ** 2) / (2.0 * (kappa + 1.0))
	betaT0 := func(beta0, beta, kappa, mu, data *mat.VecDense) *mat.VecDense {
		tmp_numerator := mat.NewVecDense(beta.Len(), nil)
		tmp_numerator.SubVec(data, mu)
		tmp_numerator = PowConstant(tmp_numerator, 2)
		tmp_numerator.MulElemVec(kappa, tmp_numerator)

		tmp_denominator := AddConstant(kappa, 1.0)
		tmp_denominator = MulConstant(tmp_denominator, 2.0)

		tmp_numerator.DivElemVec(tmp_numerator, tmp_denominator)
		tmp_numerator.AddVec(beta, tmp_numerator)

		return ConcatenateVertically(beta0, tmp_numerator)
	}(tbeta0, tbeta, tkappa, tmu, tdata)

	st.mu = TransformVecDenseToSlice(muT0)
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

	// @ Create a new vector with the length equal to the sum of the lengths of a and b
	result := mat.NewVecDense(rowsA+rowsB, nil)

	// @ Copy the elements of vector a into the result vector
	for i := 0; i < rowsA; i++ {
		result.SetVec(i, a.AtVec(i))
	}
	// @ Copy the elements of vector b into the result vector
	for i := 0; i < rowsB; i++ {
		result.SetVec(rowsA+i, b.AtVec(i))
	}

	return result
}


// - ADD 3 Constant related function to VecDense element-wise operation
// AddConstant.
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
// MulConstant
func MulConstant(v *mat.VecDense, c float64) *mat.VecDense {
	n, _ := v.Dims()
	result := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		result.SetVec(i, v.AtVec(i)*c)
	}
	return result
}

// PowConstant 
func PowConstant(vec *mat.VecDense, power float64) *mat.VecDense {
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

// func to transform a vecDense to mat.Dense
// TransformVecDenseToMatDense converts a *mat.VecDense to a mat.Dense with the vector as a single row.
func TransformVecDenseToMatDense(vec *mat.VecDense, RowTag bool) *mat.Dense {
	if RowTag {
		// if RowTag is true, then the vector is transformed to 1*n matrix
		n := vec.Len()
		result := mat.NewDense(1, n, nil)
		for i := 0; i < n; i++ {
			result.Set(0, i, vec.AtVec(i))
		}
		return result
	} else {
		// if RowTag is false, then the vector is transformed to n*1 matrix
		n := vec.Len()
		result := mat.NewDense(n, 1, nil)
		for i := 0; i < n; i++ {
			result.Set(i, 0, vec.AtVec(i))
		}
		return result
	}
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
