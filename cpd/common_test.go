package cpd

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"gonum.org/v1/gonum/mat"
)

func TestConstantHazard(t *testing.T) {
	lam := 0.5
	r := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	result := ConstantHazard(lam, r)
	rRows, rCols := result.Dims()
	for i := 0; i < rRows; i++ {
		for j := 0; j < rCols; j++ {
			if result.At(i, j) != 2 {
				t.Errorf("Expected 2, but got %f", result.At(i, j))
			}
		}
	}
}

func TestPDF(t *testing.T) {

	data := []float64{0.5, 0.3, 0.8}

	// try copy data to another variable
	copiedData := make([]float64, len(data))
	copy(copiedData, data)

	// check if the copiedData is the same as data
	assert.Equal(t, data, copiedData)

	// initialize the parameters part
	t_alpha := []float64{0.1}
	t_beta := []float64{0.1}
	t_kappa := []float64{1}
	t_mu := []float64{0}

	st := NewStudentT(
		t_alpha,
		t_beta,
		t_kappa,
		t_mu,
	)
	// change the parameters, check if the instance is not changed
	t_alpha[0] = 0.2

	if st.alpha[0] != 0.1 {
		t.Errorf("Expected 0.1, but got %f", st.alpha[0])
	}

	pdfs := st.PDF(data)
	// check pdfs[0][0] should be 0.10435016543007986
	// use assert indelta to compare float64

	assert.InDelta(t, 0.10435016543007986, pdfs[0][0], 0.00000000001)

	// check the ConcatenateVertically function
	a := mat.NewVecDense(3, []float64{1, 2, 3})
	b := mat.NewVecDense(3, []float64{4, 5, 6})
	// change the a into Dense
	c := ConcatenateVertically(a, b)

	fmt.Println(c)




	st.UpdateTheta([]float64{0.2})
	st.UpdateTheta([]float64{0.3})
	fmt.Println(st.mu)

}

// func TestUpdateTheta(t *testing.T){
// 	data:= []float64{0.2, 0.1, 0.4}
// 	st := NewStudentT(0.1, 0.1, 1, 0)

// 	st.UpdateTheta(data)

// 	if st.mu != 0.5333333333333333 {
// 		t.Errorf("Expected 0.5333333333333333, but got %f", st.mu)
// 	}

// 	if st.kappa != 4 {
// 		t.Errorf("Expected 4, but got %f", st.kappa)
// 	}

// 	if st.alpha != 2.5 {
// 		t.Errorf("Expected 2.5, but got %f", st.alpha)
// 	}

// 	if st.beta != 0.013333333333333334 {
// 		t.Errorf("Expected 0.013333333333333334, but got %f", st.beta)
// 	}

// }
