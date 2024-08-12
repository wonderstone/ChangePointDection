package cpd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// test the OnlineChangepointDetection
func TestOnlineChangepointDetection(t *testing.T) {
	data := []float64{0.5, 0.3, 0.8}

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
	R, maxes := OnlineChangepointDetection(data, ConstantHazard, st)
	assert.Equal(t, 0, R.At(0, 0))
	assert.Equal(t, 1, len(maxes))
}



// test the GetVectorFrom2dInnerSlice
func TestGetVectorFrom2dInnerSlice(t *testing.T) {
	data := [][]float64{
		{0.5, 0.3},
		{0.8, 0.6},
	}
	result := GetVectorFrom2dInnerSlice(data,0)
	assert.Equal(t, 0.5, result.AtVec(0))
	assert.Equal(t, 0.8, result.AtVec(1))


}