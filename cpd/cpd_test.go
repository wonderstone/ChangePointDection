package cpd

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// test the OnlineChangepointDetection
func TestOnlineChangepointDetection(t *testing.T) {
	// read the data from the ./data_output.csv
	// the data is generated from partition 
	// [58, 74, 117, 153, 137, 129, 188]
	// so, the change point is 59, 75, 118, 154, 138, 130, 189
	data := ReadData("../data/data_output.csv")

	// initialize the parameters part
	t_alpha := []float64{0.1}
	t_beta := []float64{0.01}
	t_kappa := []float64{1}
	t_mu := []float64{0}

	st := NewStudentT_BU(
		t_alpha,
		t_beta,
		t_kappa,
		t_mu,
	)
	R, maxes := OnlineChangepointDetection(data,250, ConstantHazard, st)
	tmp := R.At(5, 600)
	assert.InDelta(t,3.253488142937348e-08,tmp, 0.0001)
	assert.Equal(t, 1.0, maxes[59])
	
}


// test the OCPD
func TestOCPD(t *testing.T) {
	// read the data from the ./data_output.csv
	// the data is generated from partition 
	// [58, 74, 117, 153, 137, 129, 188]
	// so, the change point is 59, 75, 118, 154, 138, 130, 189
	data := ReadData("../data/data_output.csv")
	// initialize the parameters part
	t_alpha := []float64{0.1}
	t_beta := []float64{0.01}
	t_kappa := []float64{1}
	t_mu := []float64{0}

	st := NewStudentT_BU(
		t_alpha,
		t_beta,
		t_kappa,
		t_mu,
	)
	cpd := NewOCPD(250, ConstantHazardSlice, st)
	for t, x := range data {
		fmt.Println(t,x)
		cpd.OCPD_Update(x)
		fmt.Println(cpd.Res)
		fmt.Println(cpd.Maxes)
		fmt.Println("XXXXXXXXXXXXX")
	}
	fmt.Println(cpd)
	


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

