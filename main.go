package main

import (
	"fmt"

	"github.com/wonderstone/change-point-detection/cpd"
)

func main() {
	//Step 1 : Read the data from the ./data/data_output.csv
	data := cpd.ReadData("./data/data_output.csv")

	// initialize the parameters part
	t_alpha := []float64{0.1}
	t_beta := []float64{0.01}
	t_kappa := []float64{1}
	t_mu := []float64{0}

	st := cpd.NewStudentT_BU(
		t_alpha,
		t_beta,
		t_kappa,
		t_mu,
	)
	R, maxes := cpd.OnlineChangepointDetection(data,250, cpd.ConstantHazard, st)
	fmt.Println(R)
	fmt.Println(maxes)

	

}