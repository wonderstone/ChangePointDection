package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/wonderstone/change-point-detection/cpd"
)

func main() {

	// Step 1: generate the data
	num := 5
	minl := 50
	maxl := 1000
	seed := 100

	partition, data := cpd.GenerateNormalTimeSeries(num, minl, maxl, int64(seed))
	// check if the data_input.csv file is there, if yes then delete it
	if _, err := os.Stat("data_input.csv"); err == nil {
		os.Remove("data_input.csv")
	}

	WriteData(data, "data_input.csv")


	fmt.Println(partition)
	fmt.Println(data)

	// Step 2: initialize the parameters part
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
	c := cpd.NewOCPD(250, cpd.ConstantHazardSlice, st)

	// Step 3: update the data
	for _, x := range data {
		// fmt.Println(t, x)
		c.OCPD_Update(x)
	}
	fmt.Println(c.Maxes)
	fmt.Println("Done")

	// Step 4 output the c.Maxes to the csv file use csv writer
	// check if the data_output.csv file is there, if yes then delete it
	if _, err := os.Stat("data_output.csv"); err == nil {
		os.Remove("data_output.csv")
	}
	WriteData(c.Maxes, "data_output.csv")
}


// func to write the slice of float64 to the csv file
func WriteData(data []float64, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
	for _, value := range data {
		err := writer.Write([]string{fmt.Sprintf("%f", value)})
		if err != nil {
			log.Fatal(err)
		}
	}
}