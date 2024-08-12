package cpd

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func OnlineChangepointDetection(
	data []float64,
	lam float64,
	hazardFunction func(float64, *mat.Dense) *mat.Dense,
	logLikelihoodClass *StudentT_Bayesian_Update) (mat.Dense, []float64) {

	// Parameters:
	// data    -- the time series data
	// lam	 -- the inital probability for hazard function, 250 in the paper

	// Outputs:
	//   R  -- is the probability at time step t that the last sequence is already s time steps long
	//   maxes -- the argmax on column axis of matrix R (growth probability value) for each time step

	maxes := make([]float64, len(data)+1)

	R := mat.NewDense(len(data)+1, len(data)+1, nil)
	R.Set(0, 0, 1)

	for t, x := range data {

		// @ 1. Evaluate the predictive distribution for the new datum under each of
		// @ the parameters.  This is the standard thing from Bayesian inference.
		predprobs := logLikelihoodClass.PDF([]float64{x})
		// @ change to VecDense
		predprobsvec := GetVectorFrom2dInnerSlice(predprobs, 0)

		// @ 2. Evaluate the hazard function for this interval
		mt := mat.NewDense(t+1, 1, nil)
		H := hazardFunction(lam, mt)
		// @ change to VecDense
		HVec := ChangeDenseToVecDense(H)

		// @ 3. Evaluate the growth probabilities
		// @ R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)
		Rsub := GetColVector(R, t, 0, t+1)
		rsm := mat.NewVecDense(t+1, nil)
		rsm.MulElemVec(Rsub, predprobsvec)
		rsm.MulElemVec(rsm, AddConstant(MulConstant(HVec, -1), 1))
		// @ update the R matrix
		ReplaceSubMatrix(R, TransformVecDenseToMatDense(rsm, false), 1, t+1)

		// @ 4. Evaluate the probability of changepoint, which is the sum of the growth probabilities
		// @    R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)
		tmpres := mat.NewVecDense(t+1, nil)
		tmpres.MulElemVec(Rsub, predprobsvec)
		tmpres.MulElemVec(tmpres, HVec)
		tmpVal := mat.Sum(tmpres)
		// @ update the R matrix
		R.Set(0, t+1, tmpVal)

		// @ 5. Normalize the run length probabilities
		// @ R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])
		nRow, _ := R.Dims()
		tmpVec := GetColVector(R, t+1, 0, nRow)
		tmpsum := mat.Sum(tmpVec)
		// @ change to VecDense
		tmpColVectDense := mat.NewVecDense(tmpVec.Len(), tmpVec.RawVector().Data)
		// @ scale the vector
		tmpColVectDense.ScaleVec(1/tmpsum, tmpColVectDense)
		// @ update the R matrix
		ReplaceSubMatrix(R, TransformVecDenseToMatDense(tmpVec, false), 0, t+1)

		// @ 6. Update the parameter set for Distribution
		logLikelihoodClass.UpdateTheta([]float64{x})
		// @ 7. Store the maximum value of the growth probabilities
		maxes[t] = float64(ArgmaxVecDense(GetColVector(R, t, 0, t+1)))
	}
	return *R, maxes
}

func GetVectorFrom2dInnerSlice(slice [][]float64, inner int) *mat.VecDense {
	res := mat.NewVecDense(len(slice), nil)
	for i, v := range slice {
		res.SetVec(i, v[inner])
	}
	return res
}

// func change dense to vecdense
func ChangeDenseToVecDense(dense *mat.Dense) *mat.VecDense {
	rows, _ := dense.Dims()
	data := make([]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = dense.At(i, 0)
	}
	return mat.NewVecDense(rows, data)
}

// func to get argmax of a vector
// ArgmaxVecDense returns the index of the maximum value in a vector
func ArgmaxVecDense(vec *mat.VecDense) int {
	maxVal := vec.AtVec(0)
	maxIdx := 0
	for i := 1; i < vec.Len(); i++ {
		if val := vec.AtVec(i); val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

// func to readData from the file
func ReadData(filename string) []float64 {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	var data []float64
	for _, line := range lines {
		val, err := strconv.ParseFloat(line[0], 64)
		if err != nil {
			log.Fatal(err)
		}
		data = append(data, val)
	}
	return data
}
