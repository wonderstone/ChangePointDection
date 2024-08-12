package cpd

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func OnlineChangepointDetection(
	data []float64, 
	hazardFunction func(float64, *mat.Dense) *mat.Dense, 
	logLikelihoodClass *StudentT) (mat.Dense, []float64) {
    // Use online bayesian changepoint detection
    // https://scientya.com/bayesian-online-change-point-detection-an-intuitive-understanding-b2d2b9dc165b

    // Parameters:
    // data    -- the time series data

    // Outputs:
    //     R  -- is the probability at time step t that the last sequence is already s time steps long
    //     maxes -- the argmax on column axis of matrix R (growth probability value) for each time step

	
	// maxes = np.zeros(len(data) + 1)
	maxes := make([]float64, len(data)+1)
	// R = np.zeros((len(data) + 1, len(data) + 1))
    // R[0, 0] = 1
	R := mat.NewDense(len(data)+1, len(data)+1, nil)
	R.Set(0, 0, 1)
	
	for t, x := range data {
		predprobs := logLikelihoodClass.PDF([]float64{x})


		mt := mat.NewDense(t+1, 1, nil)
		H := hazardFunction(250,mt)
		HVec := ChangeDenseToVecDense(H)
		
		predprobsvec := GetVectorFrom2dInnerSlice(predprobs,0)
		fmt.Println(t, predprobsvec)
		Rsub := GetColVector(R,t,0,t+1)
		rsm := mat.NewVecDense(t+1, nil)
		rsm.MulElemVec(Rsub,predprobsvec)
		rsm.MulElemVec(rsm,AddConstant(MulConstant(HVec,-1),1))

		// tmpVec := GetColVector(R,t,0,t+1)
		// tmpVec.MulElemVec(tmpVec,predprobsmat)
		// tmpVec.MulElemVec(tmpVec,)


		logLikelihoodClass.UpdateTheta([]float64{x})

	}

	return *R, maxes
}




func GetVectorFrom2dInnerSlice(slice [][]float64, inner int) *mat.VecDense {
	res := mat.NewVecDense(len(slice), nil)
	for  i, v := range slice {
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