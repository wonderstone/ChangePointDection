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

	// lam - inital prob for hazard function
	lam := 250.0
	
	// maxes = np.zeros(len(data) + 1)
	maxes := make([]float64, len(data)+1)
	// R = np.zeros((len(data) + 1, len(data) + 1))
    // R[0, 0] = 1
	R := mat.NewDense(len(data)+1, len(data)+1, nil)
	R.Set(0, 0, 1)
	
	for t, x := range data {
		predprobs := logLikelihoodClass.PDF([]float64{x})


		mt := mat.NewDense(t+1, 1, nil)
		H := hazardFunction(lam,mt)
		HVec := ChangeDenseToVecDense(H)
		
		predprobsvec := GetVectorFrom2dInnerSlice(predprobs,0)
		fmt.Println(t, predprobsvec)
		Rsub := GetColVector(R,t,0,t+1)
		rsm := mat.NewVecDense(t+1, nil)
		rsm.MulElemVec(Rsub,predprobsvec)
		rsm.MulElemVec(rsm,AddConstant(MulConstant(HVec,-1),1))
		
		
		// 1.
		ReplaceSubMatrix(R,TransformVecDenseToMatDense(rsm,false),1,t+1)
		// 2.
		tmpres:= mat.NewVecDense(t+1, nil)
		tmpres.MulElemVec(Rsub,predprobsvec)
		tmpres.MulElemVec(tmpres,HVec)
		
		
		tmpVal := mat.Sum(tmpres)
		R.Set(0,t+1,tmpVal)



		// 3. get the R matrix t+1 column as a vecDense

		
		// get the instance where the pointer R is pointing to

		tmpVec := GetColVector(R,t+1,0,len(data)+1)
		


		tmpsum := mat.Sum(tmpVec)

		tmpColVectDense := mat.NewVecDense(tmpVec.Len(), tmpVec.RawVector().Data)

		// Scale the tmpVec by 1/tmpsum
		tmpColVectDense.ScaleVec(1/tmpsum, tmpColVectDense)


		// 4.
		ReplaceSubMatrix(R,TransformVecDenseToMatDense(tmpVec,false),0,t+1)


		// tmpVec := GetColVector(R,t,0,t+1)
		// tmpVec.MulElemVec(tmpVec,predprobsmat)
		// tmpVec.MulElemVec(tmpVec,)


		logLikelihoodClass.UpdateTheta([]float64{x})
		maxes[t] = float64(ArgmaxVecDense(GetColVector(R,t,0,t+1)))

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