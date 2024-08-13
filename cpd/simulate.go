package cpd

import (
	"math"
	"math/rand"
)

// GenerateNormalTimeSeries generates a time series of normally distributed data.
func GenerateNormalTimeSeries(num, minl, maxl int, seed int64) ([]int, []float64) {
	source := rand.NewSource(seed)
	rng := rand.New(source)

	var data []float64
	partition := make([]int, num)

	for i := 0; i < num; i++ {
		partition[i] = rng.Intn(maxl-minl) + minl
	}

	for _, p := range partition {
		mean := rand.NormFloat64() * 10
		variance := math.Abs(rand.NormFloat64())
		for j := 0; j < p; j++ {
			tdata := rand.NormFloat64()*variance + mean
			data = append(data, tdata)
		}
	}

	return partition, data
}
