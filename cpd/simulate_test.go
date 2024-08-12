package cpd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenerateNormalTimeSeries(t *testing.T) {
	num := 5
	minl := 50
	maxl := 1000
	seed := 100

	partition, data := generateNormalTimeSeries(num, minl, maxl, int64(seed))
	assert.Equal(t, 5, len(partition))
	tmpsum := 0
	for _, v := range partition {
		tmpsum += v
	}
	assert.Equal(t, tmpsum, len(data))
}