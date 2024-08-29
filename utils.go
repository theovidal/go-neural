package main

func parseHolds(holds []byte) (result []float64) {
	var data [11][18]float64
	for i := 0; i < len(holds)/2; i++ {
		holdIndex := holds[i*2] & 0xFF
		// hold type
		_ = holds[i*2+1] & 0xFF

		x := holdIndex % 11
		y := holdIndex / 11

		data[x][y] = 1.0
	}

	for _, xrow := range data {
		for _, y := range xrow {
			result = append(result, y)
		}
	}
	return result
}

type problem struct {
	Holds []byte
	Grade int
}
