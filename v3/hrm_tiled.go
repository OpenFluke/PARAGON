package paragon

import (
	"math"
	"math/rand"
)

// HRM represents a Hierarchical Reasoning Model
type HRM struct {
	// Core components
	InputNet  *Dense
	LowNet    *Dense
	HighNet   *Dense
	OutputNet *Dense
	// State vectors
	StateL []float32 // Low-level state
	StateH []float32 // High-level state
	// Hyperparameters
	InputSize    int
	HiddenSize   int
	OutputSize   int
	LowSteps     int // T in paper - steps per cycle
	HighCycles   int // N in paper - number of high-level cycles
	LearningRate float64
}

// Dense represents a simple dense layer
type Dense struct {
	Weights    [][]float32
	Biases     []float32
	InputSize  int
	OutputSize int
}

// NewDense creates a new dense layer
func NewDense(inputSize, outputSize int) *Dense {
	weights := make([][]float32, outputSize)
	biases := make([]float32, outputSize)
	// Xavier/Glorot initialization
	scale := float32(math.Sqrt(2.0 / float64(inputSize+outputSize)))
	for i := 0; i < outputSize; i++ {
		weights[i] = make([]float32, inputSize)
		for j := 0; j < inputSize; j++ {
			weights[i][j] = float32(rand.NormFloat64()) * scale
		}
		biases[i] = float32(rand.NormFloat64()) * scale * 0.1
	}
	return &Dense{
		Weights:    weights,
		Biases:     biases,
		InputSize:  inputSize,
		OutputSize: outputSize,
	}
}

// Forward performs forward pass through dense layer
func (d *Dense) Forward(input []float32) []float32 {
	if len(input) != d.InputSize {
		padded := make([]float32, d.InputSize)
		for i := 0; i < d.InputSize && i < len(input); i++ {
			padded[i] = input[i]
		}
		input = padded
	}
	output := make([]float32, d.OutputSize)
	for i := 0; i < d.OutputSize; i++ {
		sum := d.Biases[i]
		for j := 0; j < d.InputSize; j++ {
			sum += d.Weights[i][j] * input[j]
		}
		// ReLU activation
		if sum > 0 {
			output[i] = sum
		} else {
			output[i] = sum * 0.01 // Leaky ReLU
		}
	}
	return output
}

// NewHRM creates a new Hierarchical Reasoning Model
func NewHRM(inputSize, hiddenSize, outputSize int) *HRM {
	return &HRM{
		InputNet:     NewDense(inputSize, hiddenSize),
		LowNet:       NewDense(hiddenSize*2, hiddenSize), // Takes L state + H state
		HighNet:      NewDense(hiddenSize*2, hiddenSize), // Takes H state + L state
		OutputNet:    NewDense(hiddenSize, outputSize),
		StateL:       make([]float32, hiddenSize),
		StateH:       make([]float32, hiddenSize),
		InputSize:    inputSize,
		HiddenSize:   hiddenSize,
		OutputSize:   outputSize,
		LowSteps:     5, // Default, adjustable via SetArchitecture
		HighCycles:   4, // Default, adjustable via SetArchitecture
		LearningRate: 0.001,
	}
}

// initializeStates initializes hidden states
func (h *HRM) initializeStates() {
	for i := range h.StateL {
		h.StateL[i] = float32(rand.NormFloat64() * 0.01)
	}
	for i := range h.StateH {
		h.StateH[i] = float32(rand.NormFloat64() * 0.01)
	}
}

// Forward performs the complete HRM forward pass
func (h *HRM) Forward(input [][]float64) [][]float64 {
	inputVec := h.flattenInput(input)
	if h.isStateEmpty() {
		h.initializeStates()
	}
	// Input embedding
	inputState := h.InputNet.Forward(inputVec)
	for i := range h.StateL {
		h.StateL[i] += inputState[i] * 0.1 // Weak initialization
	}
	// Main HRM computation loop
	for cycle := 0; cycle < h.HighCycles; cycle++ {
		currentH := make([]float32, len(h.StateH))
		copy(currentH, h.StateH)
		for step := 0; step < h.LowSteps; step++ {
			combinedInput := make([]float32, len(h.StateL)+len(currentH))
			copy(combinedInput, h.StateL)
			copy(combinedInput[len(h.StateL):], currentH)
			newStateL := h.LowNet.Forward(combinedInput)
			for i := range h.StateL {
				h.StateL[i] = 0.7*h.StateL[i] + 0.3*newStateL[i] // Residual
				if math.IsNaN(float64(h.StateL[i])) {
					h.StateL[i] = 0 // Reset NaN
				}
			}
		}
		combinedHInput := make([]float32, len(h.StateH)+len(h.StateL))
		copy(combinedHInput, h.StateH)
		copy(combinedHInput[len(h.StateH):], h.StateL)
		newStateH := h.HighNet.Forward(combinedHInput)
		for i := range h.StateH {
			h.StateH[i] = 0.8*h.StateH[i] + 0.2*newStateH[i] // Residual
			if math.IsNaN(float64(h.StateH[i])) {
				h.StateH[i] = 0 // Reset NaN
			}
		}
	}
	output := h.OutputNet.Forward(h.StateH)
	return h.unflattenOutput(output)
}

// Train performs one training step
func (h *HRM) Train(input, target [][]float64) float64 {
	output := h.Forward(input)
	loss := h.calculateLoss(output, target)
	if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
		h.updateWeights(input, target, output, loss)
	}
	return loss
}

// calculateLoss computes MSE loss
func (h *HRM) calculateLoss(output, target [][]float64) float64 {
	loss := 0.0
	count := 0
	for i := 0; i < len(output) && i < len(target); i++ {
		for j := 0; j < len(output[i]) && j < len(target[i]); j++ {
			diff := float64(output[i][j] - target[i][j])
			loss += diff * diff
			count++
		}
	}
	if count > 0 {
		loss /= float64(count)
	} else {
		loss = 1.0 // Default loss
	}
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 1.0 // Cap loss
	}
	return loss
}

// updateWeights performs simplified weight updates
func (h *HRM) updateWeights(input, target, output [][]float64, loss float64) {
	gradient := float32(math.Min(1.0, loss)) * float32(h.LearningRate)
	// Update OutputNet
	for i := range h.OutputNet.Biases {
		update := gradient * 0.1
		h.OutputNet.Biases[i] -= update
		for j := 0; j < h.HiddenSize && j < len(h.OutputNet.Weights[i]); j++ { // Use HiddenSize for OutputNet
			h.OutputNet.Weights[i][j] -= gradient * float32(h.StateH[j]) * 0.001
		}
	}
	// Update HighNet
	for i := range h.HighNet.Biases {
		update := gradient * 0.05
		h.HighNet.Biases[i] -= update
		for j := 0; j < h.HiddenSize && j < len(h.HighNet.Weights[i]); j++ {
			h.HighNet.Weights[i][j] -= gradient * float32(h.StateL[j%len(h.StateL)]) * 0.001
		}
	}
	// Update LowNet
	for i := range h.LowNet.Biases {
		update := gradient * 0.02
		h.LowNet.Biases[i] -= update
		for j := 0; j < h.HiddenSize && j < len(h.LowNet.Weights[i]); j++ {
			h.LowNet.Weights[i][j] -= gradient * float32(h.StateH[j%len(h.StateH)]) * 0.001
		}
	}
}

// Helper functions
func (h *HRM) flattenInput(input [][]float64) []float32 {
	result := make([]float32, h.InputSize)
	idx := 0
	for i := 0; i < len(input) && idx < h.InputSize; i++ {
		for j := 0; j < len(input[i]) && idx < h.InputSize; j++ {
			result[idx] = float32(input[i][j])
			idx++
		}
	}
	return result
}

func (h *HRM) unflattenOutput(output []float32) [][]float64 {
	result := make([][]float64, 1)
	result[0] = make([]float64, len(output))
	for i := range output {
		result[0][i] = float64(output[i])
	}
	return result
}

func (h *HRM) isStateEmpty() bool {
	for _, val := range h.StateL {
		if val != 0 {
			return false
		}
	}
	for _, val := range h.StateH {
		if val != 0 {
			return false
		}
	}
	return true
}

// SetLearningRate adjusts the learning rate
func (h *HRM) SetLearningRate(lr float64) {
	h.LearningRate = lr
}

// SetArchitecture allows adjusting the recurrent structure
func (h *HRM) SetArchitecture(lowSteps, highCycles int) {
	h.LowSteps = lowSteps
	h.HighCycles = highCycles
}
