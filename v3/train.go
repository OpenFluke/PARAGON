package paragon

import (
	"fmt"
	"math"
	"math/rand"
)

// In the Forward method in webgpu_optimized.go, ensure initialization:
func (n *Network[T]) Forward(inputs [][]float64) {
	// Initialize optimized GPU if not already done

	// Fallback to existing implementation
	n.forwardCPU(inputs)
}

// Backward method with GPU support
func (n *Network[T]) Backward(targets [][]float64, lr float64, clipUpper, clipLower T) {

	n.backwardCPU(targets, lr, clipUpper, clipLower)
}

// ---------------------------------------------------------------------------
// Forward pass with optional layer‑replay
// ---------------------------------------------------------------------------
func (n *Network[T]) forwardCPU(inputs [][]float64) {
	in := n.Layers[n.InputLayer]

	if len(inputs) != in.Height || len(inputs[0]) != in.Width {
		panic(fmt.Sprintf("input mismatch: want %dx%d, got %dx%d",
			in.Height, in.Width, len(inputs), len(inputs[0])))
	}

	// Convert float64 input into T-typed neurons
	for y := 0; y < in.Height; y++ {
		for x := 0; x < in.Width; x++ {
			in.Neurons[y][x].Value = T(inputs[y][x])
		}
	}

	for l := 1; l < len(n.Layers); l++ {
		layer := &n.Layers[l]

		// 🚀 Main forward pass
		n.forwardLayer(l, false)
		layer.CachedOutputs = CastFloat64SliceToT[T](layer.GetOutputValues())

	}

	// Final softmax pass
	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// ---------------------------------------------------------------------------
// Back‑prop with optional layer‑replay  (incl. attention weight update)
// ---------------------------------------------------------------------------
func (n *Network[T]) backwardCPU(
	targets [][]float64,
	lr float64,
	clipUpper T,
	clipLower T,
) {
	nLayers := len(n.Layers)

	// Allocate error tensor using T
	err := make([][][]T, nLayers)
	for l := range err {
		err[l] = make([][]T, n.Layers[l].Height)
		for y := range err[l] {
			err[l][y] = make([]T, n.Layers[l].Width)
		}
	}

	// Compute output error
	out := n.Layers[n.OutputLayer]
	for y := 0; y < out.Height; y++ {
		for x := 0; x < out.Width; x++ {
			neuron := out.Neurons[y][x]
			pred := float64(any(neuron.Value).(T))
			targ := targets[y][x]
			diff := targ - pred

			// Derivative is computed in T-space
			grad := ActivationDerivativeGeneric(neuron.Value, neuron.Activation)
			err[n.OutputLayer][y][x] = T(diff) * grad
		}
	}

	for l := n.OutputLayer; l > 0; l-- {

		// 🔁 Normal backprop
		n.backwardLayer(l, err, float64(lr), clipUpper, clipLower, false)

	}
}

// Train runs the training loop
func (n *Network[T]) Train(
	inputs [][][]float64,
	targets [][][]float64,
	epochs int,
	learningRate float64,
	earlyStopOnNegativeLoss bool,
	clipUpper T,
	clipLower T,
) {

	// CPU training path
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		perm := rand.Perm(len(inputs))
		shuffledInputs := make([][][]float64, len(inputs))
		shuffledTargets := make([][][]float64, len(targets))
		for i, p := range perm {
			shuffledInputs[i] = inputs[p]
			shuffledTargets[i] = targets[p]
		}

		for b := 0; b < len(shuffledInputs); b++ {
			n.Forward(shuffledInputs[b])
			loss := n.ComputeLoss(shuffledTargets[b])
			if math.IsNaN(loss) {
				fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, epoch)
				continue
			}
			if earlyStopOnNegativeLoss && loss < 0 {
				fmt.Printf("⚠️ Negative loss (%.4f) detected at sample %d, epoch %d. Stopping training early.\n", loss, b, epoch)
				return
			}
			totalLoss += loss
			n.Backward(shuffledTargets[b], learningRate, clipUpper, clipLower)
		}

		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}
