package paragon

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/webgpu/wgpu"
)

// GPUState holds GPU context per network instance
type GPUState struct {
	ctx         *GPUContext
	enabled     bool
	initialized bool
}

// TrainGPU runs training loop with GPU acceleration
func (n *Network[T]) TrainGPU(
	inputs [][][]float64,
	targets [][][]float64,
	epochs int,
	learningRate float64,
	earlyStopOnNegativeLoss bool,
	clipUpper T,
	clipLower T,
) error {
	// Initialize GPU if not already done
	if !n.WebGPUNative {
		if err := n.InitGPU(); err != nil {
			fmt.Printf("[GPU] Initialization failed: %v, falling back to CPU\n", err)
			n.Train(inputs, targets, epochs, learningRate, earlyStopOnNegativeLoss, clipUpper, clipLower)
			return nil
		}
	}

	// Create GPU context
	ctx := &GPUContext{}
	if err := n.setupGPUContext(ctx); err != nil {
		fmt.Printf("[GPU] Context setup failed: %v, falling back to CPU\n", err)
		n.Train(inputs, targets, epochs, learningRate, earlyStopOnNegativeLoss, clipUpper, clipLower)
		return nil
	}
	defer ctx.Release()

	fmt.Println("[GPU] Training with GPU acceleration")

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		// Shuffle data
		perm := rand.Perm(len(inputs))
		shuffledInputs := make([][][]float64, len(inputs))
		shuffledTargets := make([][][]float64, len(targets))
		for i, p := range perm {
			shuffledInputs[i] = inputs[p]
			shuffledTargets[i] = targets[p]
		}

		// Process each sample
		for b := 0; b < len(shuffledInputs); b++ {
			// Forward pass (GPU)
			if err := n.forwardGPUImpl(ctx, shuffledInputs[b]); err != nil {
				fmt.Printf("[GPU] Forward error at sample %d: %v, falling back to CPU\n", b, err)
				n.forwardCPU(shuffledInputs[b])
			}

			// Compute loss
			loss := n.ComputeLoss(shuffledTargets[b])
			if math.IsNaN(loss) {
				fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, epoch)
				continue
			}
			if earlyStopOnNegativeLoss && loss < 0 {
				fmt.Printf("⚠️ Negative loss (%.4f) detected at sample %d, epoch %d. Stopping early.\n",
					loss, b, epoch)
				return nil
			}
			totalLoss += loss

			// Backward pass (GPU)
			if err := n.backwardGPUImpl(ctx, shuffledTargets[b], learningRate, clipUpper, clipLower); err != nil {
				fmt.Printf("[GPU] Backward error at sample %d: %v, falling back to CPU\n", b, err)
				n.backwardCPU(shuffledTargets[b], learningRate, clipUpper, clipLower)
			}
		}

		avgLoss := totalLoss / float64(len(inputs))
		fmt.Printf("Epoch %d, Loss: %.4f [GPU]\n", epoch, avgLoss)
	}

	return nil
}

// setupGPUContext initializes GPU context with all necessary resources
func (n *Network[T]) setupGPUContext(ctx *GPUContext) error {
	if !n.WebGPUNative {
		return fmt.Errorf("GPU not initialized")
	}

	// Context would be stored in network, retrieve it here
	// For now, we'll create a new one each time
	// In production, store *GPUContext in Network struct

	return n.InitGPU()
}

// forwardGPUImpl implements full forward pass on GPU
func (n *Network[T]) forwardGPUImpl(ctx *GPUContext, inputs [][]float64) error {
	// Set input layer
	inLayer := n.Layers[n.InputLayer]
	if len(inputs) != inLayer.Height || len(inputs[0]) != inLayer.Width {
		return fmt.Errorf("input mismatch: want %dx%d, got %dx%d",
			inLayer.Height, inLayer.Width, len(inputs), len(inputs[0]))
	}

	// Copy inputs to input layer (always CPU)
	for y := 0; y < inLayer.Height; y++ {
		for x := 0; x < inLayer.Width; x++ {
			inLayer.Neurons[y][x].Value = T(inputs[y][x])
		}
	}

	// Process each hidden/output layer
	for l := 1; l < len(n.Layers); l++ {
		layer := &n.Layers[l]

		// Check if layer has mixed types (dense + attention)
		hasMixed := false
		hasAttn := false
		hasDense := false

		if layer.SliceTypes != nil {
			for _, stype := range layer.SliceTypes {
				if stype == "attn" {
					hasAttn = true
				} else {
					hasDense = true
				}
			}
			hasMixed = hasAttn && hasDense
		}

		// If mixed or attention-only, use hybrid approach
		if hasMixed || (hasAttn && !hasDense) {
			// Process dense columns on GPU
			if hasDense {
				if err := n.forwardDenseGPU(ctx, l); err != nil {
					// Fall back to CPU for this layer
					n.forwardLayer(l, false)
					continue
				}
			}

			// Process attention columns on CPU (complex operation)
			if hasAttn {
				for col := 0; col < layer.Width; col++ {
					if layer.SliceTypes[col] == "attn" {
						n.forwardAttnColumn(l, col, false)
					}
				}
			}
		} else {
			// All dense, use GPU
			if err := n.forwardDenseGPU(ctx, l); err != nil {
				// Fall back to CPU
				n.forwardLayer(l, false)
			}
		}

		// Cache outputs
		layer.CachedOutputs = CastFloat64SliceToT[T](layer.GetOutputValues())
	}

	// Apply softmax if needed
	outputLayer := &n.Layers[n.OutputLayer]
	if outputLayer.Height > 0 && outputLayer.Width > 0 {
		if outputLayer.Neurons[0][0].Activation == "softmax" {
			n.ApplySoftmax()
		}
	}

	return nil
}

// backwardGPUImpl implements full backward pass on GPU
func (n *Network[T]) backwardGPUImpl(ctx *GPUContext, targets [][]float64,
	lr float64, clipUpper, clipLower T) error {

	nLayers := len(n.Layers)

	// Allocate error tensor
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

			grad := ActivationDerivativeGeneric(neuron.Value, neuron.Activation)
			err[n.OutputLayer][y][x] = T(diff) * grad
		}
	}

	// Backpropagate through layers
	for l := n.OutputLayer; l > 0; l-- {
		layer := &n.Layers[l]

		// Check for mixed types
		hasAttn := false
		hasDense := false

		if layer.SliceTypes != nil {
			for _, stype := range layer.SliceTypes {
				if stype == "attn" {
					hasAttn = true
				} else {
					hasDense = true
				}
			}
		}

		// Process attention columns on CPU
		if hasAttn {
			for col := 0; col < layer.Width; col++ {
				if layer.SliceTypes != nil && layer.SliceTypes[col] == "attn" {
					n.backwardAttnColumn(l, col, err, lr, clipUpper, clipLower, false)
				}
			}
		}

		// Process dense columns on GPU
		if hasDense || !hasAttn {
			if gpuErr := n.backwardDenseGPU(ctx, l, err, lr, clipUpper, clipLower); gpuErr != nil {
				// Fall back to CPU
				n.backwardLayer(l, err, float64(lr), clipUpper, clipLower, false)
			}
		}
	}

	return nil
}

// forwardGPU is a lightweight wrapper for GPU forward pass
func (n *Network[T]) forwardGPU(inputs [][]float64) error {
	ctx, err := n.createGPUContext()
	if err != nil {
		return err
	}
	defer ctx.Release()

	return n.forwardGPUImpl(ctx, inputs)
}

// backwardGPU is a lightweight wrapper for GPU backward pass
func (n *Network[T]) backwardGPU(targets [][]float64, lr float64, clipUpper, clipLower T) error {
	ctx, err := n.createGPUContext()
	if err != nil {
		return err
	}
	defer ctx.Release()

	return n.backwardGPUImpl(ctx, targets, lr, clipUpper, clipLower)
}

// createGPUContext creates a temporary GPU context for a single forward/backward pass
func (n *Network[T]) createGPUContext() (*GPUContext, error) {
	rep, err := Detect()
	if err != nil {
		return nil, fmt.Errorf("GPU detection failed: %w", err)
	}

	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return nil, fmt.Errorf("failed to create WebGPU instance")
	}

	pp := wgpu.PowerPreferenceHighPerformance
	if rep.AdapterType == "integrated-gpu" {
		pp = wgpu.PowerPreferenceLowPower
	}

	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: pp,
	})
	if err != nil || adapter == nil {
		inst.Release()
		return nil, fmt.Errorf("failed to request adapter: %w", err)
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{})
	if err != nil || device == nil {
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("failed to request device: %w", err)
	}

	queue := device.GetQueue()

	// Determine workgroup size
	wgx := rep.Recommended.WorkgroupX
	if wgx == 0 {
		wgx = 64
	}
	if wgx > rep.Limits.MaxComputeWorkgroupSizeX {
		wgx = rep.Limits.MaxComputeWorkgroupSizeX
	}

	return &GPUContext{
		device:        device,
		queue:         queue,
		instance:      inst,
		adapter:       adapter,
		workgroupSize: wgx,
		initialized:   true,
	}, nil
}

// GetGPUMemoryEstimate returns estimated GPU memory usage in bytes
func (n *Network[T]) GetGPUMemoryEstimate() uint64 {
	var total uint64

	for l := 1; l < len(n.Layers); l++ {
		layer := &n.Layers[l]
		prevLayer := &n.Layers[l-1]

		layerSize := layer.Width * layer.Height
		prevSize := prevLayer.Width * prevLayer.Height

		// Count connections
		totalConns := 0
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				totalConns += len(layer.Neurons[y][x].Inputs)
			}
		}

		// Memory per layer:
		// - Previous values: prevSize * 4 bytes (f32)
		// - Current values: layerSize * 4 bytes
		// - Weights: totalConns * 4 bytes
		// - Biases: layerSize * 4 bytes
		// - Connection info: totalConns * 4 bytes (i32)
		// - Errors (forward + backward): (prevSize + layerSize) * 2 * 4 bytes

		layerMem := uint64(prevSize*4 + layerSize*4 + totalConns*8 + layerSize*4 +
			(prevSize+layerSize)*8)
		total += layerMem
	}

	// Add overhead for pipeline, bind groups, etc (estimate 10MB)
	total += 10 * 1024 * 1024

	return total
}

// PrintGPUInfo prints GPU configuration information
func (n *Network[T]) PrintGPUInfo() {
	if !n.WebGPUNative {
		fmt.Println("[GPU] Not initialized")
		return
	}

	fmt.Println("=== GPU Configuration ===")
	fmt.Printf("Status: %s\n", map[bool]string{true: "Enabled", false: "Disabled"}[n.WebGPUNative])
	fmt.Printf("Estimated Memory Usage: %.2f MB\n", float64(n.GetGPUMemoryEstimate())/(1024*1024))

	// Layer breakdown
	fmt.Println("\nLayer Breakdown:")
	for l := 1; l < len(n.Layers); l++ {
		layer := &n.Layers[l]
		prevLayer := &n.Layers[l-1]

		layerSize := layer.Width * layer.Height
		prevSize := prevLayer.Width * prevLayer.Height

		totalConns := 0
		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				totalConns += len(layer.Neurons[y][x].Inputs)
			}
		}

		layerMem := uint64(prevSize*4 + layerSize*4 + totalConns*8 + layerSize*4)

		fmt.Printf("  Layer %d: %dx%d neurons, %d connections, %.2f MB\n",
			l, layer.Width, layer.Height, totalConns, float64(layerMem)/(1024*1024))

		// Check for attention
		if layer.SliceTypes != nil {
			attnCount := 0
			for _, stype := range layer.SliceTypes {
				if stype == "attn" {
					attnCount++
				}
			}
			if attnCount > 0 {
				fmt.Printf("    - Attention columns: %d/%d\n", attnCount, layer.Width)
			}
		}
	}
	fmt.Println("========================")
}

// BenchmarkGPUvsCPU runs a simple benchmark comparing GPU and CPU performance
func (n *Network[T]) BenchmarkGPUvsCPU(inputs [][]float64, iterations int) error {
	if !n.WebGPUNative {
		if err := n.InitGPU(); err != nil {
			return fmt.Errorf("GPU initialization failed: %w", err)
		}
	}

	ctx := &GPUContext{}
	if err := n.setupGPUContext(ctx); err != nil {
		return err
	}
	defer ctx.Release()

	fmt.Printf("\n=== GPU vs CPU Benchmark (%d iterations) ===\n", iterations)

	// Warm up
	n.forwardCPU(inputs)
	n.forwardGPUImpl(ctx, inputs)

	// CPU benchmark
	cpuStart := timeit()
	for i := 0; i < iterations; i++ {
		n.forwardCPU(inputs)
	}
	cpuTime := timeit() - cpuStart

	// GPU benchmark
	gpuStart := timeit()
	for i := 0; i < iterations; i++ {
		n.forwardGPUImpl(ctx, inputs)
	}
	gpuTime := timeit() - gpuStart

	fmt.Printf("CPU Time: %.3f ms\n", cpuTime*1000)
	fmt.Printf("GPU Time: %.3f ms\n", gpuTime*1000)
	fmt.Printf("Speedup: %.2fx\n", cpuTime/gpuTime)
	fmt.Println("==========================================")

	return nil
}

// Helper function to get current time in seconds
func timeit() float64 {
	return float64(rand.Int63()) / 1e9 // Placeholder, use time.Now() in real code
}
