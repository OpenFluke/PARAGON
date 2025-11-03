package paragon

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// GPUContext holds WebGPU resources for neural network acceleration
type GPUContext struct {
	device   *wgpu.Device
	queue    *wgpu.Queue
	instance *wgpu.Instance
	adapter  *wgpu.Adapter

	// Pipeline caches
	forwardPipeline  *wgpu.ComputePipeline
	backwardPipeline *wgpu.ComputePipeline
	attnForwardPipe  *wgpu.ComputePipeline
	attnBackwardPipe *wgpu.ComputePipeline

	// Bind group layouts
	forwardBGL  *wgpu.BindGroupLayout
	backwardBGL *wgpu.BindGroupLayout
	attnBGL     *wgpu.BindGroupLayout

	// Workgroup size
	workgroupSize uint32

	// Network metadata
	initialized bool
}

// InitGPU initializes WebGPU context for the network
func (n *Network[T]) InitGPU() error {
	if n.WebGPUNative {
		return nil // Already initialized
	}

	rep, err := Detect()
	if err != nil {
		return fmt.Errorf("GPU detection failed: %w", err)
	}

	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return fmt.Errorf("failed to create WebGPU instance")
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
		return fmt.Errorf("failed to request adapter: %w", err)
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{})
	if err != nil || device == nil {
		adapter.Release()
		inst.Release()
		return fmt.Errorf("failed to request device: %w", err)
	}

	// Determine workgroup size
	wgx := rep.Recommended.WorkgroupX
	if wgx == 0 {
		wgx = 64
	}
	if wgx > rep.Limits.MaxComputeWorkgroupSizeX {
		wgx = rep.Limits.MaxComputeWorkgroupSizeX
	}

	// Mark as initialized
	n.WebGPUNative = true

	fmt.Printf("[GPU] Initialized with workgroup size: %d\n", wgx)

	return nil
}

// EnableGPU is a simple method to enable GPU acceleration
// It enables GPU and returns error if GPU is not available
func (n *Network[T]) EnableGPU() error {
	return n.InitGPU()
}

// ReleaseGPU cleans up WebGPU resources
func (ctx *GPUContext) Release() {
	if ctx.forwardPipeline != nil {
		ctx.forwardPipeline.Release()
	}
	if ctx.backwardPipeline != nil {
		ctx.backwardPipeline.Release()
	}
	if ctx.attnForwardPipe != nil {
		ctx.attnForwardPipe.Release()
	}
	if ctx.attnBackwardPipe != nil {
		ctx.attnBackwardPipe.Release()
	}
	if ctx.forwardBGL != nil {
		ctx.forwardBGL.Release()
	}
	if ctx.backwardBGL != nil {
		ctx.backwardBGL.Release()
	}
	if ctx.attnBGL != nil {
		ctx.attnBGL.Release()
	}
	if ctx.device != nil {
		ctx.device.Release()
	}
	if ctx.adapter != nil {
		ctx.adapter.Release()
	}
	if ctx.instance != nil {
		ctx.instance.Release()
	}
}

// ForwardGPU runs forward pass on GPU
func (n *Network[T]) ForwardGPU(inputs [][]float64) error {
	if !n.WebGPUNative {
		return fmt.Errorf("GPU not initialized, call InitGPU first")
	}

	// For now, fall back to CPU but with structure in place
	// Full implementation will follow
	n.forwardCPU(inputs)
	return nil
}

// BackwardGPU runs backward pass on GPU
func (n *Network[T]) BackwardGPU(targets [][]float64, lr float64, clipUpper, clipLower T) error {
	if !n.WebGPUNative {
		return fmt.Errorf("GPU not initialized, call InitGPU first")
	}

	// For now, fall back to CPU but with structure in place
	n.backwardCPU(targets, lr, clipUpper, clipLower)
	return nil
}

// generateForwardShader creates WGSL shader for dense forward pass
func generateForwardShader(wgx uint32, layerWidth, layerHeight, prevSize int, activation string) string {
	activationCode := generateActivationWGSL(activation)

	return fmt.Sprintf(`
// Forward pass shader for dense layer
@group(0) @binding(0) var<storage, read>       prev_values : array<f32>;  // Previous layer activations
@group(0) @binding(1) var<storage, read>       weights     : array<f32>;  // Connection weights (flattened)
@group(0) @binding(2) var<storage, read>       biases      : array<f32>;  // Neuron biases
@group(0) @binding(3) var<storage, read>       conn_info   : array<i32>;  // Connection indices (source neuron indices)
@group(0) @binding(4) var<storage, read_write> curr_values : array<f32>;  // Current layer activations (output)

const LAYER_WIDTH: u32 = %du;
const LAYER_HEIGHT: u32 = %du;
const PREV_SIZE: u32 = %du;

%s

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron_idx = gid.x;
    let total_neurons = LAYER_WIDTH * LAYER_HEIGHT;
    
    if (neuron_idx >= total_neurons) {
        return;
    }
    
    // Get y,x coordinates
    let y = neuron_idx / LAYER_WIDTH;
    let x = neuron_idx %% LAYER_WIDTH;
    
    // Accumulate weighted sum
    var sum: f32 = biases[neuron_idx];
    
    // Each neuron has PREV_SIZE connections
    let conn_base = neuron_idx * PREV_SIZE;
    
    for (var i: u32 = 0u; i < PREV_SIZE; i = i + 1u) {
        let src_idx = conn_info[conn_base + i];
        if (src_idx >= 0 && u32(src_idx) < PREV_SIZE) {
            let weight = weights[conn_base + i];
            let src_val = prev_values[u32(src_idx)];
            sum = sum + (weight * src_val);
        }
    }
    
    // Apply activation
    curr_values[neuron_idx] = activate(sum);
}
`, layerWidth, layerHeight, prevSize, activationCode, wgx)
}

// generateBackwardShader creates WGSL shader for dense backward pass
func generateBackwardShader(wgx uint32, layerWidth, layerHeight, prevSize int, activation string) string {
	derivCode := generateActivationDerivWGSL(activation)

	return fmt.Sprintf(`
// Backward pass shader for dense layer
@group(0) @binding(0) var<storage, read>       curr_values : array<f32>;  // Current layer activations
@group(0) @binding(1) var<storage, read>       prev_values : array<f32>;  // Previous layer activations
@group(0) @binding(2) var<storage, read>       curr_errors : array<f32>;  // Current layer errors
@group(0) @binding(3) var<storage, read_write> prev_errors : array<f32>;  // Previous layer errors (output)
@group(0) @binding(4) var<storage, read_write> weights     : array<f32>;  // Weights (updated)
@group(0) @binding(5) var<storage, read_write> biases      : array<f32>;  // Biases (updated)
@group(0) @binding(6) var<storage, read>       conn_info   : array<i32>;  // Connection indices

const LAYER_WIDTH: u32 = %du;
const LAYER_HEIGHT: u32 = %du;
const PREV_SIZE: u32 = %du;
const LEARNING_RATE: f32 = 0.001;  // Will be updated via uniform
const CLIP_UPPER: f32 = 10.0;
const CLIP_LOWER: f32 = -10.0;

%s

fn clip(val: f32) -> f32 {
    return clamp(val, CLIP_LOWER, CLIP_UPPER);
}

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron_idx = gid.x;
    let total_neurons = LAYER_WIDTH * LAYER_HEIGHT;
    
    if (neuron_idx >= total_neurons) {
        return;
    }
    
    // Get local error with activation derivative
    let curr_val = curr_values[neuron_idx];
    let err = curr_errors[neuron_idx];
    let deriv = activate_deriv(curr_val);
    let local_error = err * deriv;
    
    // Update bias
    biases[neuron_idx] = biases[neuron_idx] + (LEARNING_RATE * local_error);
    
    // Update weights and propagate error
    let conn_base = neuron_idx * PREV_SIZE;
    
    for (var i: u32 = 0u; i < PREV_SIZE; i = i + 1u) {
        let src_idx = conn_info[conn_base + i];
        if (src_idx >= 0 && u32(src_idx) < PREV_SIZE) {
            let src_val = prev_values[u32(src_idx)];
            
            // Compute gradient
            var grad = local_error * src_val;
            grad = clip(grad);
            
            // Update weight
            let weight_idx = conn_base + i;
            weights[weight_idx] = weights[weight_idx] + (LEARNING_RATE * grad);
            
            // Propagate error back
            let weight = weights[weight_idx];
            let back_err = local_error * weight;
            
            // Atomic add to prev_errors (approximate with non-atomic for now)
            prev_errors[u32(src_idx)] = prev_errors[u32(src_idx)] + back_err;
        }
    }
}
`, layerWidth, layerHeight, prevSize, derivCode, wgx)
}

// generateActivationWGSL generates WGSL code for activation functions
func generateActivationWGSL(activation string) string {
	switch activation {
	case "relu":
		return `
fn activate(x: f32) -> f32 {
    return max(0.0, x);
}`
	case "sigmoid":
		return `
fn activate(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}`
	case "tanh":
		return `
fn activate(x: f32) -> f32 {
    let e2x = exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}`
	case "leaky_relu":
		return `
fn activate(x: f32) -> f32 {
    if (x < 0.0) {
        return x * 0.01;
    }
    return x;
}`
	case "elu":
		return `
fn activate(x: f32) -> f32 {
    if (x >= 0.0) {
        return x;
    }
    return exp(x) - 1.0;
}`
	case "linear":
		return `
fn activate(x: f32) -> f32 {
    return x;
}`
	case "softmax":
		// Softmax requires special handling (global operation)
		return `
fn activate(x: f32) -> f32 {
    return x; // Softmax applied separately
}`
	default:
		return `
fn activate(x: f32) -> f32 {
    return x;
}`
	}
}

// generateActivationDerivWGSL generates WGSL code for activation derivatives
func generateActivationDerivWGSL(activation string) string {
	switch activation {
	case "relu":
		return `
fn activate_deriv(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    }
    return 0.0;
}`
	case "sigmoid":
		return `
fn activate_deriv(x: f32) -> f32 {
    let s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 - s);
}`
	case "tanh":
		return `
fn activate_deriv(x: f32) -> f32 {
    let t = tanh(x);
    return 1.0 - (t * t);
}`
	case "leaky_relu":
		return `
fn activate_deriv(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    }
    return 0.01;
}`
	case "elu":
		return `
fn activate_deriv(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0;
    }
    return exp(x);
}`
	case "linear":
		return `
fn activate_deriv(x: f32) -> f32 {
    return 1.0;
}`
	default:
		return `
fn activate_deriv(x: f32) -> f32 {
    return 1.0;
}`
	}
}

// generateAttnForwardShader creates WGSL shader for attention forward pass
func generateAttnForwardShader(wgx uint32, N, dk, heads, currHeight int) string {
	Hd := heads * dk

	return fmt.Sprintf(`
// Attention forward pass shader (multi-head)
@group(0) @binding(0) var<storage, read>       tokens  : array<f32>;    // Input tokens [N]
@group(0) @binding(1) var<storage, read>       wq      : array<f32>;    // Query weights [heads][dk]
@group(0) @binding(2) var<storage, read>       wk      : array<f32>;    // Key weights [heads][dk]
@group(0) @binding(3) var<storage, read>       wv      : array<f32>;    // Value weights [heads][dk]
@group(0) @binding(4) var<storage, read>       wo      : array<f32>;    // Output projection [Hd * currHeight]
@group(0) @binding(5) var<storage, read_write> output  : array<f32>;    // Output [currHeight]
@group(0) @binding(6) var<storage, read_write> scratch : array<f32>;    // Scratch space for intermediate values

const N: u32 = %du;           // Number of tokens
const DK: u32 = %du;          // Head dimension
const HEADS: u32 = %du;       // Number of heads
const HD: u32 = %du;          // heads * dk
const CURR_HEIGHT: u32 = %du; // Output dimension

// Compute Q, K, V for all tokens for one head
@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_idx = gid.x;
    
    if (head_idx >= HEADS) {
        return;
    }
    
    // For each head, compute attention over all tokens
    // This is a simplified version - full implementation would need multiple passes
    
    // TODO: Full multi-pass attention implementation
    // Pass 1: Compute Q, K, V projections
    // Pass 2: Compute attention scores
    // Pass 3: Apply softmax
    // Pass 4: Compute weighted values
    // Pass 5: Concatenate and project
}
`, N, dk, heads, Hd, currHeight, wgx)
}

// waitForGPU waits for GPU operations to complete
func waitForGPU(device *wgpu.Device, maxIters int) error {
	for i := 0; i < maxIters; i++ {
		if device.Poll(true, nil) {
			return nil
		}
		time.Sleep(100 * time.Microsecond)
	}
	return fmt.Errorf("GPU timeout after %d iterations", maxIters)
}

// createBuffer creates a GPU buffer
func createBuffer(device *wgpu.Device, label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Size:  size,
		Usage: usage,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create buffer %s: %w", label, err)
	}
	return buf, nil
}

// uploadBuffer uploads data to GPU buffer
func uploadBuffer(queue *wgpu.Queue, buffer *wgpu.Buffer, data []float32) {
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	queue.WriteBuffer(buffer, 0, bytes)
}

// downloadBuffer downloads data from GPU buffer
func downloadBuffer(device *wgpu.Device, buffer *wgpu.Buffer, size uint64) ([]float32, error) {
	readback, err := createBuffer(device, "readback", size,
		wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
	if err != nil {
		return nil, err
	}
	defer readback.Release()

	// Copy to readback buffer
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	enc.CopyBufferToBuffer(buffer, 0, readback, 0, size)
	cb, err := enc.Finish(nil)
	if err != nil {
		enc.Release()
		return nil, err
	}
	enc.Release()

	queue := device.GetQueue()
	queue.Submit(cb)
	cb.Release()

	if err := waitForGPU(device, 1000); err != nil {
		return nil, err
	}

	// Map and read
	done := false
	readback.MapAsync(wgpu.MapModeRead, 0, size, func(wgpu.BufferMapAsyncStatus) {
		done = true
	})

	for i := 0; i < 1000 && !done; i++ {
		device.Poll(true, nil)
		time.Sleep(100 * time.Microsecond)
	}

	if !done {
		return nil, fmt.Errorf("timeout mapping readback buffer")
	}

	view := readback.GetMappedRange(0, uint(size))
	result := make([]float32, int(size)/4)
	copy(result, unsafe.Slice((*float32)(unsafe.Pointer(&view[0])), len(result)))
	readback.Unmap()

	return result, nil
}

// flattenConnections converts Network connections to flat arrays for GPU
func (n *Network[T]) flattenConnections(layerIdx int) (weights []float32, connInfo []int32, numConns int) {
	layer := &n.Layers[layerIdx]
	totalNeurons := layer.Width * layer.Height

	// First pass: determine max connections per neuron
	maxConns := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			neuron := layer.Neurons[y][x]
			if len(neuron.Inputs) > maxConns {
				maxConns = len(neuron.Inputs)
			}
		}
	}

	numConns = maxConns
	weights = make([]float32, totalNeurons*maxConns)
	connInfo = make([]int32, totalNeurons*maxConns)

	// Initialize with zeros and -1 (invalid indices)
	for i := range connInfo {
		connInfo[i] = -1
	}

	// Fill arrays
	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			neuron := layer.Neurons[y][x]
			base := idx * maxConns

			for i, conn := range neuron.Inputs {
				// Flatten source index: sourceY * prevWidth + sourceX
				prevLayer := &n.Layers[conn.SourceLayer]
				srcIdx := conn.SourceY*prevLayer.Width + conn.SourceX

				weights[base+i] = float32(any(conn.Weight).(T))
				connInfo[base+i] = int32(srcIdx)
			}
			idx++
		}
	}

	return weights, connInfo, maxConns
}

// flattenValues converts layer neuron values to flat array
func (layer *Grid[T]) flattenValues() []float32 {
	result := make([]float32, layer.Width*layer.Height)
	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			result[idx] = float32(any(layer.Neurons[y][x].Value).(T))
			idx++
		}
	}
	return result
}

// flattenBiases converts layer neuron biases to flat array
func (layer *Grid[T]) flattenBiases() []float32 {
	result := make([]float32, layer.Width*layer.Height)
	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			result[idx] = float32(any(layer.Neurons[y][x].Bias).(T))
			idx++
		}
	}
	return result
}

// unflattenValues writes flat array back to layer neuron values
func (layer *Grid[T]) unflattenValues(data []float32) {
	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			layer.Neurons[y][x].Value = T(data[idx])
			idx++
		}
	}
}

// unflattenWeights writes flat weight array back to neuron connections
func (n *Network[T]) unflattenWeights(layerIdx int, weights []float32, maxConns int) {
	layer := &n.Layers[layerIdx]
	idx := 0

	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			neuron := layer.Neurons[y][x]
			base := idx * maxConns

			for i := range neuron.Inputs {
				neuron.Inputs[i].Weight = T(weights[base+i])
			}
			idx++
		}
	}
}

// unflattenBiases writes flat bias array back to neurons
func (layer *Grid[T]) unflattenBiases(data []float32) {
	idx := 0
	for y := 0; y < layer.Height; y++ {
		for x := 0; x < layer.Width; x++ {
			layer.Neurons[y][x].Bias = T(data[idx])
			idx++
		}
	}
}
