package paragon

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// forwardLayerGPU executes forward pass for a single layer on GPU
func (n *Network[T]) forwardLayerGPU(ctx *GPUContext, layerIdx int) error {
	currLayer := &n.Layers[layerIdx]

	// Check for attention columns
	hasAttn := false
	if currLayer.SliceTypes != nil {
		for _, stype := range currLayer.SliceTypes {
			if stype == "attn" {
				hasAttn = true
				break
			}
		}
	}

	// Process dense columns
	if err := n.forwardDenseGPU(ctx, layerIdx); err != nil {
		return fmt.Errorf("forward dense GPU: %w", err)
	}

	// Process attention columns if any
	if hasAttn {
		for col := 0; col < currLayer.Width; col++ {
			if currLayer.SliceTypes[col] == "attn" {
				// Attention still on CPU for now (complex multi-pass operation)
				n.forwardAttnColumn(layerIdx, col, false)
			}
		}
	}

	// Cache outputs
	currLayer.CachedOutputs = CastFloat64SliceToT[T](currLayer.GetOutputValues())

	return nil
}

// forwardDenseGPU executes dense forward pass on GPU
func (n *Network[T]) forwardDenseGPU(ctx *GPUContext, layerIdx int) error {
	currLayer := &n.Layers[layerIdx]
	prevLayer := &n.Layers[layerIdx-1]

	currSize := currLayer.Width * currLayer.Height

	// Flatten data
	weights, connInfo, maxConns := n.flattenConnections(layerIdx)
	prevValues := prevLayer.flattenValues()
	biases := currLayer.flattenBiases()
	currValues := make([]float32, currSize)

	// Create buffers
	prevBuf, err := createBuffer(ctx.device, "prev_values", uint64(len(prevValues)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer prevBuf.Release()

	weightsBuf, err := createBuffer(ctx.device, "weights", uint64(len(weights)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer weightsBuf.Release()

	biasesBuf, err := createBuffer(ctx.device, "biases", uint64(len(biases)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer biasesBuf.Release()

	connBuf, err := createBuffer(ctx.device, "conn_info", uint64(len(connInfo)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer connBuf.Release()

	currBuf, err := createBuffer(ctx.device, "curr_values", uint64(len(currValues)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		return err
	}
	defer currBuf.Release()

	// Upload data
	uploadBuffer(ctx.queue, prevBuf, prevValues)
	uploadBuffer(ctx.queue, weightsBuf, weights)
	uploadBuffer(ctx.queue, biasesBuf, biases)
	uploadBufferInt32(ctx.queue, connBuf, connInfo)

	if err := waitForGPU(ctx.device, 100); err != nil {
		return err
	}

	// Get activation function
	activation := "relu"
	if currLayer.Height > 0 && currLayer.Width > 0 {
		activation = currLayer.Neurons[0][0].Activation
	}

	// Create shader
	shader := generateForwardShader(ctx.workgroupSize, currLayer.Width, currLayer.Height,
		maxConns, activation)

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "forward_shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return fmt.Errorf("create shader: %w", err)
	}
	defer module.Release()

	// Create bind group layout
	bgl, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "forward_bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		module.Release()
		return err
	}
	defer bgl.Release()

	// Create pipeline
	pl, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "forward_pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return err
	}
	defer pl.Release()

	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "forward_pipeline",
		Layout: pl,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return err
	}
	defer pipeline.Release()

	// Create bind group
	bg, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "forward_bg",
		Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: prevBuf, Offset: 0, Size: prevBuf.GetSize()},
			{Binding: 1, Buffer: weightsBuf, Offset: 0, Size: weightsBuf.GetSize()},
			{Binding: 2, Buffer: biasesBuf, Offset: 0, Size: biasesBuf.GetSize()},
			{Binding: 3, Buffer: connBuf, Offset: 0, Size: connBuf.GetSize()},
			{Binding: 4, Buffer: currBuf, Offset: 0, Size: currBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	// Execute
	enc, err := ctx.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "forward_enc",
	})
	if err != nil {
		return err
	}

	pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
		Label: "forward_pass",
	})
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bg, nil)

	// Dispatch workgroups
	numGroups := uint32((currSize + int(ctx.workgroupSize) - 1) / int(ctx.workgroupSize))
	if numGroups == 0 {
		numGroups = 1
	}
	pass.DispatchWorkgroups(numGroups, 1, 1)
	pass.End()

	cb, err := enc.Finish(nil)
	if err != nil {
		enc.Release()
		return err
	}
	enc.Release()

	ctx.queue.Submit(cb)
	cb.Release()

	if err := waitForGPU(ctx.device, 1000); err != nil {
		return err
	}

	// Download results
	result, err := downloadBuffer(ctx.device, currBuf, uint64(len(currValues)*4))
	if err != nil {
		return err
	}

	// Write back to network
	currLayer.unflattenValues(result)

	return nil
}

// backwardLayerGPU executes backward pass for a single layer on GPU
func (n *Network[T]) backwardLayerGPU(ctx *GPUContext, layerIdx int, err [][][]T,
	lr float64, clipUpper, clipLower T) error {

	currLayer := &n.Layers[layerIdx]

	// Check for attention columns
	hasAttn := false
	if currLayer.SliceTypes != nil {
		for _, stype := range currLayer.SliceTypes {
			if stype == "attn" {
				hasAttn = true
				break
			}
		}
	}

	// Process attention columns if any
	if hasAttn {
		for col := 0; col < currLayer.Width; col++ {
			if currLayer.SliceTypes[col] == "attn" {
				// Attention backward still on CPU (complex multi-pass operation)
				n.backwardAttnColumn(layerIdx, col, err, lr, clipUpper, clipLower, false)
			}
		}
	}

	// Process dense columns
	if err := n.backwardDenseGPU(ctx, layerIdx, err, lr, clipUpper, clipLower); err != nil {
		return fmt.Errorf("backward dense GPU: %w", err)
	}

	return nil
}

// backwardDenseGPU executes dense backward pass on GPU
func (n *Network[T]) backwardDenseGPU(ctx *GPUContext, layerIdx int, errTensor [][][]T,
	lr float64, clipUpper, clipLower T) error {

	currLayer := &n.Layers[layerIdx]
	prevLayer := &n.Layers[layerIdx-1]

	prevSize := prevLayer.Width * prevLayer.Height
	currSize := currLayer.Width * currLayer.Height

	// Flatten data
	weights, connInfo, maxConns := n.flattenConnections(layerIdx)
	currValues := currLayer.flattenValues()
	prevValues := prevLayer.flattenValues()
	biases := currLayer.flattenBiases()

	// Flatten error tensors
	currErrors := make([]float32, currSize)
	prevErrors := make([]float32, prevSize)
	idx := 0
	for y := 0; y < currLayer.Height; y++ {
		for x := 0; x < currLayer.Width; x++ {
			currErrors[idx] = float32(any(errTensor[layerIdx][y][x]).(T))
			idx++
		}
	}

	// Create buffers
	currValBuf, err := createBuffer(ctx.device, "curr_values", uint64(len(currValues)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer currValBuf.Release()

	prevValBuf, err := createBuffer(ctx.device, "prev_values", uint64(len(prevValues)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer prevValBuf.Release()

	currErrBuf, err := createBuffer(ctx.device, "curr_errors", uint64(len(currErrors)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer currErrBuf.Release()

	prevErrBuf, err := createBuffer(ctx.device, "prev_errors", uint64(len(prevErrors)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		return err
	}
	defer prevErrBuf.Release()

	weightsBuf, err := createBuffer(ctx.device, "weights", uint64(len(weights)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	if err != nil {
		return err
	}
	defer weightsBuf.Release()

	biasesBuf, err := createBuffer(ctx.device, "biases", uint64(len(biases)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
	if err != nil {
		return err
	}
	defer biasesBuf.Release()

	connBuf, err := createBuffer(ctx.device, "conn_info", uint64(len(connInfo)*4),
		wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		return err
	}
	defer connBuf.Release()

	// Upload data
	uploadBuffer(ctx.queue, currValBuf, currValues)
	uploadBuffer(ctx.queue, prevValBuf, prevValues)
	uploadBuffer(ctx.queue, currErrBuf, currErrors)
	uploadBuffer(ctx.queue, prevErrBuf, prevErrors) // Initialize to zero
	uploadBuffer(ctx.queue, weightsBuf, weights)
	uploadBuffer(ctx.queue, biasesBuf, biases)
	uploadBufferInt32(ctx.queue, connBuf, connInfo)

	if err := waitForGPU(ctx.device, 100); err != nil {
		return err
	}

	// Get activation function
	activation := "relu"
	if currLayer.Height > 0 && currLayer.Width > 0 {
		activation = currLayer.Neurons[0][0].Activation
	}

	// Create shader
	shader := generateBackwardShaderWithParams(ctx.workgroupSize, currLayer.Width, currLayer.Height,
		maxConns, activation, float32(lr), float32(clipUpper), float32(clipLower))

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "backward_shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
	})
	if err != nil {
		return fmt.Errorf("create shader: %w", err)
	}
	defer module.Release()

	// Create bind group layout
	bgl, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "backward_bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 6, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return err
	}
	defer bgl.Release()

	// Create pipeline
	pl, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "backward_pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return err
	}
	defer pl.Release()

	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  "backward_pipeline",
		Layout: pl,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return err
	}
	defer pipeline.Release()

	// Create bind group
	bg, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "backward_bg",
		Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: currValBuf, Offset: 0, Size: currValBuf.GetSize()},
			{Binding: 1, Buffer: prevValBuf, Offset: 0, Size: prevValBuf.GetSize()},
			{Binding: 2, Buffer: currErrBuf, Offset: 0, Size: currErrBuf.GetSize()},
			{Binding: 3, Buffer: prevErrBuf, Offset: 0, Size: prevErrBuf.GetSize()},
			{Binding: 4, Buffer: weightsBuf, Offset: 0, Size: weightsBuf.GetSize()},
			{Binding: 5, Buffer: biasesBuf, Offset: 0, Size: biasesBuf.GetSize()},
			{Binding: 6, Buffer: connBuf, Offset: 0, Size: connBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	// Execute
	enc, err := ctx.device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "backward_enc",
	})
	if err != nil {
		return err
	}

	pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
		Label: "backward_pass",
	})
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bg, nil)

	// Dispatch workgroups
	numGroups := uint32((currSize + int(ctx.workgroupSize) - 1) / int(ctx.workgroupSize))
	if numGroups == 0 {
		numGroups = 1
	}
	pass.DispatchWorkgroups(numGroups, 1, 1)
	pass.End()

	cb, err := enc.Finish(nil)
	if err != nil {
		enc.Release()
		return err
	}
	enc.Release()

	ctx.queue.Submit(cb)
	cb.Release()

	if err := waitForGPU(ctx.device, 1000); err != nil {
		return err
	}

	// Download updated weights and biases
	updatedWeights, err := downloadBuffer(ctx.device, weightsBuf, uint64(len(weights)*4))
	if err != nil {
		return err
	}

	updatedBiases, err := downloadBuffer(ctx.device, biasesBuf, uint64(len(biases)*4))
	if err != nil {
		return err
	}

	// Download prev errors
	prevErrResult, err := downloadBuffer(ctx.device, prevErrBuf, uint64(len(prevErrors)*4))
	if err != nil {
		return err
	}

	// Write back to network
	n.unflattenWeights(layerIdx, updatedWeights, maxConns)
	currLayer.unflattenBiases(updatedBiases)

	// Write prev errors back to error tensor
	idx = 0
	for y := 0; y < prevLayer.Height; y++ {
		for x := 0; x < prevLayer.Width; x++ {
			errTensor[layerIdx-1][y][x] += T(prevErrResult[idx])
			idx++
		}
	}

	// Apply activation derivative to prev layer errors
	if layerIdx-1 > 0 {
		for y := 0; y < prevLayer.Height; y++ {
			for x := 0; x < prevLayer.Width; x++ {
				prevErr := errTensor[layerIdx-1][y][x]
				deriv := ActivationDerivativeGeneric(
					prevLayer.Neurons[y][x].Value,
					prevLayer.Neurons[y][x].Activation,
				)
				errTensor[layerIdx-1][y][x] = prevErr * deriv
			}
		}
	}

	return nil
}

// generateBackwardShaderWithParams creates backward shader with runtime parameters
func generateBackwardShaderWithParams(wgx uint32, layerWidth, layerHeight, prevSize int,
	activation string, lr, clipUpper, clipLower float32) string {

	derivCode := generateActivationDerivWGSL(activation)

	return fmt.Sprintf(`
// Backward pass shader for dense layer
@group(0) @binding(0) var<storage, read>       curr_values : array<f32>;
@group(0) @binding(1) var<storage, read>       prev_values : array<f32>;
@group(0) @binding(2) var<storage, read>       curr_errors : array<f32>;
@group(0) @binding(3) var<storage, read_write> prev_errors : array<f32>;
@group(0) @binding(4) var<storage, read_write> weights     : array<f32>;
@group(0) @binding(5) var<storage, read_write> biases      : array<f32>;
@group(0) @binding(6) var<storage, read>       conn_info   : array<i32>;

const LAYER_WIDTH: u32 = %du;
const LAYER_HEIGHT: u32 = %du;
const PREV_SIZE: u32 = %du;
const LEARNING_RATE: f32 = %f;
const CLIP_UPPER: f32 = %f;
const CLIP_LOWER: f32 = %f;

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
    
    let curr_val = curr_values[neuron_idx];
    let err = curr_errors[neuron_idx];
    let deriv = activate_deriv(curr_val);
    let local_error = err * deriv;
    
    biases[neuron_idx] = biases[neuron_idx] + (LEARNING_RATE * local_error);
    
    let conn_base = neuron_idx * PREV_SIZE;
    
    for (var i: u32 = 0u; i < PREV_SIZE; i = i + 1u) {
        let src_idx = conn_info[conn_base + i];
        if (src_idx >= 0 && u32(src_idx) < PREV_SIZE) {
            let src_val = prev_values[u32(src_idx)];
            
            var grad = local_error * src_val;
            grad = clip(grad);
            
            let weight_idx = conn_base + i;
            weights[weight_idx] = weights[weight_idx] + (LEARNING_RATE * grad);
            
            let weight = weights[weight_idx];
            let back_err = local_error * weight;
            
            prev_errors[u32(src_idx)] = prev_errors[u32(src_idx)] + back_err;
        }
    }
}
`, layerWidth, layerHeight, prevSize, lr, clipUpper, clipLower, derivCode, wgx)
}

// uploadBufferInt32 uploads int32 data to GPU buffer
func uploadBufferInt32(queue *wgpu.Queue, buffer *wgpu.Buffer, data []int32) {
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	queue.WriteBuffer(buffer, 0, bytes)
}
