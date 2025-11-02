// webgpu_backward.go
package paragon

import (
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
)

// GPUBackwardResources holds GPU resources for backpropagation
type GPUBackwardResources struct {
	errorBuffers    []*wgpu.Buffer // Error signal for each layer
	gradientBuffers []*wgpu.Buffer // Gradient accumulation buffers
	targetBuffer    *wgpu.Buffer   // Target values buffer
	lrBuffer        *wgpu.Buffer   // Learning rate buffer
	clipBuffer      *wgpu.Buffer   // Clipping values buffer

	backwardPipelines  []*wgpu.ComputePipeline
	updatePipelines    []*wgpu.ComputePipeline
	backwardBindGroups []*wgpu.BindGroup
	updateBindGroups   []*wgpu.BindGroup
}

// InitializeBackwardGPU sets up GPU resources for backpropagation
func (n *Network[T]) InitializeBackwardGPU() error {
	if n.gpu.optimized == nil || !n.gpu.optimized.initialized {
		return fmt.Errorf("forward GPU must be initialized first")
	}

	n.gpu.wgslType = getWGSLType[float32]()

	// Create backward resources
	n.gpu.optimized.backward = &GPUBackwardResources{}

	// Create error buffers for each layer
	for l := 0; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]
		size := uint64(layer.Width * layer.Height * 4)

		errorBuf, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("Layer_%d_Error", l),
			Size:  size,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return fmt.Errorf("failed to create error buffer for layer %d: %v", l, err)
		}
		n.gpu.optimized.backward.errorBuffers = append(n.gpu.optimized.backward.errorBuffers, errorBuf)
	}

	// Create gradient accumulation buffers for each layer
	for l := 1; l <= n.OutputLayer; l++ {
		layer := n.Layers[l]
		prevLayer := n.Layers[l-1]

		// Size for weights + biases
		weightSize := layer.Width * layer.Height * prevLayer.Width * prevLayer.Height
		biasSize := layer.Width * layer.Height
		totalSize := uint64((weightSize + biasSize) * 4)

		gradBuf, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("Layer_%d_Gradients", l),
			Size:  totalSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return fmt.Errorf("failed to create gradient buffer for layer %d: %v", l, err)
		}
		n.gpu.optimized.backward.gradientBuffers = append(n.gpu.optimized.backward.gradientBuffers, gradBuf)
	}

	// Create constant buffers
	n.gpu.optimized.backward.lrBuffer, _ = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "LearningRate",
		Size:  4,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})

	n.gpu.optimized.backward.clipBuffer, _ = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ClipValues",
		Size:  8, // upper and lower
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})

	// Create target buffer for output layer
	outputSize := n.Layers[n.OutputLayer].Width * n.Layers[n.OutputLayer].Height
	n.gpu.optimized.backward.targetBuffer, _ = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "Targets",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})

	// Create backward propagation pipelines
	for l := n.OutputLayer; l > 0; l-- {
		pipeline, bindGroup, err := n.createBackwardPipeline(l)
		if err != nil {
			return fmt.Errorf("failed to create backward pipeline for layer %d: %v", l, err)
		}
		n.gpu.optimized.backward.backwardPipelines = append(n.gpu.optimized.backward.backwardPipelines, pipeline)
		n.gpu.optimized.backward.backwardBindGroups = append(n.gpu.optimized.backward.backwardBindGroups, bindGroup)

		// Create weight update pipeline
		updatePipeline, updateBindGroup, err := n.createWeightUpdatePipeline(l)
		if err != nil {
			return fmt.Errorf("failed to create update pipeline for layer %d: %v", l, err)
		}
		n.gpu.optimized.backward.updatePipelines = append(n.gpu.optimized.backward.updatePipelines, updatePipeline)
		n.gpu.optimized.backward.updateBindGroups = append(n.gpu.optimized.backward.updateBindGroups, updateBindGroup)
	}

	return nil
}

// Create backward propagation pipeline for a layer
func (n *Network[T]) createBackwardPipeline(layerIdx int) (*wgpu.ComputePipeline, *wgpu.BindGroup, error) {
	// Generate backward shader
	shaderCode := n.generateBackwardShader(layerIdx)

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Backward_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return nil, nil, err
	}
	defer module.Release()

	// Create bind group layout
	var entries []wgpu.BindGroupLayoutEntry

	if layerIdx == n.OutputLayer {
		// Output layer needs targets
		entries = []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // values
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // targets
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // errors
		}
	} else {
		// Hidden layers
		entries = []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // values
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // next layer errors
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // weights from next layer
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // current errors
		}
	}

	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label:   fmt.Sprintf("Layer_%d_Backward_Layout", layerIdx),
		Entries: entries,
	})
	if err != nil {
		return nil, nil, err
	}

	// Create pipeline layout
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_Backward_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, nil, err
	}
	defer pipelineLayout.Release()

	// Create compute pipeline
	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Backward_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, nil, err
	}

	// Create bind group
	var bindEntries []wgpu.BindGroupEntry

	if layerIdx == n.OutputLayer {
		bindEntries = []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: n.gpu.optimized.layers[layerIdx-1].outputBuffer, Size: n.gpu.optimized.layers[layerIdx-1].outputBuffer.GetSize()},
			{Binding: 1, Buffer: n.gpu.optimized.backward.targetBuffer, Size: n.gpu.optimized.backward.targetBuffer.GetSize()},
			{Binding: 2, Buffer: n.gpu.optimized.backward.errorBuffers[layerIdx], Size: n.gpu.optimized.backward.errorBuffers[layerIdx].GetSize()},
		}
	} else {
		// For hidden layers, we need the weights from the next layer
		nextLayerGPU := n.gpu.optimized.layers[layerIdx] // layerIdx because GPU layers are 0-indexed
		bindEntries = []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: n.gpu.optimized.layers[layerIdx-1].outputBuffer, Size: n.gpu.optimized.layers[layerIdx-1].outputBuffer.GetSize()},
			{Binding: 1, Buffer: n.gpu.optimized.backward.errorBuffers[layerIdx+1], Size: n.gpu.optimized.backward.errorBuffers[layerIdx+1].GetSize()},
			{Binding: 2, Buffer: nextLayerGPU.weightBuffer, Size: nextLayerGPU.weightBuffer.GetSize()},
			{Binding: 3, Buffer: n.gpu.optimized.backward.errorBuffers[layerIdx], Size: n.gpu.optimized.backward.errorBuffers[layerIdx].GetSize()},
		}
	}

	bindGroup, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   fmt.Sprintf("Layer_%d_Backward_BindGroup", layerIdx),
		Layout:  bindGroupLayout,
		Entries: bindEntries,
	})
	if err != nil {
		pipeline.Release()
		bindGroupLayout.Release()
		return nil, nil, err
	}

	return pipeline, bindGroup, nil
}

// Create weight update pipeline
func (n *Network[T]) createWeightUpdatePipeline(layerIdx int) (*wgpu.ComputePipeline, *wgpu.BindGroup, error) {
	shaderCode := n.generateWeightUpdateShader(layerIdx)

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Update_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return nil, nil, err
	}
	defer module.Release()

	// Create bind group layout
	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("Layer_%d_Update_Layout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // prev layer values
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // current layer errors
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // weights
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // biases
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},         // learning rate
			{Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},         // clip values
		},
	})
	if err != nil {
		return nil, nil, err
	}

	// Create pipeline
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_Update_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, nil, err
	}
	defer pipelineLayout.Release()

	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Update_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, nil, err
	}

	// Create bind group
	layerGPU := n.gpu.optimized.layers[layerIdx-1]
	var prevBuffer *wgpu.Buffer
	if layerIdx == 1 {
		prevBuffer = layerGPU.inputBuffer
	} else {
		prevBuffer = n.gpu.optimized.layers[layerIdx-2].outputBuffer
	}

	bindGroup, err := ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Update_BindGroup", layerIdx),
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: prevBuffer, Size: prevBuffer.GetSize()},
			{Binding: 1, Buffer: n.gpu.optimized.backward.errorBuffers[layerIdx], Size: n.gpu.optimized.backward.errorBuffers[layerIdx].GetSize()},
			{Binding: 2, Buffer: layerGPU.weightBuffer, Size: layerGPU.weightBuffer.GetSize()},
			{Binding: 3, Buffer: layerGPU.biasBuffer, Size: layerGPU.biasBuffer.GetSize()},
			{Binding: 4, Buffer: n.gpu.optimized.backward.lrBuffer, Size: n.gpu.optimized.backward.lrBuffer.GetSize()},
			{Binding: 5, Buffer: n.gpu.optimized.backward.clipBuffer, Size: n.gpu.optimized.backward.clipBuffer.GetSize()},
		},
	})
	if err != nil {
		pipeline.Release()
		bindGroupLayout.Release()
		return nil, nil, err
	}

	return pipeline, bindGroup, nil
}

// Generate backward propagation shader
func (n *Network[T]) generateBackwardShader(layerIdx int) string {
	layer := n.Layers[layerIdx]
	typ := n.gpu.wgslType
	activation := layer.Neurons[0][0].Activation

	if layerIdx == n.OutputLayer && activation == "softmax" {
		// Output layer with softmax: error = prediction - target
		outputSize := layer.Width * layer.Height
		return fmt.Sprintf(`
			@group(0) @binding(0) var<storage, read> values: array<%s>;  // post-softmax values
			@group(0) @binding(1) var<storage, read> targets: array<%s>;
			@group(0) @binding(2) var<storage, read_write> errors: array<%s>;

			@compute @workgroup_size(256, 1, 1)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				let idx = global_id.x;
				if (idx >= %du) {
					return;
				}
				let value = values[idx];
				let target_value = targets[idx];
				errors[idx] = value - target_value;  // Correct gradient for softmax + cross-entropy
			}
		`, typ, typ, typ, outputSize)
	} else {
		// Existing code for other layers or activations
		derivCode := getActivationDerivativeCode(activation, typ)
		if layerIdx == n.OutputLayer {
			// Non-softmax output layer
			outputSize := layer.Width * layer.Height
			return fmt.Sprintf(`
				@group(0) @binding(0) var<storage, read> values: array<%s>;
				@group(0) @binding(1) var<storage, read> targets: array<%s>;
				@group(0) @binding(2) var<storage, read_write> errors: array<%s>;

				%s

				@compute @workgroup_size(256, 1, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let idx = global_id.x;
					if (idx >= %du) {
						return;
					}
					let value = values[idx];
					let target_value = targets[idx];
					let diff = target_value - value;
					let deriv = activationDerivative(value);
					errors[idx] = diff * deriv;
				}
			`, typ, typ, typ, derivCode, outputSize)
		} else {
			// Hidden layer: propagate error from next layer
			currentSize := layer.Width * layer.Height
			nextLayer := n.Layers[layerIdx+1]
			nextSize := nextLayer.Width * nextLayer.Height

			return fmt.Sprintf(`
				@group(0) @binding(0) var<storage, read> values: array<%s>;
				@group(0) @binding(1) var<storage, read> nextErrors: array<%s>;
				@group(0) @binding(2) var<storage, read> nextWeights: array<%s>;
				@group(0) @binding(3) var<storage, read_write> errors: array<%s>;

				%s

				@compute @workgroup_size(256, 1, 1)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let idx = global_id.x;
					if (idx >= %du) {
						return;
					}
					var error_sum: %s = %s(0);
					for (var next_idx: u32 = 0u; next_idx < %du; next_idx++) {
						let weight_idx = next_idx * %du + idx;
						error_sum += nextErrors[next_idx] * nextWeights[weight_idx];
					}
					let deriv = activationDerivative(values[idx]);
					errors[idx] = error_sum * deriv;
				}
			`, typ, typ, typ, typ, derivCode, currentSize, typ, typ, nextSize, currentSize)
		}
	}
}

// generateWeightUpdateShader generates a WGSL shader for updating weights and biases in a layer
func (n *Network[T]) generateWeightUpdateShader(layerIdx int) string {
	layer := n.Layers[layerIdx]
	prevLayer := n.Layers[layerIdx-1]
	typ := n.gpu.wgslType

	outputSize := layer.Width * layer.Height
	inputSize := prevLayer.Width * prevLayer.Height

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> prevValues: array<%s>;
		@group(0) @binding(1) var<storage, read> errors: array<%s>;
		@group(0) @binding(2) var<storage, read_write> weights: array<%s>;
		@group(0) @binding(3) var<storage, read_write> biases: array<%s>;
		@group(0) @binding(4) var<uniform> lr: f32;
		@group(0) @binding(5) var<uniform> clipValues: vec2<%s>;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let out_idx = global_id.x;
			let in_idx = global_id.y;
			
			if (out_idx >= %du || in_idx >= %du) {
				return;
			}

			let error = errors[out_idx];
			
			// Update bias (subtract delta)
			if (in_idx == 0u) {
				let deltaBias = %s(lr) * error;
				biases[out_idx] -= deltaBias;
			}

			// Update weight (subtract delta)
			let weight_idx = out_idx * %du + in_idx;
			let input_val = prevValues[in_idx];
			var grad = error * input_val;
			
			// Gradient clipping
			grad = clamp(grad, clipValues.y, clipValues.x);
			
			let deltaWeight = %s(lr) * grad;
			weights[weight_idx] -= deltaWeight;
		}
	`, typ, typ, typ, typ, typ, outputSize, inputSize, typ, inputSize, typ)
}

// Get activation derivative code
func getActivationDerivativeCode(activation, typ string) string {
	switch activation {
	case "relu":
		return fmt.Sprintf(`fn activationDerivative(x: %s) -> %s { 
			if (x > %s(0)) { return %s(1); } else { return %s(0); }
		}`, typ, typ, typ, typ, typ)

	case "leaky_relu":
		if typ == "f32" {
			return `fn activationDerivative(x: f32) -> f32 { 
				if (x > 0.0) { return 1.0; } else { return 0.01; }
			}`
		}
		return fmt.Sprintf(`fn activationDerivative(x: %s) -> %s { 
			if (x > %s(0)) { return %s(1); } else { return %s(1) / %s(100); }
		}`, typ, typ, typ, typ, typ, typ)

	case "sigmoid":
		if typ == "f32" {
			return `fn activationDerivative(x: f32) -> f32 { 
				return x * (1.0 - x);
			}`
		}
		return fmt.Sprintf(`fn activationDerivative(x: %s) -> %s {
			let scale = %s(2147483647);
			let one_minus_x = scale - x;
			return (x * one_minus_x) / scale;
		}`, typ, typ, typ)

	case "tanh":
		if typ == "f32" {
			return `fn activationDerivative(x: f32) -> f32 { 
				return 1.0 - x * x;
			}`
		}
		return fmt.Sprintf(`fn activationDerivative(x: %s) -> %s {
			let scale = %s(2147483647);
			let x_squared = (x * x) / scale;
			return scale - x_squared;
		}`, typ, typ, typ)

	case "elu":
		if typ == "f32" {
			return `fn activationDerivative(x: f32) -> f32 {
				if (x >= 0.0) { return 1.0; }
				return x + 1.0; // Since ELU(x) = exp(x) - 1 for x < 0
			}`
		}
		return fmt.Sprintf(`fn activationDerivative(x: %s) -> %s {
			if (x >= %s(0)) { return %s(1); }
			return x + %s(2147483647); // Approximation
		}`, typ, typ, typ, typ, typ)

	default:
		return fmt.Sprintf(`fn activationDerivative(x: %s) -> %s { return %s(1); }`, typ, typ, typ)
	}
}

// BackwardGPUOptimized with added debugging
func (n *Network[T]) BackwardGPUOptimized(targets [][]float64, lr float64, clipUpper, clipLower T) error {
	if n.gpu.optimized.backward == nil {
		if n.Debug {
			fmt.Println("Initializing backward GPU resources...")
		}
		if err := n.InitializeBackwardGPU(); err != nil {
			return fmt.Errorf("failed to initialize backward GPU: %v", err)
		}
	}

	// Convert targets
	targetData := make([]T, 0, len(targets)*len(targets[0]))
	for _, row := range targets {
		for _, val := range row {
			targetData = append(targetData, T(val))
		}
	}

	// Upload targets
	ctx.queue.WriteBuffer(n.gpu.optimized.backward.targetBuffer, 0, wgpu.ToBytes(targetData))

	// Upload learning rate and clip values
	lrData := []float32{float32(lr)}
	ctx.queue.WriteBuffer(n.gpu.optimized.backward.lrBuffer, 0, wgpu.ToBytes(lrData))
	clipData := []T{clipUpper, clipLower}
	ctx.queue.WriteBuffer(n.gpu.optimized.backward.clipBuffer, 0, wgpu.ToBytes(clipData))

	// Create command encoder
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Backward pass
	for i := 0; i < len(n.gpu.optimized.backward.backwardPipelines); i++ {
		layerIdx := n.OutputLayer - i
		pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("Layer_%d_Backward", layerIdx),
		})
		pass.SetPipeline(n.gpu.optimized.backward.backwardPipelines[i])
		pass.SetBindGroup(0, n.gpu.optimized.backward.backwardBindGroups[i], nil)
		outputSize := n.Layers[layerIdx].Width * n.Layers[layerIdx].Height
		workgroups := (uint32(outputSize) + 255) / 256
		pass.DispatchWorkgroups(workgroups, 1, 1)
		pass.End()
	}

	// Weight update pass with debugging
	for i := 0; i < len(n.gpu.optimized.backward.updatePipelines); i++ {
		layerIdx := n.OutputLayer - i
		pass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("Layer_%d_Update", layerIdx),
		})
		pass.SetPipeline(n.gpu.optimized.backward.updatePipelines[i])
		pass.SetBindGroup(0, n.gpu.optimized.backward.updateBindGroups[i], nil)
		outputSize := n.Layers[layerIdx].Width * n.Layers[layerIdx].Height
		inputSize := n.Layers[layerIdx-1].Width * n.Layers[layerIdx-1].Height
		workgroupsX := (uint32(outputSize) + 15) / 16
		workgroupsY := (uint32(inputSize) + 15) / 16
		pass.DispatchWorkgroups(workgroupsX, workgroupsY, 1)
		pass.End()
	}

	// Submit commands
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish command encoder: %v", err)
	}
	ctx.queue.Submit(commandBuffer)
	ctx.device.Poll(true, nil)

	// Debug: Check if weights are updating
	if n.Debug {
		for l := 1; l <= n.OutputLayer; l++ {
			layerGPU := n.gpu.optimized.layers[l-1]
			weights, err := n.readGPUBuffer(layerGPU.weightBuffer, int(layerGPU.inputSize*layerGPU.outputSize))
			if err != nil {
				fmt.Printf("Debug: Failed to read weights for layer %d: %v\n", l, err)
				continue
			}
			nonZero := 0
			for _, w := range weights {
				if w != 0 {
					nonZero++
				}
			}
			fmt.Printf("Debug: Layer %d weights - Non-zero count: %d / %d\n", l, nonZero, len(weights))
		}
	}

	return nil
}

// SyncGPUWeightsToCPU with validation
func (n *Network[T]) SyncGPUWeightsToCPU() error {
	if n.gpu.optimized == nil || !n.gpu.optimized.initialized {
		return fmt.Errorf("GPU not initialized")
	}

	for l := 1; l <= n.OutputLayer; l++ {
		layerGPU := n.gpu.optimized.layers[l-1]
		layer := &n.Layers[l]

		// Read weights
		weights, err := n.readGPUBuffer(layerGPU.weightBuffer, int(layerGPU.inputSize*layerGPU.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read weights for layer %d: %v", l, err)
		}

		// Read biases
		biases, err := n.readGPUBuffer(layerGPU.biasBuffer, int(layerGPU.outputSize))
		if err != nil {
			return fmt.Errorf("failed to read biases for layer %d: %v", l, err)
		}

		// Validate and apply to CPU neurons
		expectedWeights := int(layerGPU.inputSize * layerGPU.outputSize)
		if len(weights) != expectedWeights {
			return fmt.Errorf("weight size mismatch for layer %d: got %d, expected %d", l, len(weights), expectedWeights)
		}
		if len(biases) != int(layerGPU.outputSize) {
			return fmt.Errorf("bias size mismatch for layer %d: got %d, expected %d", l, len(biases), layerGPU.outputSize)
		}

		for y := 0; y < layer.Height; y++ {
			for x := 0; x < layer.Width; x++ {
				neuron := layer.Neurons[y][x]
				neuronIdx := y*layer.Width + x
				neuron.Bias = biases[neuronIdx]
				weightOffset := neuronIdx * int(layerGPU.inputSize)
				for i := range neuron.Inputs {
					if i < int(layerGPU.inputSize) {
						neuron.Inputs[i].Weight = weights[weightOffset+i]
					}
				}
			}
		}

		if n.Debug {
			fmt.Printf("Synced layer %d weights from GPU to CPU\n", l)
		}

	}

	return nil
}

// Helper to read GPU buffer
func (n *Network[T]) readGPUBuffer(buffer *wgpu.Buffer, size int) ([]T, error) {
	// Create staging buffer
	stagingBuf, err := ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ReadStaging",
		Size:  uint64(size * 4),
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	defer stagingBuf.Destroy()

	// Copy to staging
	encoder, _ := ctx.device.CreateCommandEncoder(nil)
	encoder.CopyBufferToBuffer(buffer, 0, stagingBuf, 0, uint64(size*4))
	cmd, _ := encoder.Finish(nil)
	ctx.queue.Submit(cmd)

	// Map and read
	done := make(chan wgpu.BufferMapAsyncStatus, 1)
	err = stagingBuf.MapAsync(wgpu.MapModeRead, 0, stagingBuf.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, err
	}

	// Wait for mapping
	for {
		ctx.device.Poll(true, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("buffer mapping failed: %v", status)
			}
			goto readData
		default:
			continue
		}
	}

readData:
	data := stagingBuf.GetMappedRange(0, uint(size*4))
	if data == nil {
		return nil, fmt.Errorf("failed to get mapped range")
	}

	result := make([]T, size)
	copy(result, wgpu.FromBytes[T](data))
	stagingBuf.Unmap()

	return result, nil
}

// Backward method with GPU support
func (n *Network[T]) Backward(targets [][]float64, lr float64, clipUpper, clipLower T) {
	if n.WebGPUNative && n.gpu.optimized != nil {
		if err := n.BackwardGPUOptimized(targets, lr, clipUpper, clipLower); err != nil {
			if n.Debug {
				fmt.Printf("GPU backward failed: %v\n", err)
			}
			n.backwardCPU(targets, lr, clipUpper, clipLower)
		}
		return
	}
	n.backwardCPU(targets, lr, clipUpper, clipLower)
}

// Helper function to get max value
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Note: The GPUCompute struct in webgpu_optimized.go needs to be updated to include:
// backward    *GPUBackwardResources // Add this field to the struct

// Cleanup backward GPU resources
func (n *Network[T]) CleanupBackwardGPU() {
	if n.gpu.optimized == nil || n.gpu.optimized.backward == nil {
		return
	}

	backward := n.gpu.optimized.backward

	// Clean up buffers
	for _, buf := range backward.errorBuffers {
		if buf != nil {
			buf.Destroy()
		}
	}
	for _, buf := range backward.gradientBuffers {
		if buf != nil {
			buf.Destroy()
		}
	}
	if backward.targetBuffer != nil {
		backward.targetBuffer.Destroy()
	}
	if backward.lrBuffer != nil {
		backward.lrBuffer.Destroy()
	}
	if backward.clipBuffer != nil {
		backward.clipBuffer.Destroy()
	}

	// Clean up pipelines
	for _, pipeline := range backward.backwardPipelines {
		if pipeline != nil {
			pipeline.Release()
		}
	}
	for _, pipeline := range backward.updatePipelines {
		if pipeline != nil {
			pipeline.Release()
		}
	}

	// Clean up bind groups
	for _, bg := range backward.backwardBindGroups {
		if bg != nil {
			bg.Release()
		}
	}
	for _, bg := range backward.updateBindGroups {
		if bg != nil {
			bg.Release()
		}
	}

	n.gpu.optimized.backward = nil

	if n.Debug {
		fmt.Println("Cleaned up backward GPU resources")
	}
}

// Modified Train method to sync GPU weights after each epoch
func (n *Network[T]) TrainWithGPUSync(
	inputs [][][]float64,
	targets [][][]float64,
	epochs int,
	learningRate float64,
	earlyStopOnNegativeLoss bool,
	clipUpper T,
	clipLower T,
) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for b := 0; b < len(inputs); b++ {
			// Forward pass on GPU
			n.Forward(inputs[b])
			loss := n.ComputeLoss(targets[b])
			if math.IsNaN(loss) {
				fmt.Printf("NaN loss detected at sample %d, epoch %d\n", b, epoch)
				continue
			}
			if earlyStopOnNegativeLoss && loss < 0 {
				fmt.Printf("⚠️ Negative loss (%.4f) detected at sample %d, epoch %d. Stopping training early.\n", loss, b, epoch)
				return
			}
			totalLoss += loss

			// Backward pass on GPU
			n.Backward(targets[b], learningRate, clipUpper, clipLower)
		}

		// Sync GPU weights to CPU after each epoch
		if n.WebGPUNative && n.gpu.optimized != nil {
			//fmt.Println("COPY WEIGHTS FROM GPU TO CPU")
			if err := n.SyncGPUWeightsToCPU(); err != nil {
				fmt.Printf("Failed to sync GPU weights at epoch %d: %v\n", epoch, err)
			}
		}

		fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(inputs)))
	}
}

func (n *Network[T]) SyncCPUWeightsToGPU() error {
	if n.gpu.optimized == nil || !n.gpu.optimized.initialized {
		return fmt.Errorf("GPU not initialized")
	}

	for l := 1; l <= n.OutputLayer; l++ {
		layerGPU := n.gpu.optimized.layers[l-1]
		weights, biases := n.extractLayerWeightsAndBiases(l)

		// Write weights with error checking
		if err := ctx.queue.WriteBuffer(layerGPU.weightBuffer, 0, wgpu.ToBytes(weights)); err != nil {
			return fmt.Errorf("failed to write weights to GPU for layer %d: %v", l, err)
		}

		// Write biases with error checking
		if err := ctx.queue.WriteBuffer(layerGPU.biasBuffer, 0, wgpu.ToBytes(biases)); err != nil {
			return fmt.Errorf("failed to write biases to GPU for layer %d: %v", l, err)
		}

		//if n.Debug {
		fmt.Printf("Synced CPU weights to GPU for layer %d\n", l)
		//}

		if n.Debug {
			sampleWeights := weights[:min(5, len(weights))]
			fmt.Printf("Layer %d - CPU weights: %v\n", l, sampleWeights)
			gpuWeights, _ := n.readGPUBuffer(layerGPU.weightBuffer, 5)
			fmt.Printf("Layer %d - GPU weights: %v\n", l, gpuWeights)
		}
	}

	return nil
}
