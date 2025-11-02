// webgpu_backend_detector.go
package paragon

import (
	"fmt"
	"reflect"

	"github.com/openfluke/webgpu/wgpu"
)

/*
INTEGRATION INSTRUCTIONS:

To integrate this backend detector into your existing code:

1. In webgpu_optimized.go, replace the InitializeOptimizedGPU function with:
   - Change line 66 from: layerCompute, err := n.createLayerCompute(l)
   - To: layerCompute, err := n.CreateLayerComputeWithBackend(l)

2. Or, simply replace calls to InitializeOptimizedGPU() with InitializeOptimizedGPUWithBackendDetection()

This will enable automatic detection and routing between dense and attention compute pipelines.
*/

// BackendType identifies the compute type for a layer
type BackendType int

const (
	BackendDense BackendType = iota
	BackendAttn
	BackendMixed
)

// LayerBackendInfo stores backend selection for a layer
type LayerBackendInfo struct {
	Type         BackendType
	DenseColumns []int
	AttnColumns  []int
}

// DetectLayerBackend determines the appropriate backend for a layer
func (n *Network[T]) DetectLayerBackend(layerIdx int) LayerBackendInfo {
	if layerIdx == 0 {
		// Input layer has no computation
		return LayerBackendInfo{Type: BackendDense}
	}

	layer := n.Layers[layerIdx]
	info := LayerBackendInfo{
		DenseColumns: []int{},
		AttnColumns:  []int{},
	}

	// Check if layer has attention config and slice types
	hasAttnConfig := layer.Attn != nil && layer.Attn.DK > 0
	hasSliceTypes := len(layer.SliceTypes) > 0

	if hasAttnConfig && hasSliceTypes {
		// Mixed layer: classify each column
		for col := 0; col < layer.Width; col++ {
			if col < len(layer.SliceTypes) && layer.SliceTypes[col] == "attn" {
				info.AttnColumns = append(info.AttnColumns, col)
			} else {
				info.DenseColumns = append(info.DenseColumns, col)
			}
		}

		if len(info.AttnColumns) > 0 && len(info.DenseColumns) > 0 {
			info.Type = BackendMixed
		} else if len(info.AttnColumns) > 0 {
			info.Type = BackendAttn
		} else {
			info.Type = BackendDense
		}
	} else {
		// Pure dense layer
		info.Type = BackendDense
		for col := 0; col < layer.Width; col++ {
			info.DenseColumns = append(info.DenseColumns, col)
		}
	}

	return info
}

// CreateLayerComputeWithBackend creates the appropriate compute pipeline based on backend type
func (n *Network[T]) CreateLayerComputeWithBackend(layerIdx int) (*GPULayerCompute, error) {
	backend := n.DetectLayerBackend(layerIdx)

	prevLayer := n.Layers[layerIdx-1]
	currentLayer := n.Layers[layerIdx]

	inputSize := uint32(prevLayer.Width * prevLayer.Height)
	outputSize := uint32(currentLayer.Width * currentLayer.Height)

	if inputSize == 0 || outputSize == 0 {
		return nil, fmt.Errorf("invalid layer dimensions: input=%d, output=%d", inputSize, outputSize)
	}

	layerCompute := &GPULayerCompute{
		inputSize:    inputSize,
		outputSize:   outputSize,
		layerIndex:   layerIdx,
		hasMixed:     backend.Type == BackendMixed,
		denseColumns: backend.DenseColumns,
		attnColumns:  backend.AttnColumns,
	}

	// Create attention column computes if needed
	if backend.Type == BackendMixed || backend.Type == BackendAttn {
		layerCompute.attnComputes = make(map[int]*GPUAttnColumnCompute)
		for _, col := range backend.AttnColumns {
			attnComp, err := n.createAttnColumnCompute(layerIdx, col)
			if err != nil {
				layerCompute.cleanup()
				return nil, fmt.Errorf("failed to create attention compute for col %d: %v", col, err)
			}
			layerCompute.attnComputes[col] = attnComp
		}
	}

	// Create dense compute pipeline if needed
	if backend.Type == BackendDense || backend.Type == BackendMixed {
		if err := n.createDenseComputePipeline(layerCompute, layerIdx, inputSize, outputSize); err != nil {
			layerCompute.cleanup()
			return nil, err
		}
	} else {
		// Pure attention - create trivial pipeline
		if err := n.createTrivialPipeline(layerCompute, layerIdx); err != nil {
			layerCompute.cleanup()
			return nil, err
		}
	}

	// Create buffers
	if err := n.createLayerBuffers(layerCompute, layerIdx, inputSize, outputSize); err != nil {
		layerCompute.cleanup()
		return nil, err
	}

	return layerCompute, nil
}

// createDenseComputePipeline creates the dense computation pipeline
func (n *Network[T]) createDenseComputePipeline(lc *GPULayerCompute, layerIdx int, inputSize, outputSize uint32) error {
	shaderCode := n.generateDenseLayerShader(layerIdx, inputSize, outputSize)

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Dense_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return fmt.Errorf("failed to create shader module: %v", err)
	}
	defer module.Release()

	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("Layer_%d_Dense_BindGroupLayout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return err
	}
	lc.bindGroupLayout = bindGroupLayout

	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_Dense_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return err
	}
	defer pipelineLayout.Release()

	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Dense_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return err
	}
	lc.pipeline = pipeline

	// Calculate workgroups
	lc.workgroupsX = (outputSize + 255) / 256
	lc.workgroupsY = 1

	return nil
}

// createTrivialPipeline creates a no-op pipeline for pure attention layers
func (n *Network[T]) createTrivialPipeline(lc *GPULayerCompute, layerIdx int) error {
	shaderCode := `@compute @workgroup_size(1) fn main() {}`

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Trivial_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return fmt.Errorf("failed to create trivial shader module: %v", err)
	}
	defer module.Release()

	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label:   fmt.Sprintf("Layer_%d_Trivial_BindGroupLayout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{},
	})
	if err != nil {
		return err
	}
	lc.bindGroupLayout = bindGroupLayout

	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_Trivial_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return err
	}
	defer pipelineLayout.Release()

	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Trivial_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		return err
	}
	lc.pipeline = pipeline

	lc.workgroupsX = 1
	lc.workgroupsY = 1

	return nil
}

// createLayerBuffers creates all GPU buffers for a layer
func (n *Network[T]) createLayerBuffers(lc *GPULayerCompute, layerIdx int, inputSize, outputSize uint32) error {
	var err error

	// Input buffer
	lc.inputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Input", layerIdx),
		Size:  uint64(inputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create input buffer: %v", err)
	}

	// Output buffer
	lc.outputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Output", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Staging buffer
	lc.stagingBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Staging", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create staging buffer: %v", err)
	}

	// Weight and bias buffers (for dense computation)
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)

	lc.weightBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Weights", layerIdx),
		Contents: wgpu.ToBytes(weights),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create weight buffer: %v", err)
	}

	lc.biasBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Biases", layerIdx),
		Contents: wgpu.ToBytes(biases),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return fmt.Errorf("failed to create bias buffer: %v", err)
	}

	// Create bind group
	lc.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("Layer_%d_BindGroup", layerIdx),
		Layout: lc.bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: lc.inputBuffer, Size: lc.inputBuffer.GetSize()},
			{Binding: 1, Buffer: lc.outputBuffer, Size: lc.outputBuffer.GetSize()},
			{Binding: 2, Buffer: lc.weightBuffer, Size: lc.weightBuffer.GetSize()},
			{Binding: 3, Buffer: lc.biasBuffer, Size: lc.biasBuffer.GetSize()},
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create bind group: %v", err)
	}

	return nil
}

// UpdateInitializeOptimizedGPU integrates backend detection into InitializeOptimizedGPU
// This should replace the original InitializeOptimizedGPU in webgpu_optimized.go
func (n *Network[T]) InitializeOptimizedGPUWithBackendDetection() error {
	typ := reflect.TypeOf(*new(T)).Kind()

	switch typ {
	case reflect.Float32, reflect.Int32, reflect.Uint32:
		// Supported GPU types
	default:
		return fmt.Errorf("❌ WebGPU acceleration only supports f32, i32, or u32 numeric types — got: %v", typ)
	}

	ensureGPU()

	n.gpu.optimized = &GPUCompute{
		device: ctx.device,
		queue:  ctx.queue,
		debug:  n.Debug,
	}

	// Create layer-specific compute pipelines using backend detection
	for l := 1; l <= n.OutputLayer; l++ {
		layerCompute, err := n.CreateLayerComputeWithBackend(l)
		if err != nil {
			return fmt.Errorf("failed to create compute for layer %d: %v", l, err)
		}
		n.gpu.optimized.layers = append(n.gpu.optimized.layers, layerCompute)
	}

	n.gpu.optimized.initialized = true
	return nil
}
