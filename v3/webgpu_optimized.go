// webgpu_optimized.go
package paragon

import (
	"fmt"
	"reflect"

	"github.com/openfluke/webgpu/wgpu"
)

// GPULayerCompute represents a single layer's GPU computation
type GPULayerCompute struct {
	pipeline        *wgpu.ComputePipeline
	bindGroup       *wgpu.BindGroup
	bindGroupLayout *wgpu.BindGroupLayout
	inputBuffer     *wgpu.Buffer
	outputBuffer    *wgpu.Buffer
	weightBuffer    *wgpu.Buffer
	biasBuffer      *wgpu.Buffer
	stagingBuffer   *wgpu.Buffer
	workgroupsX     uint32
	workgroupsY     uint32
	inputSize       uint32
	outputSize      uint32
	layerIndex      int

	// For mixed layers with attention
	hasMixed      bool
	denseColumns  []int                         // Column indices that are dense
	attnColumns   []int                         // Column indices that are attention
	attnComputes  map[int]*GPUAttnColumnCompute // Per-attention-column compute
	columnOutputs *wgpu.Buffer                  // Staging for per-column outputs
}

// GPUCompute manages optimized GPU neural network computation
type GPUCompute struct {
	device      *wgpu.Device
	queue       *wgpu.Queue
	layers      []*GPULayerCompute
	initialized bool
	debug       bool
	backward    *GPUBackwardResources
}

// Initialize GPU compute for the network
func (n *Network[T]) InitializeOptimizedGPU() error {
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

	// Create layer-specific compute pipelines
	for l := 1; l <= n.OutputLayer; l++ {
		layerCompute, err := n.createLayerCompute(l)
		if err != nil {
			return fmt.Errorf("failed to create compute for layer %d: %v", l, err)
		}
		n.gpu.optimized.layers = append(n.gpu.optimized.layers, layerCompute)
	}

	n.gpu.optimized.initialized = true
	return nil
}

// Create GPU compute resources for a single layer
func (n *Network[T]) createLayerCompute(layerIdx int) (*GPULayerCompute, error) {
	prevLayer := n.Layers[layerIdx-1]
	currentLayer := n.Layers[layerIdx]

	inputSize := uint32(prevLayer.Width * prevLayer.Height)
	outputSize := uint32(currentLayer.Width * currentLayer.Height)

	if inputSize == 0 || outputSize == 0 {
		return nil, fmt.Errorf("invalid layer dimensions: input=%d, output=%d", inputSize, outputSize)
	}

	// Check if this is a mixed layer
	hasMixed := false
	denseColumns := []int{}
	attnColumns := []int{}

	if currentLayer.Attn != nil && len(currentLayer.SliceTypes) > 0 {
		for col, st := range currentLayer.SliceTypes {
			if st == "attn" {
				hasMixed = true
				attnColumns = append(attnColumns, col)
			} else {
				denseColumns = append(denseColumns, col)
			}
		}
	} else {
		// All dense
		for col := 0; col < currentLayer.Width; col++ {
			denseColumns = append(denseColumns, col)
		}
	}

	layerCompute := &GPULayerCompute{
		inputSize:    inputSize,
		outputSize:   outputSize,
		layerIndex:   layerIdx,
		hasMixed:     hasMixed,
		denseColumns: denseColumns,
		attnColumns:  attnColumns,
	}

	if hasMixed {
		// Create attention column computes
		layerCompute.attnComputes = make(map[int]*GPUAttnColumnCompute)
		for _, col := range attnColumns {
			attnComp, err := n.createAttnColumnCompute(layerIdx, col)
			if err != nil {
				layerCompute.cleanup()
				return nil, fmt.Errorf("failed to create attention compute for col %d: %v", col, err)
			}
			layerCompute.attnComputes[col] = attnComp
		}
	}

	// Create shader for dense portion (or entire layer if no attention)
	var shaderCode string
	if len(denseColumns) > 0 {
		shaderCode = n.generateDenseLayerShader(layerIdx, inputSize, outputSize)
	} else {
		// No dense columns - use a trivial shader
		shaderCode = `
@compute @workgroup_size(1)
fn main() {}
`
	}

	// Check ctx.device before use
	if ctx.device == nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("WebGPU device not initialized for layer %d", layerIdx)
	}

	// Create shader module
	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Shader", layerIdx),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create shader module: %v", err)
	}
	defer module.Release() // Clean up after pipeline creation

	// Create bind group layout first
	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("Layer_%d_BindGroupLayout", layerIdx),
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    1,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    2,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
			{
				Binding:    3,
				Visibility: wgpu.ShaderStageCompute,
				Buffer: wgpu.BufferBindingLayout{
					Type:             wgpu.BufferBindingTypeReadOnlyStorage,
					HasDynamicOffset: false,
				},
			},
		},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create bind group layout: %v", err)
	}

	layerCompute.bindGroupLayout = bindGroupLayout

	// Create pipeline layout
	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_PipelineLayout", layerIdx),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create pipeline layout: %v", err)
	}
	defer pipelineLayout.Release() // Clean up after pipeline creation

	// Create compute pipeline
	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Pipeline", layerIdx),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create compute pipeline: %v", err)
	}

	layerCompute.pipeline = pipeline

	// Create buffers
	// Input buffer
	layerCompute.inputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Input", layerIdx),
		Size:  uint64(inputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create input buffer: %v", err)
	}

	// Output buffer
	layerCompute.outputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Output", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create output buffer: %v", err)
	}

	// Staging buffer for CPU readback
	layerCompute.stagingBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Staging", layerIdx),
		Size:  uint64(outputSize) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create staging buffer: %v", err)
	}

	// Prepare weight and bias data (for dense columns only)
	weights, biases := n.extractLayerWeightsAndBiases(layerIdx)

	// Weight buffer
	layerCompute.weightBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Weights", layerIdx),
		Contents: wgpu.ToBytes(weights),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create weight buffer: %v", err)
	}

	// Bias buffer
	layerCompute.biasBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Biases", layerIdx),
		Contents: wgpu.ToBytes(biases),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create bias buffer: %v", err)
	}

	// Create bind group
	layerCompute.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("Layer_%d_BindGroup", layerIdx),
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: layerCompute.inputBuffer, Size: layerCompute.inputBuffer.GetSize()},
			{Binding: 1, Buffer: layerCompute.outputBuffer, Size: layerCompute.outputBuffer.GetSize()},
			{Binding: 2, Buffer: layerCompute.weightBuffer, Size: layerCompute.weightBuffer.GetSize()},
			{Binding: 3, Buffer: layerCompute.biasBuffer, Size: layerCompute.biasBuffer.GetSize()},
		},
	})
	if err != nil {
		layerCompute.cleanup()
		return nil, fmt.Errorf("failed to create bind group: %v", err)
	}

	// Calculate optimal workgroup dimensions
	layerCompute.workgroupsX = (outputSize + 255) / 256 // 256 threads per workgroup
	layerCompute.workgroupsY = 1

	if n.Debug {
		fmt.Printf("Created layer %d compute: %dx%d -> %dx%d, dense cols: %v, attn cols: %v, workgroups: %d\n",
			layerIdx, prevLayer.Width, prevLayer.Height,
			currentLayer.Width, currentLayer.Height, denseColumns, attnColumns, layerCompute.workgroupsX)
	}

	return layerCompute, nil
}

// Helper method to clean up a layer's resources
func (lc *GPULayerCompute) cleanup() {
	if lc.inputBuffer != nil {
		lc.inputBuffer.Destroy()
	}
	if lc.outputBuffer != nil {
		lc.outputBuffer.Destroy()
	}
	if lc.weightBuffer != nil {
		lc.weightBuffer.Destroy()
	}
	if lc.biasBuffer != nil {
		lc.biasBuffer.Destroy()
	}
	if lc.stagingBuffer != nil {
		lc.stagingBuffer.Destroy()
	}
	if lc.columnOutputs != nil {
		lc.columnOutputs.Destroy()
	}
	if lc.bindGroup != nil {
		lc.bindGroup.Release()
	}
	if lc.bindGroupLayout != nil {
		lc.bindGroupLayout.Release()
	}
	if lc.pipeline != nil {
		lc.pipeline.Release()
	}
	// Clean up attention computes
	if lc.attnComputes != nil {
		for _, ac := range lc.attnComputes {
			ac.cleanup()
		}
	}
}

// Generate optimized shader for a specific layer
func (n *Network[T]) generateLayerShader(layerIdx int, inputSize, outputSize uint32) string {
	curr := n.Layers[layerIdx]

	// Check if this layer has any attention columns
	hasAttn := false
	if curr.Attn != nil && len(curr.SliceTypes) > 0 {
		for _, st := range curr.SliceTypes {
			if st == "attn" {
				hasAttn = true
				break
			}
		}
	}

	if !hasAttn {
		// Pure dense layer
		return n.generateDenseLayerShader(layerIdx, inputSize, outputSize)
	}

	// Mixed layer with attention
	return n.generateMixedLayerShader(layerIdx, inputSize, outputSize)
}

// Generate shader for pure dense layer
func (n *Network[T]) generateDenseLayerShader(layerIdx int, inputSize, outputSize uint32) string {
	currentLayer := n.Layers[layerIdx]
	activation := currentLayer.Neurons[0][0].Activation
	typ := n.gpu.wgslType
	if typ == "" {
		typ = getWGSLType[T]()
		if typ == "" {
			typ = "f32" // final fallback
		}
	}
	activationCode := getActivationCode(activation, typ)

	return fmt.Sprintf(`
		@group(0) @binding(0) var<storage, read> input: array<%s>;
		@group(0) @binding(1) var<storage, read_write> output: array<%s>;
		@group(0) @binding(2) var<storage, read> weights: array<%s>;
		@group(0) @binding(3) var<storage, read> biases: array<%s>;

		%s

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
			let output_idx = global_id.x;

			if (output_idx >= %du) {
				return;
			}

			var sum: %s = biases[output_idx];
			let weight_offset = output_idx * %du;
			for (var i: u32 = 0u; i < %du; i++) {
				sum += weights[weight_offset + i] * input[i];
			}

			output[output_idx] = activate(sum);
		}
	`, typ, typ, typ, typ, activationCode, outputSize, typ, inputSize, inputSize)
}

// Generate shader for mixed dense+attention layer
func (n *Network[T]) generateMixedLayerShader(layerIdx int, inputSize, outputSize uint32) string {
	curr := n.Layers[layerIdx]
	prev := n.Layers[layerIdx-1]
	cfg := curr.Attn

	if cfg == nil {
		return n.generateDenseLayerShader(layerIdx, inputSize, outputSize)
	}

	heads := cfg.Heads
	if heads <= 0 {
		heads = 1
	}
	dk := cfg.DK
	hd := heads * dk
	prevW := prev.Width
	prevH := prev.Height
	N := prevW * prevH

	posEncAmp := cfg.PosEncAmp
	if posEncAmp == 0 {
		posEncAmp = 0.02
	}
	normEps := cfg.NormEps
	if normEps == 0 {
		normEps = 1e-6
	}

	activation := curr.Neurons[0][0].Activation
	typ := n.gpu.wgslType
	activationCode := getActivationCode(activation, typ)

	// Build shader that handles column types
	shader := fmt.Sprintf(`
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;

%s

const PREV_W: u32 = %du;
const PREV_H: u32 = %du;
const N: u32 = %du;
const HEADS: u32 = %du;
const DK: u32 = %du;
const HD: u32 = %du;
const CURR_H: u32 = %du;
const POS_ENC_AMP: f32 = %f;
const NORM_EPS: f32 = %f;
const USE_NORM: u32 = %du;
const USE_WO: u32 = %du;
const CURR_W: u32 = %du;

fn dot_product(a: ptr<function, array<f32, DK>>, b: ptr<function, array<f32, DK>>) -> f32 {
    var sum = 0.0;
    for (var t = 0u; t < DK; t++) {
        sum += (*a)[t] * (*b)[t];
    }
    return sum;
}

fn softmax_row(row: ptr<function, array<f32, N>>) {
    var mx = (*row)[0];
    for (var j = 1u; j < N; j++) {
        mx = max(mx, (*row)[j]);
    }
    var sum = 0.0;
    for (var j = 0u; j < N; j++) {
        let e = exp((*row)[j] - mx);
        (*row)[j] = e;
        sum += e;
    }
    if (sum > 0.0) {
        let inv = 1.0 / sum;
        for (var j = 0u; j < N; j++) {
            (*row)[j] *= inv;
        }
    }
}

// Process attention column
fn process_attn_column(col: u32, weight_offset: u32) -> f32 {
    // 1) Build tokens with positional encoding
    var X: array<f32, N>;
    for (var i = 0u; i < N; i++) {
        X[i] = input[i];
    }
    `, activationCode, prevW, prevH, N, heads, dk, hd, curr.Height, posEncAmp, normEps,
		boolToU32(cfg.UseNorm), boolToU32(cfg.UseWo), curr.Width)

	if cfg.PosEnc2D {
		shader += `
    // Add 2D positional encoding
    for (var y = 0u; y < PREV_H; y++) {
        for (var x = 0u; x < PREV_W; x++) {
            let i = y * PREV_W + x;
            let fy = f32(y) / f32(max(1u, PREV_H - 1u));
            let fx = f32(x) / f32(max(1u, PREV_W - 1u));
            X[i] += POS_ENC_AMP * (sin(6.283185 * fy) + cos(6.283185 * fx));
        }
    }
`
	}

	shader += fmt.Sprintf(`
    // 2) Multi-head attention
    var zbar_cat: array<f32, HD>;
    var head_offset = 0u;
    
    for (var h = 0u; h < HEADS; h++) {
        // Q, K, V projections for this head
        var Q: array<array<f32, DK>, N>;
        var K: array<array<f32, DK>, N>;
        var V: array<array<f32, DK>, N>;
        
        let wq_base = weight_offset + h * DK;
        let wk_base = weight_offset + HEADS * DK + h * DK;
        let wv_base = weight_offset + 2u * HEADS * DK + h * DK;
        
        for (var i = 0u; i < N; i++) {
            let x = X[i];
            for (var t = 0u; t < DK; t++) {
                Q[i][t] = x * weights[wq_base + t];
                K[i][t] = x * weights[wk_base + t];
                V[i][t] = x * weights[wv_base + t];
            }
        }
        
        // Attention scores
        let scale = 1.0 / sqrt(f32(DK));
        var A: array<array<f32, N>, N>;
        for (var i = 0u; i < N; i++) {
            for (var j = 0u; j < N; j++) {
                A[i][j] = scale * dot_product(&Q[i], &K[j]);
            }
            softmax_row(&A[i]);
        }
        
        // Z = A * V and pool to zbar_h
        var zbar_h: array<f32, DK>;
        for (var i = 0u; i < N; i++) {
            var zi: array<f32, DK>;
            for (var j = 0u; j < N; j++) {
                let a = A[i][j];
                if (a != 0.0) {
                    for (var t = 0u; t < DK; t++) {
                        zi[t] += a * V[j][t];
                    }
                }
            }
            for (var t = 0u; t < DK; t++) {
                zbar_h[t] += zi[t];
            }
        }
        
        let invN = 1.0 / f32(N);
        for (var t = 0u; t < DK; t++) {
            zbar_h[t] *= invN;
            zbar_cat[head_offset + t] = zbar_h[t];
        }
        head_offset += DK;
    }
    
    // 3) Optional normalization
    if (USE_NORM == 1u) {
        var mu = 0.0;
        for (var t = 0u; t < HD; t++) {
            mu += zbar_cat[t];
        }
        mu /= f32(HD);
        
        var vv = 0.0;
        for (var t = 0u; t < HD; t++) {
            let d = zbar_cat[t] - mu;
            vv += d * d;
        }
        vv /= f32(HD);
        let inv_std = 1.0 / sqrt(vv + NORM_EPS);
        
        for (var t = 0u; t < HD; t++) {
            zbar_cat[t] = (zbar_cat[t] - mu) * inv_std;
        }
    }
    
    // 4) Return value for this output neuron (y)
    // This function is called per output neuron, so we compute just one y value
    return 0.0; // Placeholder - will be replaced below
}

// Process dense column
fn process_dense_column(col: u32, y: u32, weight_offset: u32) -> f32 {
    var sum = biases[y * CURR_W + col];
    let w_base = weight_offset + y * N;
    for (var i = 0u; i < N; i++) {
        sum += input[i] * weights[w_base + i];
    }
    return sum;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    if (output_idx >= %du) { return; }
    
    let y = output_idx / CURR_W;
    let col = output_idx %% CURR_W;
    
    // Determine column type and process
    // Column types are encoded in weights buffer metadata
    // For simplicity, we'll use a convention: first %d cols are according to SliceTypes
    
    var value = 0.0;
    
    // TODO: Need to encode column type in buffer or use separate shader per column
    // For now, stub implementation
    value = process_dense_column(col, y, 0u);
    
    output[output_idx] = activate(value);
}
`, outputSize, len(curr.SliceTypes))

	return shader
}

// relu,sigmoid,tanh,leaky_relu,elu,linear
// Get activation function WGSL code
func getActivationCode(activation, typ string) string {
	switch activation {
	case "relu":
		return fmt.Sprintf(`fn activate(x: %s) -> %s { return max(%s(0), x); }`, typ, typ, typ)

	case "leaky_relu":
		if typ == "f32" {
			return `fn activate(x: f32) -> f32 { return select(0.01 * x, x, x > 0.0); }`
		}
		if typ == "u32" {
			return `fn activate(x: u32) -> u32 { return x; }`
		}
		// Pure integer leaky ReLU matching CPU
		return `fn activate(x: i32) -> i32 {
			if (x >= i32(0)) { 
				return x; 
			}
			// Pure integer 1% leak: x / 100
			var leak = x / i32(100);
			if (leak == i32(0) && x < i32(0)) {
				leak = i32(-1); // Minimum leak for negative values
			}
			return leak;
		}`

	case "elu":
		if typ == "f32" {
			return `fn activate(x: f32) -> f32 {
				if (x >= 0.0) { return x; }
				return exp(max(x, -10.0)) - 1.0;
			}`
		}
		if typ == "u32" {
			return `fn activate(x: u32) -> u32 { return x; }`
		}
		// Pure integer ELU approximation matching CPU
		return `fn activate(x: i32) -> i32 {
			if (x >= i32(0)) { 
				return x; 
			}
			let scale = i32(2147483647);
			if (x <= -scale) {
				return -scale; // Cap at -1.0 in fixed point
			}
			// Simple approximation: ELU(x) ≈ x/2 for negative x
			return x / i32(2);
		}`

	case "tanh":
		// 1) the single piecewise approximation
		const tanhApprox = `
fn tanh_approx(input: f32) -> f32 {
    if (input >  1.0) { return  1.0; }
    if (input < -1.0) { return -1.0; }
    if (input >= 0.0) {
        if (input < 0.25) { return input; }
        let denom = 1.0 + input * 2.0;
        return 1.0 - 2.0 / denom;
    } else {
        if (input > -0.25) { return input; }
        let abs_input = -input;
        let denom = 1.0 + abs_input * 2.0;
        return -1.0 + 2.0 / denom;
    }
}
`

		// 2) f32 is trivial
		if typ == "f32" {
			return tanhApprox + `
fn activate(x: f32) -> f32 {
    return tanh_approx(x);
}`
		}

		// 3) i32: cast to f32, approx, cast back (truncates toward zero exactly like Go’s conversion)
		if typ == "i32" || typ == "int32" {
			return tanhApprox + `
fn activate(x: i32) -> i32 {
    return i32(tanh_approx(f32(x)));
}`
		}

		// 4) u32: cast to f32, approx, clamp negative→0, cast back
		if typ == "u32" || typ == "uint32" {
			return tanhApprox + `
fn activate(x: u32) -> u32 {
    let t = tanh_approx(f32(x));
    // clamp any negative result to zero before casting
    return u32(max(t, 0.0));
}`
		}

		// 5) everything else falls back to identity
		return fmt.Sprintf(`fn activate(x: %s) -> %s { return x; }`, typ, typ)

	case "sigmoid":
		if typ == "f32" {
			return `fn activate(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`
		}
		if typ == "u32" {
			return `fn activate(x: u32) -> u32 {
				let scaled = f32(x) / f32(2147483647);
				let s = 1.0 / (1.0 + exp(-scaled * 0.5));
				return u32(round(s * f32(2147483647) * 2.0));
			}`
		}
		return `fn activate(x: i32) -> i32 {
			let scaled = f32(x) / f32(2147483647);
			let s = 1.0 / (1.0 + exp(-scaled));
			return i32(round(s * f32(2147483647)));
		}`

	default:
		return fmt.Sprintf(`fn activate(x: %s) -> %s { return x; }`, typ, typ)
	}
}

// Extract weights and biases for a specific layer
func (n *Network[T]) extractLayerWeightsAndBiases(layerIdx int) ([]T, []T) {
	currentLayer := n.Layers[layerIdx]
	prevLayer := n.Layers[layerIdx-1]

	inputSize := prevLayer.Width * prevLayer.Height
	outputSize := currentLayer.Width * currentLayer.Height

	weights := make([]T, outputSize*inputSize)
	biases := make([]T, outputSize)

	for y := 0; y < currentLayer.Height; y++ {
		for x := 0; x < currentLayer.Width; x++ {
			neuronIdx := y*currentLayer.Width + x
			neuron := currentLayer.Neurons[y][x]

			// Extract bias
			biases[neuronIdx] = T(any(neuron.Bias).(T))

			// Extract weights in proper order for matrix multiplication
			weightOffset := neuronIdx * inputSize
			for i, conn := range neuron.Inputs {
				if i < inputSize {
					weights[weightOffset+i] = T(any(conn.Weight).(T))
				}
			}
		}
	}

	return weights, biases
}

// Optimized forward pass using proper GPU parallelization
func (n *Network[T]) ForwardGPUOptimized(inputs [][]float64) error {
	if !n.gpu.optimized.initialized {
		return fmt.Errorf("optimized GPU not initialized")
	}

	// Check if any layer has mixed columns - if so, fall back to CPU
	// TODO: Implement proper per-column dense compute for mixed layers
	for _, lc := range n.gpu.optimized.layers {
		if lc.hasMixed {
			// Fall back to CPU for networks with mixed layers
			n.forwardCPU(inputs)
			return nil
		}
	}

	var inputData []T
	inputData = make([]T, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, T(val))
		}
	}

	// Create command encoder
	encoder, err := ctx.device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("failed to create command encoder: %v", err)
	}

	// Process each layer sequentially
	for i, layerCompute := range n.gpu.optimized.layers {
		// Upload input data to GPU
		ctx.queue.WriteBuffer(layerCompute.inputBuffer, 0, wgpu.ToBytes(inputData))

		// Pure dense layer - process on GPU
		computePass := encoder.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("Layer_%d_Compute", i+1),
		})
		computePass.SetPipeline(layerCompute.pipeline)
		computePass.SetBindGroup(0, layerCompute.bindGroup, nil)
		computePass.DispatchWorkgroups(layerCompute.workgroupsX, layerCompute.workgroupsY, 1)
		computePass.End()

		// Copy output to staging buffer for readback
		encoder.CopyBufferToBuffer(
			layerCompute.outputBuffer, 0,
			layerCompute.stagingBuffer, 0,
			uint64(layerCompute.outputSize)*4,
		)

		// For next iteration, copy output to next layer's input
		if i < len(n.gpu.optimized.layers)-1 {
			nextLayerCompute := n.gpu.optimized.layers[i+1]
			encoder.CopyBufferToBuffer(
				layerCompute.outputBuffer, 0,
				nextLayerCompute.inputBuffer, 0,
				uint64(layerCompute.outputSize)*4,
			)
		}
	}

	// Submit all commands at once
	commandBuffer, err := encoder.Finish(nil)
	if err != nil {
		return fmt.Errorf("failed to finish command encoder: %v", err)
	}

	ctx.queue.Submit(commandBuffer)

	// Wait for completion
	ctx.device.Poll(true, nil)

	// Handle final layer output
	finalLayer := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	finalOutput, err := n.readStagingBuffer(finalLayer.stagingBuffer, int(finalLayer.outputSize))
	if err != nil {
		return fmt.Errorf("failed to read final output: %v", err)
	}

	// Check if output layer uses softmax
	if n.Layers[n.OutputLayer].Neurons[0][0].Activation == "softmax" {
		// Convert pre-activation values to float64 for softmax computation
		preActFloat := make([]float64, len(finalOutput))
		for i, v := range finalOutput {
			preActFloat[i] = float64(v)
		}

		// Apply softmax
		postActFloat := Softmax(preActFloat)

		// Convert back to type T
		postAct := make([]T, len(postActFloat))
		for i, v := range postActFloat {
			postAct[i] = T(v)
		}

		// Write post-activation values back to GPU output buffer
		ctx.queue.WriteBuffer(finalLayer.outputBuffer, 0, wgpu.ToBytes(postAct))

		// Update neuron values with post-activation values
		outputLayer := &n.Layers[n.OutputLayer]
		idx := 0
		for y := 0; y < outputLayer.Height; y++ {
			for x := 0; x < outputLayer.Width; x++ {
				if idx < len(postAct) {
					outputLayer.Neurons[y][x].Value = postAct[idx]
					idx++
				}
			}
		}
	} else {
		// For non-softmax activations, shader already applied activation
		n.applyFinalOutput(finalOutput)
	}

	return nil
}

// Read data from staging buffer with proper synchronization
func (n *Network[T]) readStagingBuffer(buffer *wgpu.Buffer, size int) ([]T, error) {
	done := make(chan wgpu.BufferMapAsyncStatus, 1)

	err := buffer.MapAsync(wgpu.MapModeRead, 0, buffer.GetSize(),
		func(status wgpu.BufferMapAsyncStatus) {
			done <- status
		})
	if err != nil {
		return nil, fmt.Errorf("failed to map buffer: %v", err)
	}

	// Poll until mapped
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
	data := buffer.GetMappedRange(0, uint(size*4))
	if data == nil {
		buffer.Unmap()
		return nil, fmt.Errorf("failed to get mapped range")
	}

	result := make([]T, size)
	copy(result, wgpu.FromBytes[T](data))
	buffer.Unmap()

	return result, nil
}

// Apply final GPU output to network neurons
func (n *Network[T]) applyFinalOutput(output []T) {
	outputLayer := &n.Layers[n.OutputLayer]
	idx := 0
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			if idx < len(output) {
				outputLayer.Neurons[y][x].Value = output[idx]
				idx++
			}
		}
	}
	if outputLayer.Neurons[0][0].Activation == "softmax" {
		n.ApplySoftmax()
	}
}

// Pipeline GPU computation using double buffering to reduce CPU-GPU sync
func (n *Network[T]) ForwardGPUPipelined(inputs [][]float64) error {
	if len(n.gpu.optimized.layers) < 2 {
		return n.ForwardGPUOptimized(inputs) // Fall back to sequential
	}

	// Convert input
	inputData := make([]float32, 0, len(inputs)*len(inputs[0]))
	for _, row := range inputs {
		for _, val := range row {
			inputData = append(inputData, float32(val))
		}
	}

	// Create multiple command encoders for pipelining
	encoders := make([]*wgpu.CommandEncoder, 0)

	// First layer
	encoder1, _ := ctx.device.CreateCommandEncoder(nil)
	encoders = append(encoders, encoder1)

	layer0 := n.gpu.optimized.layers[0]
	ctx.queue.WriteBuffer(layer0.inputBuffer, 0, wgpu.ToBytes(inputData))

	pass1 := encoder1.BeginComputePass(nil)
	pass1.SetPipeline(layer0.pipeline)
	pass1.SetBindGroup(0, layer0.bindGroup, nil)
	pass1.DispatchWorkgroups(layer0.workgroupsX, layer0.workgroupsY, 1)
	pass1.End()

	// Copy to next layer's input
	if len(n.gpu.optimized.layers) > 1 {
		nextLayer := n.gpu.optimized.layers[1]
		encoder1.CopyBufferToBuffer(
			layer0.outputBuffer, 0,
			nextLayer.inputBuffer, 0,
			uint64(layer0.outputSize)*4,
		)
	}

	// Submit first layer
	cmd1, _ := encoder1.Finish(nil)
	ctx.queue.Submit(cmd1)

	// Process remaining layers with overlapping
	for i := 1; i < len(n.gpu.optimized.layers); i++ {
		encoder, _ := ctx.device.CreateCommandEncoder(nil)
		layer := n.gpu.optimized.layers[i]

		pass := encoder.BeginComputePass(nil)
		pass.SetPipeline(layer.pipeline)
		pass.SetBindGroup(0, layer.bindGroup, nil)
		pass.DispatchWorkgroups(layer.workgroupsX, layer.workgroupsY, 1)
		pass.End()

		// Copy to staging for final layer
		if i == len(n.gpu.optimized.layers)-1 {
			encoder.CopyBufferToBuffer(
				layer.outputBuffer, 0,
				layer.stagingBuffer, 0,
				uint64(layer.outputSize)*4,
			)
		} else {
			// Copy to next layer
			nextLayer := n.gpu.optimized.layers[i+1]
			encoder.CopyBufferToBuffer(
				layer.outputBuffer, 0,
				nextLayer.inputBuffer, 0,
				uint64(layer.outputSize)*4,
			)
		}

		cmd, _ := encoder.Finish(nil)
		ctx.queue.Submit(cmd)
	}

	// Wait and read final results
	ctx.device.Poll(true, nil)
	finalLayer := n.gpu.optimized.layers[len(n.gpu.optimized.layers)-1]
	finalOutput, err := n.readStagingBuffer(finalLayer.stagingBuffer, int(finalLayer.outputSize))
	if err != nil {
		return err
	}

	n.applyFinalOutput(finalOutput)
	return nil
}

// Add this to your Network struct's gpu field:
/*
type gpuResources struct {
	// ... existing fields ...
	optimized *GPUCompute  // Add this field
}
*/

// In the Forward method in webgpu_optimized.go, ensure initialization:
func (n *Network[T]) Forward(inputs [][]float64) {
	// Initialize optimized GPU if not already done
	if n.WebGPUNative && n.gpu.optimized == nil {
		if err := n.InitializeOptimizedGPU(); err != nil {
			//if n.Debug {
			fmt.Errorf("Failed to initialize optimized GPU: %v\n", err)
			//}
			n.WebGPUNative = false
			n.forwardCPU(inputs)
			n.WebGPUNative = true
			return
		}
	}

	// Use optimized GPU if available and enabled
	if n.WebGPUNative && n.gpu.optimized != nil && n.gpu.optimized.initialized {
		err := n.ForwardGPUOptimized(inputs)
		if err != nil {
			//	if n.Debug {
			fmt.Errorf("Optimized GPU forward failed, falling back to CPU: %v\n", err)
			//}
			// Fall back to CPU
			n.forwardCPU(inputs)
		}
		return
	}

	// Fallback to existing implementation
	n.forwardCPU(inputs)
}

// Clean up GPU resources
func (n *Network[T]) CleanupOptimizedGPU() {
	if n.gpu.optimized == nil {
		return
	}

	for i, layer := range n.gpu.optimized.layers {
		if layer.inputBuffer != nil {
			layer.inputBuffer.Destroy()
		}
		if layer.outputBuffer != nil {
			layer.outputBuffer.Destroy()
		}
		if layer.weightBuffer != nil {
			layer.weightBuffer.Destroy()
		}
		if layer.biasBuffer != nil {
			layer.biasBuffer.Destroy()
		}
		if layer.stagingBuffer != nil {
			layer.stagingBuffer.Destroy()
		}
		if n.Debug {
			fmt.Printf("Cleaned up GPU resources for layer %d\n", i)
		}
	}

	n.gpu.optimized.layers = nil
	n.gpu.optimized.initialized = false
	n.gpu.optimized = nil
}
