// webgpu_attn.go
package paragon

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// GPUAttnColumnCompute handles a single attention column's GPU computation
type GPUAttnColumnCompute struct {
	pipeline        *wgpu.ComputePipeline
	bindGroup       *wgpu.BindGroup
	bindGroupLayout *wgpu.BindGroupLayout

	// Buffers
	inputBuffer   *wgpu.Buffer // Previous layer activations
	outputBuffer  *wgpu.Buffer // Column output (height values)
	wqBuffer      *wgpu.Buffer // Query weights per head
	wkBuffer      *wgpu.Buffer // Key weights per head
	wvBuffer      *wgpu.Buffer // Value weights per head
	woBuffer      *wgpu.Buffer // Output projection (if UseWo)
	stagingBuffer *wgpu.Buffer

	workgroupsX uint32
	column      int
	layerIndex  int
}

// Create GPU compute for an attention column
func (n *Network[T]) createAttnColumnCompute(layerIdx, col int) (*GPUAttnColumnCompute, error) {
	curr := &n.Layers[layerIdx]
	prev := &n.Layers[layerIdx-1]
	cfg := curr.Attn

	if cfg == nil || cfg.DK <= 0 {
		return nil, fmt.Errorf("no attention config for layer %d", layerIdx)
	}

	// Get or create attention parameters for this column
	params := ensureAttnParams[T](curr, prev, col)
	if params == nil {
		return nil, fmt.Errorf("failed to create attention params for layer %d col %d", layerIdx, col)
	}

	heads := params.Heads
	dk := params.DK
	N := prev.Width * prev.Height
	hd := heads * dk

	// Generate shader
	shaderCode := n.generateAttnShader(layerIdx, col, cfg, params)

	module, err := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          fmt.Sprintf("Layer_%d_Col_%d_Attn_Shader", layerIdx, col),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderCode},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create shader module: %v", err)
	}
	defer module.Release()

	// Bind group layout: input, output, wq, wk, wv, wo
	bindGroupLayout, err := ctx.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: fmt.Sprintf("Layer_%d_Col_%d_Attn_BindGroupLayout", layerIdx, col),
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 4, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			{Binding: 5, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
		},
	})
	if err != nil {
		return nil, err
	}

	pipelineLayout, err := ctx.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("Layer_%d_Col_%d_Attn_PipelineLayout", layerIdx, col),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, err
	}
	defer pipelineLayout.Release()

	pipeline, err := ctx.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Col_%d_Attn_Pipeline", layerIdx, col),
		Layout: pipelineLayout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     module,
			EntryPoint: "main",
		},
	})
	if err != nil {
		bindGroupLayout.Release()
		return nil, err
	}

	ac := &GPUAttnColumnCompute{
		pipeline:        pipeline,
		bindGroupLayout: bindGroupLayout,
		column:          col,
		layerIndex:      layerIdx,
		workgroupsX:     1, // Single workgroup processes entire attention
	}

	// Create buffers
	ac.inputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Col_%d_Attn_Input", layerIdx, col),
		Size:  uint64(N) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, err
	}

	ac.outputBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Col_%d_Attn_Output", layerIdx, col),
		Size:  uint64(curr.Height) * 4,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, err
	}

	ac.stagingBuffer, err = ctx.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: fmt.Sprintf("Layer_%d_Col_%d_Attn_Staging", layerIdx, col),
		Size:  uint64(curr.Height) * 4,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}

	// Pack multi-head weights: Wq, Wk, Wv (each is heads * dk floats)
	wq := make([]float32, heads*dk)
	wk := make([]float32, heads*dk)
	wv := make([]float32, heads*dk)
	for h := 0; h < heads; h++ {
		for t := 0; t < dk; t++ {
			wq[h*dk+t] = float32(params.Wq[h][t])
			wk[h*dk+t] = float32(params.Wk[h][t])
			wv[h*dk+t] = float32(params.Wv[h][t])
		}
	}

	ac.wqBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Col_%d_Wq", layerIdx, col),
		Contents: wgpu.ToBytes(wq),
		Usage:    wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, err
	}

	ac.wkBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Col_%d_Wk", layerIdx, col),
		Contents: wgpu.ToBytes(wk),
		Usage:    wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, err
	}

	ac.wvBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Col_%d_Wv", layerIdx, col),
		Contents: wgpu.ToBytes(wv),
		Usage:    wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, err
	}

	// Wo buffer (if UseWo)
	var wo []float32
	if cfg.UseWo && len(params.Wo) == hd*curr.Height {
		wo = make([]float32, len(params.Wo))
		for i := range params.Wo {
			wo[i] = float32(params.Wo[i])
		}
	} else {
		// Dummy buffer (size 4)
		wo = make([]float32, 1)
	}

	ac.woBuffer, err = ctx.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    fmt.Sprintf("Layer_%d_Col_%d_Wo", layerIdx, col),
		Contents: wgpu.ToBytes(wo),
		Usage:    wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, err
	}

	// Create bind group
	ac.bindGroup, err = ctx.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  fmt.Sprintf("Layer_%d_Col_%d_Attn_BindGroup", layerIdx, col),
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: ac.inputBuffer, Size: ac.inputBuffer.GetSize()},
			{Binding: 1, Buffer: ac.outputBuffer, Size: ac.outputBuffer.GetSize()},
			{Binding: 2, Buffer: ac.wqBuffer, Size: ac.wqBuffer.GetSize()},
			{Binding: 3, Buffer: ac.wkBuffer, Size: ac.wkBuffer.GetSize()},
			{Binding: 4, Buffer: ac.wvBuffer, Size: ac.wvBuffer.GetSize()},
			{Binding: 5, Buffer: ac.woBuffer, Size: ac.woBuffer.GetSize()},
		},
	})
	if err != nil {
		return nil, err
	}

	return ac, nil
}

// Generate WGSL shader for multi-head attention column
func (n *Network[T]) generateAttnShader(layerIdx, col int, cfg *AttnConfig[T], params *AttnParams[T]) string {
	prev := n.Layers[layerIdx-1]
	curr := n.Layers[layerIdx]

	heads := params.Heads
	dk := params.DK
	hd := heads * dk
	N := prev.Width * prev.Height

	posEncAmp := cfg.PosEncAmp
	if posEncAmp == 0 {
		posEncAmp = 0.02
	}
	normEps := cfg.NormEps
	if normEps == 0 {
		normEps = 1e-6
	}

	activation := curr.Neurons[0][col].Activation
	activationCode := getActivationCode(activation, "f32")

	useReplay := cfg.UseReplay && cfg.ForceReplay
	replayGain := cfg.ReplayGain
	if replayGain == 0 {
		replayGain = 1.1
	}

	shader := fmt.Sprintf(`
@group(0) @binding(0) var<storage, read> input: array<f32>;      // N elements
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // CURR_H elements
@group(0) @binding(2) var<storage, read> wq: array<f32>;        // HEADS * DK
@group(0) @binding(3) var<storage, read> wk: array<f32>;        // HEADS * DK
@group(0) @binding(4) var<storage, read> wv: array<f32>;        // HEADS * DK
@group(0) @binding(5) var<storage, read> wo: array<f32>;        // HD * CURR_H (if UseWo)

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
const USE_REPLAY: u32 = %du;
const REPLAY_GAIN: f32 = %f;

%s

@compute @workgroup_size(1, 1, 1)
fn main() {
    // 1) Build tokens with positional encoding
    var X: array<f32, N>;
    for (var i = 0u; i < N; i++) {
        X[i] = input[i];
    }
`,
		prev.Width, prev.Height, N, heads, dk, hd, curr.Height,
		posEncAmp, normEps,
		boolToU32(cfg.UseNorm), boolToU32(cfg.UseWo && len(params.Wo) == hd*curr.Height),
		boolToU32(useReplay), float32(replayGain),
		activationCode)

	if cfg.PosEnc2D {
		shader += `
    // Add 2D positional encoding
    for (var y = 0u; y < PREV_H; y++) {
        for (var x = 0u; x < PREV_W; x++) {
            let i = y * PREV_W + x;
            let fy = f32(y) / f32(max(1u, PREV_H - 1u));
            let fx = f32(x) / f32(max(1u, PREV_W - 1u));
            X[i] += POS_ENC_AMP * (sin(6.283185307 * fy) + cos(6.283185307 * fx));
        }
    }
`
	}

	shader += `
    // 2) Multi-head attention
    var zbar_cat: array<f32, HD>;
    
    for (var h = 0u; h < HEADS; h++) {
        // Q, K, V projections for this head
        var Q: array<array<f32, DK>, N>;
        var K: array<array<f32, DK>, N>;
        var V: array<array<f32, DK>, N>;
        
        let wq_base = h * DK;
        let wk_base = h * DK;
        let wv_base = h * DK;
        
        for (var i = 0u; i < N; i++) {
            let x = X[i];
            for (var t = 0u; t < DK; t++) {
                Q[i][t] = x * wq[wq_base + t];
                K[i][t] = x * wk[wk_base + t];
                V[i][t] = x * wv[wv_base + t];
            }
        }
        
        // Attention scores
        let scale = 1.0 / sqrt(f32(DK));
        var A: array<array<f32, N>, N>;
        for (var i = 0u; i < N; i++) {
            // Compute Q[i] · K[j] for all j
            for (var j = 0u; j < N; j++) {
                var dot_sum = 0.0;
                for (var t = 0u; t < DK; t++) {
                    dot_sum += Q[i][t] * K[j][t];
                }
                A[i][j] = scale * dot_sum;
            }
            
            // Softmax row i
            var mx = A[i][0];
            for (var j = 1u; j < N; j++) {
                mx = max(mx, A[i][j]);
            }
            var sum = 0.0;
            for (var j = 0u; j < N; j++) {
                let e = exp(A[i][j] - mx);
                A[i][j] = e;
                sum += e;
            }
            if (sum > 0.0) {
                let inv = 1.0 / sum;
                for (var j = 0u; j < N; j++) {
                    A[i][j] *= inv;
                }
            }
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
            zbar_cat[h * DK + t] = zbar_h[t];
        }
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
    
    // 4) Project to column output
    for (var y = 0u; y < CURR_H; y++) {
        var val = 0.0;
        
        if (USE_WO == 1u) {
            // Use Wo projection
            let base = y * HD;
            for (var t = 0u; t < HD; t++) {
                val += zbar_cat[t] * wo[base + t];
            }
        } else {
            // Fallback: weighted sum
            var sum = 0.0;
            for (var t = 0u; t < HD; t++) {
                sum += zbar_cat[t];
            }
            let alpha = f32(y) / f32(max(1u, CURR_H - 1u));
            val = (1.0 - alpha) * sum + 0.5 * alpha * sum;
        }
        
        // Apply activation
        val = activate(val);
        
        // Apply replay gain if enabled
        if (USE_REPLAY == 1u) {
            val *= REPLAY_GAIN;
        }
        
        output[y] = val;
    }
}
`
	return shader
}

func boolToU32(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

func (ac *GPUAttnColumnCompute) cleanup() {
	if ac.inputBuffer != nil {
		ac.inputBuffer.Destroy()
	}
	if ac.outputBuffer != nil {
		ac.outputBuffer.Destroy()
	}
	if ac.wqBuffer != nil {
		ac.wqBuffer.Destroy()
	}
	if ac.wkBuffer != nil {
		ac.wkBuffer.Destroy()
	}
	if ac.wvBuffer != nil {
		ac.wvBuffer.Destroy()
	}
	if ac.woBuffer != nil {
		ac.woBuffer.Destroy()
	}
	if ac.stagingBuffer != nil {
		ac.stagingBuffer.Destroy()
	}
	if ac.bindGroup != nil {
		ac.bindGroup.Release()
	}
	if ac.bindGroupLayout != nil {
		ac.bindGroupLayout.Release()
	}
	if ac.pipeline != nil {
		ac.pipeline.Release()
	}
}
