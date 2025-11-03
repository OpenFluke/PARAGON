// modelbundle.go
package paragon

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
)

/* =============================================================================
   Wire types (JSON schema)  — v3 (multi-head)
   ========================================================================== */

// Whole file format (can hold multiple models)
type Bundle struct {
	Type    string        `json:"type"`    // e.g. "modelhost/bundle"
	Version int           `json:"version"` // bump if schema changes
	Models  []BundleModel `json:"models"`
}

// One model entry (layout/knobs + weights)
type BundleModel struct {
	ID      string   `json:"id"`
	Cfg     ModelCfg `json:"cfg"`
	Weights Weights  `json:"weights"`
}

// Layout/knobs only (no weights)
type ModelCfg struct {
	ID           string     `json:"id"`
	Layers       []LayerCfg `json:"layers"`
	Activations  []string   `json:"activations"`
	Seed         *int64     `json:"seed,omitempty"`
	Connectivity []bool     `json:"connectivity,omitempty"`
}

type LayerCfg struct {
	Width  int          `json:"Width"`
	Height int          `json:"Height"`
	Slices []string     `json:"slices,omitempty"` // per-column: "dense","attn",...
	Attn   *AttnCfgWire `json:"attn,omitempty"`   // attention knobs (wire version)

	// Persisted attention params (prevents lazy re-init on load)
	AttnParams         *AttnParamsWire         `json:"attn_params,omitempty"`           // when Share == "layer"
	AttnParamsPerSlice map[int]*AttnParamsWire `json:"attn_params_per_slice,omitempty"` // when Share == "per-slice"`

	Replay      *ReplayCfg `json:"replay,omitempty"`
	Description string     `json:"description,omitempty"`
}

type AttnCfgWire struct {
	// Multi-head
	Heads int  `json:"heads,omitempty"` // default 1 if absent/zero
	DK    int  `json:"dk"`
	UseWo bool `json:"useWo"`

	Share    string  `json:"share"`    // "layer" | "per-slice"
	Dropout  float32 `json:"dropout"`  // training only (no-op here)
	PosEnc2D bool    `json:"posEnc2D"` // spatial awareness
	UseNorm  bool    `json:"useNorm,omitempty"`

	// Tunables (v2+; keep omitempty for back-compat)
	PosEncAmp float64 `json:"posEncAmp,omitempty"`
	NormEps   float64 `json:"normEps,omitempty"`

	// Replay controls (v2+)
	UseReplay   bool    `json:"useReplay,omitempty"`
	ForceReplay bool    `json:"forceReplay,omitempty"`
	ReplayGain  float64 `json:"replayGain,omitempty"`
}

// Learned attention parameter payload (v3: per-head)
type AttnParamsWire struct {
	// Per-head projection weights; each inner slice len = dk
	Wq [][]float64 `json:"wq,omitempty"` // shape: [heads][dk]
	Wk [][]float64 `json:"wk,omitempty"` // shape: [heads][dk]
	Wv [][]float64 `json:"wv,omitempty"` // shape: [heads][dk]

	// Output projection over concatenated heads: len = (heads*dk)*Hcurr, row-major
	Wo []float64 `json:"wo,omitempty"`

	// Meta
	DK    int `json:"dk"`
	Heads int `json:"heads,omitempty"`

	/* Back-compat (v1/v2 single-head):
	   If these are present (flat vectors), we will wrap them into [][] with Heads=1.
	*/
	WqFlat []float64 `json:"wq_flat,omitempty"`
	WkFlat []float64 `json:"wk_flat,omitempty"`
	WvFlat []float64 `json:"wv_flat,omitempty"`
}

type ReplayCfg struct {
	Enabled bool   `json:"enabled"`
	Offset  int    `json:"offset"`
	Phase   string `json:"phase"` // "before"|"after"
	Max     int    `json:"max"`
	Budget  int    `json:"budget"`
}

// Weights envelope. Fmt "jsonModelB64" wraps your existing ToS/FromS JSON.
type Weights struct {
	Fmt  string `json:"fmt"`  // "jsonModelB64"
	Data string `json:"data"` // base64(JSON(sNet))
}

/* =============================================================================
   Public API (string-in / string-out)
   ========================================================================== */

// Export one model to a bundle JSON (single-entry Models).
func ExportBundleJSON[T Numeric](id string, n *Network[T], seed *int64) (string, error) {
	// Ensure attention params exist before packing, so they get persisted.
	ensureAttnMaterialized(n)

	m, err := PackModel(id, n, seed)
	if err != nil {
		return "", err
	}
	bun := Bundle{
		Type:    "modelhost/bundle",
		Version: 3, // bumped for multi-head
		Models:  []BundleModel{m},
	}
	out, err := json.Marshal(bun)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// Import bundle JSON (expects at least one model) → typed *Network[...] as any.
func ImportBundleJSON(bundleJSON string) (any, error) {
	var bun Bundle
	if err := json.Unmarshal([]byte(bundleJSON), &bun); err != nil {
		return nil, fmt.Errorf("bundle parse: %w", err)
	}
	if bun.Type == "" {
		return nil, errors.New("bundle: missing type")
	}
	if len(bun.Models) == 0 {
		return nil, errors.New("bundle: no models")
	}
	// For now, return first model. (Extend to select by ID if needed.)
	nAny, err := UnpackModel(bun.Models[0])
	if err != nil {
		return nil, err
	}
	// Back-compat defaults for older bundles
	switch n := nAny.(type) {
	case *Network[float32]:
		fillAttnDefaults(n)
	case *Network[float64]:
		fillAttnDefaults(n)
	}
	return nAny, nil
}

/* =============================================================================
   Pack / Unpack a single model
   ========================================================================== */

func PackModel[T Numeric](id string, n *Network[T], seed *int64) (BundleModel, error) {
	cfg := modelCfgFromNetwork(id, n, seed)

	raw, err := n.MarshalJSONModel() // existing sNet JSON (weights + topology)
	if err != nil {
		return BundleModel{}, err
	}
	w := Weights{
		Fmt:  "jsonModelB64",
		Data: base64.StdEncoding.EncodeToString(raw),
	}
	return BundleModel{ID: id, Cfg: cfg, Weights: w}, nil
}

// Build a typed network from weights, then apply cfg (slices/attn/replay + attn params).
func UnpackModel(m BundleModel) (any, error) {
	if m.Weights.Fmt != "jsonModelB64" {
		return nil, fmt.Errorf("unsupported weights fmt: %s", m.Weights.Fmt)
	}
	b, err := base64.StdEncoding.DecodeString(m.Weights.Data)
	if err != nil {
		return nil, fmt.Errorf("decode weights: %w", err)
	}

	// Use your dynamic loader to pick the correct T (float32, int, etc.)
	nAny, err := LoadNamedNetworkFromJSONString(string(b))
	if err != nil {
		return nil, fmt.Errorf("load weighted model: %w", err)
	}

	// Apply cfg knobs and persisted attention params.
	switch n := nAny.(type) {
	case *Network[float32]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[float64]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[int]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[int8]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[int16]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[int32]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[int64]:
		if err := applyCfgToNetwork(n, m.Cfg); err != nil {
			return nil, err
		}
	case *Network[uint], *Network[uint8], *Network[uint16], *Network[uint32], *Network[uint64]:
		// add explicit cases if you actually use uint nets
	}

	return nAny, nil
}

/* =============================================================================
   Internal: build ModelCfg from a live Network[T]
   ========================================================================== */

func modelCfgFromNetwork[T Numeric](id string, n *Network[T], seed *int64) ModelCfg {
	cfg := ModelCfg{
		ID:          id,
		Layers:      make([]LayerCfg, len(n.Layers)),
		Activations: make([]string, len(n.Layers)),
		Seed:        seed,
	}
	for i, g := range n.Layers {
		lc := LayerCfg{
			Width:  g.Width,
			Height: g.Height,
		}
		// Slices
		if len(g.SliceTypes) == g.Width {
			cp := make([]string, len(g.SliceTypes))
			copy(cp, g.SliceTypes)
			lc.Slices = cp
		}
		// Attention knobs + PARAMS (persist)
		if g.Attn != nil {
			heads := defHeads(g.Attn)
			lc.Attn = &AttnCfgWire{
				Heads:     heads,
				DK:        g.Attn.DK,
				UseWo:     g.Attn.UseWo,
				Share:     g.Attn.Share,
				Dropout:   g.Attn.Dropout,
				PosEnc2D:  g.Attn.PosEnc2D,
				UseNorm:   g.Attn.UseNorm,
				PosEncAmp: g.Attn.PosEncAmp,
				NormEps:   g.Attn.NormEps,
			}
			switch g.Attn.Share {
			case "layer":
				if g.attnLayer != nil {
					lc.AttnParams = &AttnParamsWire{
						DK:    g.attnLayer.DK,
						Heads: g.attnLayer.Heads,
						Wq:    t2DToF64(g.attnLayer.Wq),
						Wk:    t2DToF64(g.attnLayer.Wk),
						Wv:    t2DToF64(g.attnLayer.Wv),
						Wo:    tVecToF64(g.attnLayer.Wo),
					}
					if lc.AttnParams.Heads == 0 {
						lc.AttnParams.Heads = heads
					}
				}
			case "per-slice":
				if g.attnSlices != nil && len(g.attnSlices) > 0 {
					lc.AttnParamsPerSlice = make(map[int]*AttnParamsWire, len(g.attnSlices))
					for col, p := range g.attnSlices {
						if p == nil {
							continue
						}
						lc.AttnParamsPerSlice[col] = &AttnParamsWire{
							DK:    p.DK,
							Heads: p.Heads,
							Wq:    t2DToF64(p.Wq),
							Wk:    t2DToF64(p.Wk),
							Wv:    t2DToF64(p.Wv),
							Wo:    tVecToF64(p.Wo),
						}
						if lc.AttnParamsPerSlice[col].Heads == 0 {
							lc.AttnParamsPerSlice[col].Heads = heads
						}
					}
				}
			}
		}

		cfg.Layers[i] = lc
		// capture the layer's default activation (taken from any neuron)
		if g.Height > 0 && g.Width > 0 && g.Neurons[0][0] != nil {
			cfg.Activations[i] = g.Neurons[0][0].Activation
		}
	}
	return cfg
}

/* =============================================================================
   Internal: apply cfg (slices/attn/replay + attn params) to a live Network[T]
   ========================================================================== */

func applyCfgToNetwork[T Numeric](n *Network[T], cfg ModelCfg) error {
	if len(cfg.Layers) != len(n.Layers) {
		return fmt.Errorf("cfg/layer-count mismatch: cfg=%d, net=%d", len(cfg.Layers), len(n.Layers))
	}
	for i := range n.Layers {
		gl := &n.Layers[i]
		cl := cfg.Layers[i]

		// Slices
		if len(cl.Slices) > 0 {
			if len(cl.Slices) != gl.Width {
				return fmt.Errorf("cfg slices width mismatch at layer %d: %d != %d", i, len(cl.Slices), gl.Width)
			}
			if gl.SliceTypes == nil || len(gl.SliceTypes) != gl.Width {
				gl.SliceTypes = make([]string, gl.Width)
			}
			copy(gl.SliceTypes, cl.Slices)
		}

		// Attention knobs + PARAMS
		if cl.Attn != nil {
			if gl.Attn == nil {
				gl.Attn = &AttnConfig[T]{}
			}
			gl.Attn.Heads = cl.Attn.Heads
			gl.Attn.DK = cl.Attn.DK
			gl.Attn.UseWo = cl.Attn.UseWo
			gl.Attn.Share = cl.Attn.Share
			gl.Attn.Dropout = cl.Attn.Dropout
			gl.Attn.PosEnc2D = cl.Attn.PosEnc2D
			gl.Attn.UseNorm = cl.Attn.UseNorm
			gl.Attn.PosEncAmp = cl.Attn.PosEncAmp
			gl.Attn.NormEps = cl.Attn.NormEps

			// Back-compat defaults for older bundles or zeroed fields
			if gl.Attn.Heads <= 0 {
				gl.Attn.Heads = 1
			}
			if gl.Attn.PosEncAmp == 0 {
				gl.Attn.PosEncAmp = 1e-2
			}
			if gl.Attn.NormEps == 0 {
				gl.Attn.NormEps = 1e-6
			}

			if gl.Attn.Share == "" {
				gl.Attn.Share = "layer"
			}

			// Clear any existing param caches; we’ll fill from wire (if present)
			gl.attnLayer = nil
			if gl.attnSlices != nil {
				for k := range gl.attnSlices {
					delete(gl.attnSlices, k)
				}
			}

			// Load params
			switch gl.Attn.Share {
			case "layer":
				if cl.AttnParams != nil {
					p := attnParamsFromWire[T](cl.AttnParams)
					gl.attnLayer = p
				}
			case "per-slice":
				if cl.AttnParamsPerSlice != nil {
					if gl.attnSlices == nil {
						gl.attnSlices = make(map[int]*AttnParams[T], len(cl.AttnParamsPerSlice))
					}
					for col, pw := range cl.AttnParamsPerSlice {
						if pw == nil {
							continue
						}
						gl.attnSlices[col] = attnParamsFromWire[T](pw)
					}
				}
			default:
				// leave nil; lazy-init later
			}
		}

	}
	return nil
}

/* =============================================================================
   Convenience: pack/unpack weights as raw strings
   ========================================================================== */

func PackWeightsJSONB64[T Numeric](n *Network[T]) (string, error) {
	raw, err := n.MarshalJSONModel()
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(raw), nil
}

func UnpackWeightsJSONB64(weightsB64 string) (any, error) {
	b, err := base64.StdEncoding.DecodeString(weightsB64)
	if err != nil {
		return nil, err
	}
	return LoadNamedNetworkFromJSONString(string(b))
}

/* =============================================================================
   Small helpers: []T/[][]T <-> []float64/[][]float64
   ========================================================================== */

func tVecToF64[T Numeric](v []T) []float64 {
	if v == nil {
		return nil
	}
	out := make([]float64, len(v))
	for i := range v {
		out[i] = float64(any(v[i]).(T))
	}
	return out
}

func f64VecToT[T Numeric](v []float64) []T {
	if v == nil {
		return nil
	}
	out := make([]T, len(v))
	for i := range v {
		out[i] = T(v[i])
	}
	return out
}

func t2DToF64[T Numeric](vv [][]T) [][]float64 {
	if vv == nil {
		return nil
	}
	out := make([][]float64, len(vv))
	for i := range vv {
		out[i] = tVecToF64(vv[i])
	}
	return out
}

func f642DToT[T Numeric](vv [][]float64) [][]T {
	if vv == nil {
		return nil
	}
	out := make([][]T, len(vv))
	for i := range vv {
		out[i] = f64VecToT[T](vv[i])
	}
	return out
}

/* =============================================================================
   Param conversions: wire <-> runtime
   ========================================================================== */

func attnParamsFromWire[T Numeric](pw *AttnParamsWire) *AttnParams[T] {
	if pw == nil {
		return nil
	}
	p := &AttnParams[T]{DK: pw.DK, Heads: pw.Heads}
	if p.Heads <= 0 {
		// Back-compat: if flat vectors exist, wrap as single head
		if len(pw.Wq) == 0 && len(pw.WqFlat) > 0 {
			p.Heads = 1
			p.Wq = [][]T{f64VecToT[T](pw.WqFlat)}
			p.Wk = [][]T{f64VecToT[T](pw.WkFlat)}
			p.Wv = [][]T{f64VecToT[T](pw.WvFlat)}
		} else {
			p.Heads = 1
		}
	} else {
		p.Wq = f642DToT[T](pw.Wq)
		p.Wk = f642DToT[T](pw.Wk)
		p.Wv = f642DToT[T](pw.Wv)
	}
	p.Wo = f64VecToT[T](pw.Wo)
	return p
}

/* =============================================================================
   NEW: materialize attention params before packing
   ========================================================================== */

// ensureAttnMaterialized runs a cheap zero-input forward so that any attention
// params that are lazily initialised become non-nil and can be persisted.
func ensureAttnMaterialized[T Numeric](n *Network[T]) {
	need := false
	for i := range n.Layers {
		g := &n.Layers[i]
		if g.Attn == nil {
			continue
		}
		switch g.Attn.Share {
		case "layer":
			if g.attnLayer == nil {
				need = true
			}
		case "per-slice":
			if g.attnSlices == nil || len(g.attnSlices) == 0 {
				need = true
			}
		}
		if need {
			break
		}
	}
	if !need {
		return
	}

	// Build a zero input (H×W) and do a single forward.
	inGrid := make([][]float64, n.Layers[n.InputLayer].Height)
	for y := range inGrid {
		inGrid[y] = make([]float64, n.Layers[n.InputLayer].Width)
	}
	n.Forward(inGrid)
}

/* =============================================================================
   Back-compat: fill defaults when older bundles are loaded
   ========================================================================== */

func fillAttnDefaults[T Numeric](n *Network[T]) {
	for i := range n.Layers {
		g := &n.Layers[i]
		if g.Attn == nil {
			continue
		}
		if g.Attn.Heads <= 0 {
			g.Attn.Heads = 1
		}
		if g.Attn.Share == "" {
			g.Attn.Share = "layer"
		}
		if g.Attn.PosEncAmp == 0 {
			g.Attn.PosEncAmp = 1e-2
		}
		if g.Attn.NormEps == 0 {
			g.Attn.NormEps = 1e-6
		}
	}
}
