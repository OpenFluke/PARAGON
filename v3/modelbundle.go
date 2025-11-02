// modelbundle.go
package paragon

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
)

/* =============================================================================
   Wire types (JSON schema)
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
	DK       int     `json:"dk"`
	UseWo    bool    `json:"useWo"`
	Share    string  `json:"share"`    // "layer" | "per-slice"
	Dropout  float32 `json:"dropout"`  // training only
	PosEnc2D bool    `json:"posEnc2D"` // spatial awareness

	// NEW (v2): keep these omitempty for back-compat with v1 bundles
	UseNorm     bool    `json:"useNorm,omitempty"`
	PosEncAmp   float64 `json:"posEncAmp,omitempty"`
	NormEps     float64 `json:"normEps,omitempty"`
	UseReplay   bool    `json:"useReplay,omitempty"`
	ForceReplay bool    `json:"forceReplay,omitempty"`
	ReplayGain  float64 `json:"replayGain,omitempty"`
}

// Learned attention parameter payload
type AttnParamsWire struct {
	Wq []float64 `json:"wq"`           // len = dk
	Wk []float64 `json:"wk"`           // len = dk
	Wv []float64 `json:"wv"`           // len = dk
	Wo []float64 `json:"wo,omitempty"` // len = dk * Hcurr (only when UseWo)
	DK int       `json:"dk"`
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
		Version: 2, // bumped
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
	// For now, return first model. (You can extend to select by ID.)
	nAny, err := UnpackModel(bun.Models[0])
	if err != nil {
		return nil, err
	}
	// Back-compat defaults for v1 (or sloppy emitters)
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
			lc.Attn = &AttnCfgWire{
				DK:          g.Attn.DK,
				UseWo:       g.Attn.UseWo,
				Share:       g.Attn.Share,
				Dropout:     g.Attn.Dropout,
				PosEnc2D:    g.Attn.PosEnc2D,
				UseNorm:     g.Attn.UseNorm,
				PosEncAmp:   g.Attn.PosEncAmp,
				NormEps:     g.Attn.NormEps,
				UseReplay:   g.Attn.UseReplay,
				ForceReplay: g.Attn.ForceReplay,
				ReplayGain:  g.Attn.ReplayGain,
			}
			switch g.Attn.Share {
			case "layer":
				if g.attnLayer != nil {
					lc.AttnParams = &AttnParamsWire{
						DK: g.attnLayer.DK,
						Wq: tVecToF64(g.attnLayer.Wq),
						Wk: tVecToF64(g.attnLayer.Wk),
						Wv: tVecToF64(g.attnLayer.Wv),
						Wo: tVecToF64(g.attnLayer.Wo),
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
							DK: p.DK,
							Wq: tVecToF64(p.Wq),
							Wk: tVecToF64(p.Wk),
							Wv: tVecToF64(p.Wv),
							Wo: tVecToF64(p.Wo),
						}
					}
				}
			}
		}
		// Replay (optional legacy block separate to Attn.UseReplay fields)
		if g.ReplayEnabled || g.ReplayOffset != 0 || g.ReplayPhase != "" || g.MaxReplay != 0 || g.ReplayBudget != 0 {
			lc.Replay = &ReplayCfg{
				Enabled: g.ReplayEnabled,
				Offset:  g.ReplayOffset,
				Phase:   g.ReplayPhase,
				Max:     g.MaxReplay,
				Budget:  g.ReplayBudget,
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
			gl.Attn = &AttnConfig[T]{
				DK:          cl.Attn.DK,
				UseWo:       cl.Attn.UseWo,
				Share:       cl.Attn.Share,
				Dropout:     cl.Attn.Dropout,
				PosEnc2D:    cl.Attn.PosEnc2D,
				UseNorm:     cl.Attn.UseNorm,
				PosEncAmp:   cl.Attn.PosEncAmp,
				NormEps:     cl.Attn.NormEps,
				UseReplay:   cl.Attn.UseReplay,
				ForceReplay: cl.Attn.ForceReplay,
				ReplayGain:  cl.Attn.ReplayGain,
			}
			// Back-compat defaults for v1 bundles or zeroed fields
			if gl.Attn.PosEncAmp == 0 {
				gl.Attn.PosEncAmp = 1e-2
			}
			if gl.Attn.NormEps == 0 {
				gl.Attn.NormEps = 1e-6
			}
			if gl.Attn.ReplayGain == 0 {
				gl.Attn.ReplayGain = 1.1
			}
			if gl.Attn.Share == "" {
				gl.Attn.Share = "layer"
			}

			switch gl.Attn.Share {
			case "layer":
				if cl.AttnParams != nil {
					gl.attnLayer = &AttnParams[T]{
						DK: cl.AttnParams.DK,
						Wq: f64VecToT[T](cl.AttnParams.Wq),
						Wk: f64VecToT[T](cl.AttnParams.Wk),
						Wv: f64VecToT[T](cl.AttnParams.Wv),
						Wo: f64VecToT[T](cl.AttnParams.Wo),
					}
				} else {
					gl.attnLayer = nil // allow lazy init if truly absent
				}
				gl.attnSlices = nil

			case "per-slice":
				gl.attnLayer = nil
				if gl.attnSlices == nil {
					gl.attnSlices = make(map[int]*AttnParams[T])
				} else {
					for k := range gl.attnSlices {
						delete(gl.attnSlices, k)
					}
				}
				if cl.AttnParamsPerSlice != nil {
					for col, pw := range cl.AttnParamsPerSlice {
						if pw == nil {
							continue
						}
						gl.attnSlices[col] = &AttnParams[T]{
							DK: pw.DK,
							Wq: f64VecToT[T](pw.Wq),
							Wk: f64VecToT[T](pw.Wk),
							Wv: f64VecToT[T](pw.Wv),
							Wo: f64VecToT[T](pw.Wo),
						}
					}
				}
			default:
				gl.attnLayer = nil
				gl.attnSlices = nil
			}
		}

		// Replay (legacy block)
		if cl.Replay != nil {
			gl.ReplayEnabled = cl.Replay.Enabled
			gl.ReplayOffset = cl.Replay.Offset
			gl.ReplayPhase = cl.Replay.Phase
			gl.MaxReplay = cl.Replay.Max
			gl.ReplayBudget = cl.Replay.Budget
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
   Small helpers: []T <-> []float64
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
		if g.Attn.Share == "" {
			g.Attn.Share = "layer"
		}
		if g.Attn.PosEncAmp == 0 {
			g.Attn.PosEncAmp = 1e-2
		}
		if g.Attn.NormEps == 0 {
			g.Attn.NormEps = 1e-6
		}
		if g.Attn.ReplayGain == 0 {
			g.Attn.ReplayGain = 1.1
		}
	}
}
