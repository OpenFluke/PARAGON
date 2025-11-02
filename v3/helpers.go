// paragon/helpers.go
package paragon

import (
	"errors"
	"fmt"
	"os"
	"time"
)

/* ========== Small utilities you can reuse across demos/tests ========== */

// MustLoadOrCreateBundle loads a single-model bundle if it exists,
// otherwise builds it with builder(), bundles it, and writes to disk.
func MustLoadOrCreateBundle[T Numeric](
	path string,
	id string,
	builder func() (*Network[T], error),
) *Network[T] {
	if FileExists(path) {
		b, err := os.ReadFile(path)
		if err != nil {
			panic(fmt.Errorf("read bundle %s: %w", path, err))
		}
		nAny, err := ImportBundleJSON(string(b))
		if err != nil {
			panic(fmt.Errorf("import bundle %s: %w", path, err))
		}
		n, ok := nAny.(*Network[T])
		if !ok {
			panic(fmt.Errorf("bundle type mismatch in %s", path))
		}
		return n
	}

	n, err := builder()
	if err != nil {
		panic(err)
	}
	seed := time.Now().UnixNano()
	out, err := ExportBundleJSON(id, n, &seed)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
		panic(err)
	}
	return n
}

/* ========== Tiny “builder” helpers so your tests are 1-liners ========== */

// GridSpec is a compact way to specify layers.
type GridSpec struct {
	Width, Height int
}

// BuildGridNet builds a Network with sizes/acts/connectivity.
// It also lets you set per-layer slice types and attn config succinctly.
type BuildOpts[T Numeric] struct {
	Sizes       []GridSpec
	Activations []string
	FullyConn   []bool
	// Optional extras per layer (len should match #layers, empty -> no change)
	SliceTypes    [][]string       // per layer, per column
	Attn          []*AttnConfig[T] // per layer
	ReplayEnabled []bool           // optional replay flags
	ReplayOffset  []int
	ReplayPhase   []string // "before"/"after"
	ReplayMax     []int
	ReplayBudget  []int
}

func BuildGridNet[T Numeric](opts BuildOpts[T]) (*Network[T], error) {
	if len(opts.Sizes) != len(opts.Activations) || len(opts.Sizes) != len(opts.FullyConn) {
		return nil, errors.New("BuildGridNet: mismatched sizes/activations/connectivity")
	}

	layerSizes := make([]struct{ Width, Height int }, len(opts.Sizes))
	for i, s := range opts.Sizes {
		layerSizes[i] = struct{ Width, Height int }{s.Width, s.Height}
	}
	n, err := NewNetwork[T](layerSizes, opts.Activations, opts.FullyConn)
	if err != nil {
		return nil, err
	}

	// Optional per-layer extras
	for i := range n.Layers {
		// slice types
		if i < len(opts.SliceTypes) && len(opts.SliceTypes[i]) > 0 {
			n.Layers[i].SliceTypes = make([]string, len(opts.SliceTypes[i]))
			copy(n.Layers[i].SliceTypes, opts.SliceTypes[i])
		}
		// attn
		if i < len(opts.Attn) && opts.Attn[i] != nil {
			// Make a copy so the caller’s struct isn’t modified at runtime.
			a := *opts.Attn[i]
			n.Layers[i].Attn = &a
		}
		// replay (optional)
		if i < len(opts.ReplayEnabled) {
			n.Layers[i].ReplayEnabled = opts.ReplayEnabled[i]
		}
		if i < len(opts.ReplayOffset) {
			n.Layers[i].ReplayOffset = opts.ReplayOffset[i]
		}
		if i < len(opts.ReplayPhase) && opts.ReplayPhase[i] != "" {
			n.Layers[i].ReplayPhase = opts.ReplayPhase[i]
		}
		if i < len(opts.ReplayMax) {
			n.Layers[i].MaxReplay = opts.ReplayMax[i]
		}
		if i < len(opts.ReplayBudget) {
			n.Layers[i].ReplayBudget = opts.ReplayBudget[i]
		}
	}
	return n, nil
}
