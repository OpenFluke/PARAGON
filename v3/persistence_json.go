package paragon

import (
	"encoding/json"
	"fmt"
	"os"
)

/*────────────────────────────  aliases  ────────────────────────────────*/
// Paragon’s concrete layer type is Grid.  Alias it so we can keep using
// the more generic name “Layer” in this file.
type Layer[T Numeric] = Grid[T]

/*────────────────────────── serialisable shapes ───────────────────────*/

type sConn struct {
	L int     `json:"layer"` // source layer index
	X int     `json:"x"`     // source neuron X
	Y int     `json:"y"`     // source neuron Y
	W float64 `json:"w"`     // weight
}

type sNeuron struct {
	Bias float64 `json:"b"`
	Act  string  `json:"a"`
	In   []sConn `json:"in"`
}

type sLayer struct {
	W       int         `json:"w"`
	H       int         `json:"h"`
	Neurons [][]sNeuron `json:"n"`

	ReplayEnabled bool   `json:"re_enabled,omitempty"`
	ReplayOffset  int    `json:"re_offset,omitempty"`
	ReplayPhase   string `json:"re_phase,omitempty"`
	MaxReplay     int    `json:"re_max,omitempty"`
	ReplayBudget  int    `json:"re_budget,omitempty"`
}

type sNet struct {
	Type   string   `json:"type"`
	Layers []sLayer `json:"layers"`
}

/*────────────────── helpers: Network ↔ serialisable ───────────────────*/

// toS flattens a runtime Network into raw data ready for JSON.
func (n *Network[T]) ToS() sNet {
	s := sNet{
		Type:   n.TypeName,
		Layers: make([]sLayer, len(n.Layers))}

	for li, L := range n.Layers {
		sl := sLayer{
			W:       L.Width,
			H:       L.Height,
			Neurons: make([][]sNeuron, L.Height),
		}

		for y := 0; y < L.Height; y++ {
			row := make([]sNeuron, L.Width)

			for x := 0; x < L.Width; x++ {
				src := L.Neurons[y][x]

				sn := sNeuron{
					Bias: float64(any(src.Bias).(T)),
					Act:  src.Activation,
					In:   make([]sConn, len(src.Inputs)),
				}

				for k, c := range src.Inputs {
					sn.In[k] = sConn{
						L: c.SourceLayer,
						X: c.SourceX,
						Y: c.SourceY,
						W: float64(any(c.Weight).(T)),
					}
				}
				row[x] = sn
			}
			sl.Neurons[y] = row
		}

		s.Layers[li] = sl
	}

	return s
}

// fromS rebuilds a Network from the flattened form.
func (n *Network[T]) FromS(s sNet) error {

	// Type check!
	if s.Type != "" && n.TypeName != "" && s.Type != n.TypeName {
		return fmt.Errorf("type mismatch: model is '%s' but this network is '%s'", s.Type, n.TypeName)
	}

	n.TypeName = s.Type

	n.Layers = make([]Grid[T], len(s.Layers))

	for li, sl := range s.Layers {
		if sl.W == 0 || sl.H == 0 {
			return fmt.Errorf("layer %d has zero width or height", li)
		}

		L := Grid[T]{
			Width:   sl.W,
			Height:  sl.H,
			Neurons: make([][]*Neuron[T], sl.H),
		}

		for y := 0; y < sl.H; y++ {
			if len(sl.Neurons[y]) != sl.W {
				return fmt.Errorf("layer %d row %d width mismatch", li, y)
			}

			row := make([]*Neuron[T], sl.W)

			for x := 0; x < sl.W; x++ {
				sn := sl.Neurons[y][x]
				nn := &Neuron[T]{
					Bias:       T(sn.Bias),
					Activation: sn.Act,
					Inputs:     make([]Connection[T], len(sn.In)),
				}
				for k, c := range sn.In {
					nn.Inputs[k] = Connection[T]{
						SourceLayer: c.L,
						SourceX:     c.X,
						SourceY:     c.Y,
						Weight:      T(c.W),
					}
				}
				row[x] = nn
			}
			L.Neurons[y] = row
		}

		n.Layers[li] = L
	}

	n.OutputLayer = len(n.Layers) - 1
	return nil
}

/*──────────────────────────── public API ───────────────────────────────*/

// SaveJSON writes the full topology + weights of the network.
func (n *Network[T]) SaveJSON(path string) error {
	b, err := json.MarshalIndent(n.ToS(), "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func (n *Network[T]) LoadJSON(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var s sNet
	if err = json.Unmarshal(b, &s); err != nil {
		return err
	}
	return n.FromS(s)
}

func (n *Network[T]) MarshalJSONModel() ([]byte, error) {
	return json.Marshal(n.ToS())
}

func (n *Network[T]) UnmarshalJSONModel(b []byte) error {
	var sn sNet
	if err := json.Unmarshal(b, &sn); err != nil {
		return err
	}
	return n.FromS(sn)
}

// LoadNamedNetworkFromJSONString loads a Paragon network from a raw JSON string.
// It dynamically detects the type and returns the network as `any`.
func LoadNamedNetworkFromJSONString(jsonStr string) (any, error) {
	var s sNet
	if err := json.Unmarshal([]byte(jsonStr), &s); err != nil {
		return nil, fmt.Errorf("json unmarshal failed: %w", err)
	}

	switch s.Type {
	case "int":
		n := &Network[int]{TypeName: s.Type}
		return n, n.FromS(s)
	case "int8":
		n := &Network[int8]{TypeName: s.Type}
		return n, n.FromS(s)
	case "int16":
		n := &Network[int16]{TypeName: s.Type}
		return n, n.FromS(s)
	case "int32":
		n := &Network[int32]{TypeName: s.Type}
		return n, n.FromS(s)
	case "int64":
		n := &Network[int64]{TypeName: s.Type}
		return n, n.FromS(s)

	case "uint":
		n := &Network[uint]{TypeName: s.Type}
		return n, n.FromS(s)
	case "uint8":
		n := &Network[uint8]{TypeName: s.Type}
		return n, n.FromS(s)
	case "uint16":
		n := &Network[uint16]{TypeName: s.Type}
		return n, n.FromS(s)
	case "uint32":
		n := &Network[uint32]{TypeName: s.Type}
		return n, n.FromS(s)
	case "uint64":
		n := &Network[uint64]{TypeName: s.Type}
		return n, n.FromS(s)

	case "float32":
		n := &Network[float32]{TypeName: s.Type}
		return n, n.FromS(s)
	case "float64":
		n := &Network[float64]{TypeName: s.Type}
		return n, n.FromS(s)

	default:
		return nil, fmt.Errorf("unsupported network type: %s", s.Type)
	}
}

func LoadNamedNetworkFromJSONFile(path string) (any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}
	return LoadNamedNetworkFromJSONString(string(data))
}
