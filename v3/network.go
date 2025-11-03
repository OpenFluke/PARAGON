package paragon

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
)

// Grid represents a 2D layer of neurons
type Grid[T Numeric] struct {
	Width   int            // Number of neurons along x-axis
	Height  int            // Number of neurons along y-axis
	Neurons [][]*Neuron[T] // 2D slice of neuron pointers

	CachedOutputs        []T   // used for entropy-based replay gating
	CachedOutputsHistory [][]T // History of activations for temporal replay (3 passes)

	SliceTypes []string
	// Attention config + params
	Attn       *AttnConfig[T]         // nil ⇒ no attention configured
	attnLayer  *AttnParams[T]         // shared across attn columns if Share="layer"
	attnSlices map[int]*AttnParams[T] // per-column params if Share="per-slice"

}

// Neuron represents a single unit in the grid
type Neuron[T Numeric] struct {
	ID         int             // Unique ID
	Value      T               // Current activation
	Bias       T               // Bias term
	Activation string          // e.g., "relu", "sigmoid"
	Type       string          // "dense", extensible
	Inputs     []Connection[T] // Incoming connections
	IsNew      bool            // Flag for newly added neurons
	Dimension  *Network[T]     // Sub-network (if dimensional neuron)
	RevValue   T               // Reverse propagated value
}

// Connection defines a link from a source neuron
type Connection[T Numeric] struct {
	SourceLayer int // Layer index of source neuron
	SourceX     int // X-coordinate
	SourceY     int // Y-coordinate
	Weight      T   // Weight of the connection
}

// Network encapsulates the entire model
type Network[T Numeric] struct {
	GrowthHistory []GrowthLog `json:"growth_history,omitempty"`
	Layers        []Grid[T]
	InputLayer    int
	OutputLayer   int
	Debug         bool
	TypeName      string
	Performance   *ADHDPerformance
	Composite     *CompositePerformance
	ReplayStats   map[int][]int // layer index → replay counts per sample
	WebGPUNative  bool
	SCALE         int64
}

// NewNetwork initializes a network with specified layer sizes, activations, and connectivity
func NewNetwork[T Numeric](
	layerSizes []struct{ Width, Height int },
	activations []string,
	fullyConnected []bool,
	seed ...int64, // Optional seed for consistent initialization
) (*Network[T], error) {
	if len(layerSizes) != len(activations) || len(layerSizes) != len(fullyConnected) {
		return nil, fmt.Errorf("mismatched layer sizes, activations, or connectivity settings")
	}

	// Set random seed if provided
	if len(seed) > 0 {
		rand.Seed(seed[0])
	}

	n := &Network[T]{
		TypeName:      reflect.TypeOf(*new(T)).Name(),
		Layers:        make([]Grid[T], len(layerSizes)),
		InputLayer:    0,
		OutputLayer:   len(layerSizes) - 1,
		Performance:   NewADHDPerformance(),
		ReplayStats:   make(map[int][]int),
		GrowthHistory: []GrowthLog{},
	}

	idCounter := 0
	for i, size := range layerSizes {
		grid := Grid[T]{
			Width:      size.Width,
			Height:     size.Height,
			Neurons:    make([][]*Neuron[T], size.Height),
			SliceTypes: make([]string, size.Width),
		}

		for x := 0; x < size.Width; x++ {
			grid.SliceTypes[x] = "dense"
		}

		for y := 0; y < size.Height; y++ {
			grid.Neurons[y] = make([]*Neuron[T], size.Width)
			for x := 0; x < size.Width; x++ {
				var zero T
				grid.Neurons[y][x] = &Neuron[T]{
					ID:         idCounter,
					Bias:       zero,
					Activation: activations[i],
					Type:       "dense",
				}
				idCounter++
			}
		}
		n.Layers[i] = grid
	}

	n.ConnectLayers(fullyConnected)
	return n, nil
}

func (n *Network[T]) SetSliceTypes(layerIdx int, types []string) {
	g := &n.Layers[layerIdx]
	if len(types) != g.Width {
		panic("SetSliceTypes: len(types) must equal layer width")
	}
	copy(g.SliceTypes, types)
}

// -----------------------------------------------------------------------------
// getFullyConnectedInputs   (safe version – no Int63n overflow)
// -----------------------------------------------------------------------------
func (n *Network[T]) getFullyConnectedInputs(
	prevLayerIdx int,
	prevLayer Grid[T],
) []Connection[T] {

	// -------------------------------------------------------------------------
	// pre-amble
	// -------------------------------------------------------------------------
	numInputs := prevLayer.Width * prevLayer.Height
	conns := make([]Connection[T], numInputs)
	idx := 0

	var w T
	kind := reflect.TypeOf(w).Kind()

	// -------------------------------------------------------------------------
	// 1) Floating-point networks      → weights ∈ (-1, +1)
	// -------------------------------------------------------------------------
	if kind == reflect.Float32 || kind == reflect.Float64 {
		for y := 0; y < prevLayer.Height; y++ {
			for x := 0; x < prevLayer.Width; x++ {
				w = T(rand.Float64()*2 - 1)
				conns[idx] = Connection[T]{prevLayerIdx, x, y, w}
				idx++
			}
		}
		return conns
	}

	// -------------------------------------------------------------------------
	// 2) Integer networks             → weights ∈ [-scale, +scale] (or 0…scale)
	//    The clamp below guarantees  1 ≤ bound ≤ math.MaxInt64
	// -------------------------------------------------------------------------
	var (
		isUnsigned bool
		typeMax    int64
	)

	switch kind {
	case reflect.Int8:
		typeMax = math.MaxInt8
	case reflect.Int16:
		typeMax = math.MaxInt16
	case reflect.Int32:
		typeMax = math.MaxInt32
	case reflect.Int, reflect.Int64:
		typeMax = math.MaxInt64
	case reflect.Uint8:
		typeMax, isUnsigned = int64(math.MaxUint8), true
	case reflect.Uint16:
		typeMax, isUnsigned = int64(math.MaxUint16), true
	case reflect.Uint32:
		typeMax, isUnsigned = int64(math.MaxUint32), true
	case reflect.Uint, reflect.Uint64:
		// uint64 won't fit in Int63n’s bound arg; clamp to int64 range
		typeMax, isUnsigned = math.MaxInt64, true
	default:
		panic("unsupported Numeric type")
	}

	// He/Glorot-style scaling:  typeMax / √fan-in
	scale := typeMax / int64(math.Sqrt(float64(numInputs)))
	if scale < 1 {
		scale = 1
	}

	// -------- HARD CAP (safe) ----------------------------------------------
	// Ensure  2*scale + 1  never exceeds math.MaxInt64 without multiplying.
	// Max allowed scale = (MaxInt64-1) / 2
	const capScale = (int64(math.MaxInt64) - 1) / 2
	if scale > capScale {
		scale = capScale
	}
	// -----------------------------------------------------------------------

	for y := 0; y < prevLayer.Height; y++ {
		for x := 0; x < prevLayer.Width; x++ {
			var v int64
			if isUnsigned {
				// [0 … scale]
				v = rand.Int63n(scale + 1)
			} else {
				// [-scale … +scale]
				bound := 2*scale + 1 // guaranteed 1 ≤ bound ≤ MaxInt64
				v = rand.Int63n(bound) - scale
			}
			conns[idx] = Connection[T]{prevLayerIdx, x, y, T(v)}
			idx++
		}
	}
	return conns
}

func (n *Network[T]) getLocalConnections(
	srcLayer, centerX, centerY, size, stride int,
) []Connection[T] {
	half := size / 2
	srcGrid := n.Layers[srcLayer]
	conns := make([]Connection[T], 0, size*size)

	for dy := -half; dy <= half; dy++ {
		for dx := -half; dx <= half; dx++ {
			srcX := centerX*stride + dx
			srcY := centerY*stride + dy

			if srcX >= 0 && srcX < srcGrid.Width && srcY >= 0 && srcY < srcGrid.Height {
				var weight T
				switch any(weight).(type) {
				case float32, float64:
					weight = T(rand.NormFloat64() * math.Sqrt(2.0/float64(size*size)))
				default:
					weight = T(1) // Integer fallback
				}

				conns = append(conns, Connection[T]{
					SourceLayer: srcLayer,
					SourceX:     srcX,
					SourceY:     srcY,
					Weight:      weight,
				})
			}
		}
	}
	return conns
}

// ConnectLayers sets up connections based on connectivity settings
func (n *Network[T]) ConnectLayers(fullyConnected []bool) {
	for l := 1; l < len(n.Layers); l++ {
		prevLayer := n.Layers[l-1]
		currLayer := n.Layers[l]

		for y := 0; y < currLayer.Height; y++ {
			for x := 0; x < currLayer.Width; x++ {
				neuron := currLayer.Neurons[y][x]
				if fullyConnected[l] {
					neuron.Inputs = n.getFullyConnectedInputs(l-1, prevLayer)
				} else {
					neuron.Inputs = n.getLocalConnections(l-1, x, y, 5, 1)
				}
			}
		}
	}
}

// forwardLayer computes the forward pass for a layer
func (n *Network[T]) forwardLayer(l int, isReplay bool) {
	curr := n.Layers[l]
	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {
			kind := "dense"
			if curr.SliceTypes != nil && x < len(curr.SliceTypes) && curr.SliceTypes[x] != "" {
				kind = curr.SliceTypes[x]
			}

			switch kind {
			case "attn":
				n.forwardAttnColumn(l, x, isReplay)
			default:
				neuron := curr.Neurons[y][x]
				sum := neuron.Bias
				for _, c := range neuron.Inputs {
					src := n.Layers[c.SourceLayer].Neurons[c.SourceY][c.SourceX]
					sum += src.Value * c.Weight
				}
				value := ApplyActivationGeneric(sum, neuron.Activation)
				if isReplay {
					// Apply a scaling factor during replay to alter the output
					value = T(float64(value) * 1.1)
				}
				neuron.Value = value
			}

		}
	}
}

func (n *Network[T]) backwardLayer(
	l int,
	err [][][]T,
	lr float64,
	clipUpper T,
	clipLower T,
	isReplay bool,
) {
	curr := n.Layers[l]
	prev := n.Layers[l-1]

	for y := 0; y < curr.Height; y++ {
		for x := 0; x < curr.Width; x++ {

			// ── NEW: branch by slice type (column kind)
			kind := "dense"
			if curr.SliceTypes != nil && x < len(curr.SliceTypes) && curr.SliceTypes[x] != "" {
				kind = curr.SliceTypes[x]
			}
			if kind == "attn" {
				// Process the WHOLE column once (on the first row), since attention
				// gradients couple all y in this column.
				if y == 0 {
					n.backwardAttnColumn(l, x, err, lr, clipUpper, clipLower, isReplay)
				}
				continue // skip dense-style updates for attn columns
			}

			neuron := curr.Neurons[y][x]
			localT := err[l][y][x]
			localF := float64(localT)

			// Apply scaling factor during replay
			if isReplay {
				localF *= 1.1 // Scale gradient by 1.1 during replay
			}

			// ─── Update bias ───
			deltaBias := lr * localF
			neuron.Bias += T(deltaBias)

			// ─── Update weights and propagate error ───
			for i, c := range neuron.Inputs {
				src := prev.Neurons[c.SourceY][c.SourceX]
				srcF := float64(src.Value)

				// raw gradient in float
				grad := localF * srcF

				// clip in float
				if grad > float64(clipUpper) {
					grad = float64(clipUpper)
				} else if grad < float64(clipLower) {
					grad = float64(clipLower)
				}

				// compute and apply weight delta
				deltaW := lr * grad
				neuron.Inputs[i].Weight += T(deltaW)

				// propagate error back to prev layer
				if l-1 > 0 {
					backErr := localF * float64(c.Weight)
					err[l-1][c.SourceY][c.SourceX] += T(backErr)
				}
			}
		}
	}

	// ─── Apply activation-derivative to chain rule ───
	if l-1 > 0 {
		for y := 0; y < prev.Height; y++ {
			for x := 0; x < prev.Width; x++ {
				prevErr := err[l-1][y][x]
				deriv := ActivationDerivativeGeneric(
					prev.Neurons[y][x].Value,
					prev.Neurons[y][x].Activation,
				)
				err[l-1][y][x] = prevErr * deriv
			}
		}
	}
}

// ComputeLoss calculates the loss for a sample
func (n *Network[T]) ComputeLoss(target [][]float64) float64 {
	loss := 0.0
	outputLayer := n.Layers[n.OutputLayer]

	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			outputVal := float64(any(outputLayer.Neurons[y][x].Value).(T))
			targetVal := target[y][x]

			// Prevent log(0)
			if outputVal <= 0 {
				outputVal = 1e-10
			}

			loss += -targetVal * math.Log(outputVal)
		}
	}
	return loss
}

// AddNeuronsToLayer adds neurons to a specified layer
func (n *Network[T]) AddNeuronsToLayer(layerIdx, numToAdd int) {
	if layerIdx <= n.InputLayer || layerIdx > n.OutputLayer {
		return
	}

	grid := &n.Layers[layerIdx]
	newHeight := grid.Height + numToAdd
	newNeurons := make([][]*Neuron[T], newHeight)
	copy(newNeurons, grid.Neurons)

	for y := grid.Height; y < newHeight; y++ {
		newNeurons[y] = make([]*Neuron[T], grid.Width)
		for x := 0; x < grid.Width; x++ {
			var zero T

			newNeuron := &Neuron[T]{
				ID:         n.getNextID(),
				Bias:       zero,
				Activation: grid.Neurons[0][0].Activation,
				Type:       "dense",
				IsNew:      true,
			}

			prevLayer := n.Layers[layerIdx-1]
			if layerIdx == 1 && len(n.Layers) == 3 {
				newNeuron.Inputs = n.getFullyConnectedInputs(layerIdx-1, prevLayer)
			} else {
				newNeuron.Inputs = n.getLocalConnections(layerIdx-1, x, y, 3, 1)
			}

			newNeurons[y][x] = newNeuron
		}
	}

	grid.Neurons = newNeurons
	grid.Height = newHeight

	// Rewire next layer if needed
	if layerIdx < n.OutputLayer {
		nextLayer := n.Layers[layerIdx+1]
		isFullyConnected := len(nextLayer.Neurons[0][0].Inputs) == grid.Width*grid.Height

		for y := 0; y < nextLayer.Height; y++ {
			for x := 0; x < nextLayer.Width; x++ {
				if isFullyConnected {
					nextLayer.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx, *grid)
				} else {
					nextLayer.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx, x, y, 3, 1)
				}
			}
		}
	}

	if n.Debug {
		fmt.Printf("Added %d neurons to layer %d, new height: %d\n", numToAdd, layerIdx, grid.Height)
	}
}

// getNextID generates a unique neuron ID
func (n *Network[T]) getNextID() int {
	maxID := 0
	for _, layer := range n.Layers {
		for _, row := range layer.Neurons {
			for _, neuron := range row {
				if neuron.ID > maxID {
					maxID = neuron.ID
				}
			}
		}
	}
	return maxID + 1
}

// ApplySoftmax normalizes output layer values
func (n *Network[T]) ApplySoftmax() {
	outputGrid := n.Layers[n.OutputLayer]
	values := make([]float64, outputGrid.Width*outputGrid.Height)
	idx := 0
	for y := 0; y < outputGrid.Height; y++ {
		for x := 0; x < outputGrid.Width; x++ {
			values[idx] = float64(any(outputGrid.Neurons[y][x].Value).(T))
			idx++
		}
	}
	softmaxValues := Softmax(values)
	idx = 0
	scale := getScaleForType[T]()
	for y := 0; y < outputGrid.Height; y++ {
		for x := 0; x < outputGrid.Width; x++ {
			// Scale softmax output to fixed-point range for integers
			var val T
			switch any(val).(type) {
			case float32, float64:
				val = T(softmaxValues[idx])
			default:
				val = T(int64(math.Round(softmaxValues[idx] * float64(scale))))
			}
			outputGrid.Neurons[y][x].Value = val
			idx++
		}
	}
}

// AddLayer inserts a new layer at the specified index with given dimensions and connectivity
func (n *Network[T]) AddLayer(layerIdx int, width, height int, activation string, fullyConnectedToPrev bool) {
	if layerIdx < n.InputLayer || layerIdx > n.OutputLayer+1 {
		return
	}
	newGrid := Grid[T]{
		Width:   width,
		Height:  height,
		Neurons: make([][]*Neuron[T], height),
	}
	idCounter := n.getNextID()
	for y := 0; y < height; y++ {
		newGrid.Neurons[y] = make([]*Neuron[T], width)
		for x := 0; x < width; x++ {
			newGrid.Neurons[y][x] = &Neuron[T]{
				ID:         idCounter,
				Bias:       0.0,
				Activation: activation,
				Type:       "dense",
			}
			idCounter++
		}
	}
	newLayers := make([]Grid[T], len(n.Layers)+1)
	copy(newLayers[:layerIdx], n.Layers[:layerIdx])
	newLayers[layerIdx] = newGrid
	copy(newLayers[layerIdx+1:], n.Layers[layerIdx:])
	n.Layers = newLayers
	n.OutputLayer = len(n.Layers) - 1

	if layerIdx > n.InputLayer {
		prevLayer := n.Layers[layerIdx-1]
		for y := 0; y < newGrid.Height; y++ {
			for x := 0; x < newGrid.Width; x++ {
				if fullyConnectedToPrev {
					newGrid.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx-1, prevLayer)
				} else {
					newGrid.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx-1, x, y, 5, 1)
				}
			}
		}
	}
	if layerIdx < n.OutputLayer {
		nextLayer := n.Layers[layerIdx+1]
		isFullyConnected := layerIdx == n.OutputLayer-1
		for y := 0; y < nextLayer.Height; y++ {
			for x := 0; x < nextLayer.Width; x++ {
				if isFullyConnected {
					nextLayer.Neurons[y][x].Inputs = n.getFullyConnectedInputs(layerIdx, newGrid)
				} else {
					nextLayer.Neurons[y][x].Inputs = n.getLocalConnections(layerIdx, x, y, 5, 1)
				}
			}
		}
	}
	if n.Debug {
		fmt.Printf("Added new layer at index %d with dimensions %dx%d\n", layerIdx, width, height)
	}
}

// network.go
/*func (n *Network) SetLayerDimension(layerIdx int, subNetwork *Network) {
	if layerIdx < 0 || layerIdx >= len(n.Layers) {
		panic(fmt.Sprintf("invalid layer index: %d", layerIdx))
	}
	n.Layers[layerIdx].Dimension = subNetwork
}*/

func (n *Network[T]) GetOutput() []float64 {
	outputLayer := n.Layers[n.OutputLayer]
	// Return all neuron values from the output layer (Width × Height)
	output := make([]float64, outputLayer.Width*outputLayer.Height)

	idx := 0
	for y := 0; y < outputLayer.Height; y++ {
		for x := 0; x < outputLayer.Width; x++ {
			output[idx] = float64(any(outputLayer.Neurons[y][x].Value).(T))
			idx++
		}
	}

	return output
}

// GetOutputValues returns a flattened 1D slice of all neuron values in this grid.
func (g *Grid[T]) GetOutputValues() []float64 {
	values := make([]float64, 0, g.Width*g.Height)
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			val := float64(any(g.Neurons[y][x].Value).(T))
			values = append(values, val)
		}
	}
	return values
}

func (n *Network[T]) ResetReplayStats() {
	for k := range n.ReplayStats {
		n.ReplayStats[k] = nil
	}
}

func (n *Network[T]) LogReplay(l int, count int) {
	if _, exists := n.ReplayStats[l]; !exists {
		n.ReplayStats[l] = []int{}
	}
	n.ReplayStats[l] = append(n.ReplayStats[l], count)
}
