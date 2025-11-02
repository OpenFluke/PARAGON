// attn.go
package paragon

import (
	"math"
	"math/rand"
)

// Per-layer attention knobs
type AttnConfig[T Numeric] struct {
	DK       int     // head size (e.g., 32/48/64)
	UseWo    bool    // apply output projection
	Share    string  // "layer" | "per-slice"
	Dropout  float32 // only used in training path
	PosEnc2D bool    // (optional) tiny 2D pos-enc
}

// Trainable params
type AttnParams[T Numeric] struct {
	// token_dim = 1 (scalar from prev grid). We'll project scalar → dk per token.
	// Shapes:
	//   Wq, Wk, Wv: (1 x dk) applied per-token ⇒ Q,K,V ∈ (N x dk)
	//   Wo: (dk x Hcurr) to map pooled dk → column of length Hcurr
	Wq []T
	Wk []T
	Wv []T
	Wo []T // len = dk*Hcurr (only used if cfg.UseWo)
	DK int
}

// --- small utils ---

func f64[T Numeric](v T) float64 { return float64(any(v).(T)) }
func tOf[T Numeric](x float64) T { return T(x) }

func randInitVec[T Numeric](n int, scale float64) []T {
	out := make([]T, n)
	for i := 0; i < n; i++ {
		out[i] = tOf[T]((rand.Float64()*2 - 1) * scale)
	}
	return out
}

func addTinyPosEnc2D[T Numeric](x []float64, prev *Grid[T]) {
	// x: length N = prev.W * prev.H, scalar tokens
	// inject tiny bias by row/col sin/cos so attention can localize
	if prev.Width == 0 || prev.Height == 0 {
		return
	}
	N := prev.Width * prev.Height
	if len(x) != N {
		return
	}
	amp := 0.01
	for y := 0; y < prev.Height; y++ {
		for c := 0; c < prev.Width; c++ {
			i := y*prev.Width + c
			fy := float64(y) / float64(max(1, prev.Height-1))
			fc := float64(c) / float64(max(1, prev.Width-1))
			x[i] += amp * (math.Sin(2*math.Pi*fy) + math.Cos(2*math.Pi*fc))
		}
	}
}

func softmaxRowsInPlace(s [][]float64) {
	// s: [N][N] scores; softmax over each row j
	for i := range s {
		row := s[i]
		// stable softmax
		mx := row[0]
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		sum := 0.0
		for j, v := range row {
			e := math.Exp(v - mx)
			row[j] = e
			sum += e
		}
		if sum == 0 {
			continue
		}
		inv := 1.0 / sum
		for j := range row {
			row[j] *= inv
		}
	}
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func ensureAttnParams[T Numeric](curr *Grid[T], prev *Grid[T], col int) *AttnParams[T] {
	cfg := curr.Attn
	if cfg == nil || cfg.DK <= 0 {
		return nil
	}

	if cfg.Share == "per-slice" {
		if curr.attnSlices == nil {
			curr.attnSlices = map[int]*AttnParams[T]{}
		}
		if p, ok := curr.attnSlices[col]; ok {
			return p
		}
		p := &AttnParams[T]{DK: cfg.DK}
		// init Wq/Wk/Wv (1 x dk)
		scale := 1.0 / math.Sqrt(float64(cfg.DK))
		p.Wq = randInitVec[T](cfg.DK, scale)
		p.Wk = randInitVec[T](cfg.DK, scale)
		p.Wv = randInitVec[T](cfg.DK, scale)
		if cfg.UseWo {
			p.Wo = randInitVec[T](cfg.DK*curr.Height, scale)
		}
		curr.attnSlices[col] = p
		return p
	}

	// default: "layer"
	if curr.attnLayer != nil {
		return curr.attnLayer
	}
	p := &AttnParams[T]{DK: cfg.DK}
	scale := 1.0 / math.Sqrt(float64(cfg.DK))
	p.Wq = randInitVec[T](cfg.DK, scale)
	p.Wk = randInitVec[T](cfg.DK, scale)
	p.Wv = randInitVec[T](cfg.DK, scale)
	if cfg.UseWo {
		p.Wo = randInitVec[T](cfg.DK*curr.Height, scale)
	}
	curr.attnLayer = p
	return p
}

// Forward: computes all y in column `col`, writes curr.Neurons[y][col].Value
func (n *Network[T]) forwardAttnColumn(layerIdx int, col int, isReplay bool) {
	curr := &n.Layers[layerIdx]
	prev := &n.Layers[layerIdx-1]
	cfg := curr.Attn
	if cfg == nil || cfg.DK <= 0 {
		// No config: do nothing (or choose a fallback)
		return
	}
	params := ensureAttnParams[T](curr, prev, col)
	if params == nil {
		return
	}

	N := prev.Width * prev.Height
	dk := params.DK
	if N == 0 || dk == 0 {
		return
	}

	// 1) tokens X: scalar per prev cell → float64 scratch
	X := make([]float64, N)
	idx := 0
	for y := 0; y < prev.Height; y++ {
		for x := 0; x < prev.Width; x++ {
			X[idx] = f64(prev.Neurons[y][x].Value)
			idx++
		}
	}
	if cfg.PosEnc2D {
		addTinyPosEnc2D(X, prev)
	}

	// 2) Per-token projections (scalar * (1×dk) → (dk))
	// Q,K,V: shape [N][dk]
	Q := make([][]float64, N)
	K := make([][]float64, N)
	V := make([][]float64, N)
	for i := 0; i < N; i++ {
		q := make([]float64, dk)
		k := make([]float64, dk)
		v := make([]float64, dk)
		x := X[i]
		// Wq/Wk/Wv are (1 x dk) flattened
		for j := 0; j < dk; j++ {
			q[j] = x * f64(params.Wq[j])
			k[j] = x * f64(params.Wk[j])
			v[j] = x * f64(params.Wv[j])
		}
		Q[i], K[i], V[i] = q, k, v
	}

	// 3) Scores S = softmax( Q K^T / sqrt(dk) ) over j for each i
	scale := 1.0 / math.Sqrt(float64(dk))
	S := make([][]float64, N)
	for i := 0; i < N; i++ {
		row := make([]float64, N)
		for j := 0; j < N; j++ {
			row[j] = scale * dot(Q[i], K[j])
		}
		S[i] = row
	}
	softmaxRowsInPlace(S)

	// 4) Z = S V  (N×N @ N×dk => N×dk)
	Z := make([][]float64, N)
	for i := 0; i < N; i++ {
		zi := make([]float64, dk)
		// zi = Σ_j S[i][j] * V[j]
		for j := 0; j < N; j++ {
			w := S[i][j]
			if w == 0 {
				continue
			}
			vj := V[j]
			for t := 0; t < dk; t++ {
				zi[t] += w * vj[t]
			}
		}
		Z[i] = zi
	}

	// 5) Pool tokens → zbar (dk)
	zbar := make([]float64, dk)
	for i := 0; i < N; i++ {
		zi := Z[i]
		for t := 0; t < dk; t++ {
			zbar[t] += zi[t]
		}
	}
	invN := 1.0 / float64(N)
	for t := 0; t < dk; t++ {
		zbar[t] *= invN
	}

	// 6) Project to column of size Hcurr
	out := make([]float64, curr.Height)
	if cfg.UseWo && len(params.Wo) == dk*curr.Height {
		// y = zbar · Wo  (dk x Hcurr)
		for y := 0; y < curr.Height; y++ {
			sum := 0.0
			base := y * dk
			for t := 0; t < dk; t++ {
				sum += zbar[t] * f64(params.Wo[base+t])
			}
			out[y] = sum
		}
	} else {
		// Fallback: simple channel reduction → broadcast or banded map
		// Here: sum zbar then spread across y with a mild slope
		sum := 0.0
		for t := 0; t < dk; t++ {
			sum += zbar[t]
		}
		for y := 0; y < curr.Height; y++ {
			alpha := float64(y) / float64(max(1, curr.Height-1))
			out[y] = (1.0-alpha)*sum + alpha*sum*0.5
		}
	}

	// 7) Activation + replay gain + write column
	for y := 0; y < curr.Height; y++ {
		val := tOf[T](out[y])
		val = ApplyActivationGeneric(val, curr.Neurons[y][col].Activation)
		if isReplay {
			val = tOf[T](float64(val) * 1.1)
		}
		curr.Neurons[y][col].Value = val
	}
}

//------------training

func (n *Network[T]) backwardAttnColumn(
	layerIdx int,
	col int,
	err [][][]T,
	lr float64,
	clipUpper T,
	clipLower T,
	isReplay bool,
) {
	curr := &n.Layers[layerIdx]
	prev := &n.Layers[layerIdx-1]
	cfg := curr.Attn
	if cfg == nil || cfg.DK <= 0 {
		return
	}
	params := ensureAttnParams[T](curr, prev, col)
	if params == nil {
		return
	}

	N := prev.Width * prev.Height
	dk := params.DK
	if N == 0 || dk == 0 {
		return
	}

	// === 0) Upstream gradient dy for this column (post-activation) ===
	dy := make([]float64, curr.Height)
	for y := 0; y < curr.Height; y++ {
		v := float64(err[layerIdx][y][col])
		if isReplay {
			v *= 1.1
		}
		dy[y] = v
	}

	// === 1) Recompute forward intermediates from prev (needed for grads) ===

	// Tokens X (length N): scalar per prev cell
	X := make([]float64, N)
	idx := 0
	for y := 0; y < prev.Height; y++ {
		for x := 0; x < prev.Width; x++ {
			X[idx] = f64(prev.Neurons[y][x].Value)
			idx++
		}
	}
	if cfg.PosEnc2D {
		addTinyPosEnc2D(X, prev)
	}

	// Projections: Q,K,V in [N][dk]
	Q := make([][]float64, N)
	K := make([][]float64, N)
	V := make([][]float64, N)
	for i := 0; i < N; i++ {
		q := make([]float64, dk)
		k := make([]float64, dk)
		v := make([]float64, dk)
		x := X[i]
		for j := 0; j < dk; j++ {
			q[j] = x * f64(params.Wq[j])
			k[j] = x * f64(params.Wk[j])
			v[j] = x * f64(params.Wv[j])
		}
		Q[i], K[i], V[i] = q, k, v
	}

	// Scores S = softmax((QKᵀ)/√dk)  → A
	scale := 1.0 / math.Sqrt(float64(dk))
	A := make([][]float64, N)
	for i := 0; i < N; i++ {
		row := make([]float64, N)
		for j := 0; j < N; j++ {
			row[j] = scale * dot(Q[i], K[j])
		}
		A[i] = row
	}
	softmaxRowsInPlace(A)

	// Z = A V  (N×N @ N×dk => N×dk)
	Z := make([][]float64, N)
	for i := 0; i < N; i++ {
		zi := make([]float64, dk)
		for j := 0; j < N; j++ {
			a := A[i][j]
			if a == 0 {
				continue
			}
			vj := V[j]
			for t := 0; t < dk; t++ {
				zi[t] += a * vj[t]
			}
		}
		Z[i] = zi
	}

	// zbar = mean_i Z[i]
	zbar := make([]float64, dk)
	for i := 0; i < N; i++ {
		zi := Z[i]
		for t := 0; t < dk; t++ {
			zbar[t] += zi[t]
		}
	}
	invN := 1.0 / float64(N)
	for t := 0; t < dk; t++ {
		zbar[t] *= invN
	}

	// === 2) Backprop from y = f(zbar) ===
	// dZ, dA, dV, dQ, dK, dX, and param grads
	dZ := make([][]float64, N) // same shape as Z
	for i := 0; i < N; i++ {   // init
		dZ[i] = make([]float64, dk)
	}
	dA := make([][]float64, N)
	for i := 0; i < N; i++ {
		dA[i] = make([]float64, N)
	}
	dV := make([][]float64, N)
	for j := 0; j < N; j++ {
		dV[j] = make([]float64, dk)
	}
	dQ := make([][]float64, N)
	dK := make([][]float64, N)
	for i := 0; i < N; i++ {
		dQ[i] = make([]float64, dk)
		dK[i] = make([]float64, dk)
	}
	dX := make([]float64, N)

	// Param grads
	gWq := make([]float64, dk)
	gWk := make([]float64, dk)
	gWv := make([]float64, dk)
	var gWo []float64
	if cfg.UseWo && len(params.Wo) == dk*curr.Height {
		gWo = make([]float64, dk*curr.Height)
	}

	// (a) If UseWo: y = zbar·Wo  → dWo, dZbar
	dZbar := make([]float64, dk)
	if cfg.UseWo && gWo != nil {
		// dWo += zbar ⊗ dy  (dk x Hcurr)
		for y := 0; y < curr.Height; y++ {
			base := y * dk
			gy := dy[y]
			for t := 0; t < dk; t++ {
				gWo[base+t] += zbar[t] * gy
				dZbar[t] += f64(params.Wo[base+t]) * gy
			}
		}
	} else {
		// Fallback path used in forward:
		// out[y] = sum(zbar) * (1 - 0.5*alpha_y),  alpha_y = y/(H-1)
		sumScale := 0.0
		den := float64(max(1, curr.Height-1))
		for y := 0; y < curr.Height; y++ {
			alpha := float64(y) / den
			sumScale += dy[y] * (1.0 - 0.5*alpha)
		}
		// d/d zbar[t] of sum(zbar) is 1, so each channel gets same dZbar
		for t := 0; t < dk; t++ {
			dZbar[t] += sumScale
		}
	}

	// zbar = mean_i Z[i] → dZ[i] += dZbar / N
	for i := 0; i < N; i++ {
		for t := 0; t < dk; t++ {
			dZ[i][t] += dZbar[t] * invN
		}
	}

	// (b) Back through Z = A V
	// dV += Aᵀ dZ; dA += dZ Vᵀ
	for i := 0; i < N; i++ {
		zi := dZ[i]
		Ai := A[i]
		for j := 0; j < N; j++ {
			aij := Ai[j]
			if aij == 0 {
				continue
			}
			// dV[j] += aij * dZ[i]
			for t := 0; t < dk; t++ {
				dV[j][t] += aij * zi[t]
			}
			// dA[i][j] += dZ[i]·V[j]
			dA[i][j] += dot(zi, V[j])
		}
	}

	// (c) Back through A = softmax(S) row-wise
	// For each row i: dS_i = (dA_i - sum(dA_i * A_i)) ⊙ A_i
	dS := make([][]float64, N)
	for i := 0; i < N; i++ {
		row := make([]float64, N)
		dotRow := 0.0
		for j := 0; j < N; j++ {
			dotRow += dA[i][j] * A[i][j]
		}
		for j := 0; j < N; j++ {
			row[j] = (dA[i][j] - dotRow) * A[i][j]
		}
		dS[i] = row
	}

	// (d) Back through S = (Q Kᵀ)/√dk
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			g := dS[i][j] * scale
			if g == 0 {
				continue
			}
			// dQ[i] += g * K[j]
			for t := 0; t < dk; t++ {
				dQ[i][t] += g * K[j][t]
			}
			// dK[j] += g * Q[i]
			for t := 0; t < dk; t++ {
				dK[j][t] += g * Q[i][t]
			}
		}
	}

	// (e) Back through projections: Q = X·Wq ; K = X·Wk ; V = X·Wv
	// dW? += Xᵀ d?   and   dX += d? · W?ᵀ
	for i := 0; i < N; i++ {
		x := X[i]
		for t := 0; t < dk; t++ {
			gWq[t] += x * dQ[i][t]
			gWk[t] += x * dK[i][t]
			gWv[t] += x * dV[i][t]
			// dX accumulates from all three
			dX[i] += dQ[i][t]*f64(params.Wq[t]) +
				dK[i][t]*f64(params.Wk[t]) +
				dV[i][t]*f64(params.Wv[t])
		}
	}

	// === 3) Apply clipping and SGD updates to params ===
	clip := func(g float64) float64 {
		up := float64(clipUpper)
		lo := float64(clipLower)
		if g > up {
			return up
		}
		if g < lo {
			return lo
		}
		return g
	}

	for t := 0; t < dk; t++ {
		params.Wq[t] += T(lr * clip(gWq[t]))
		params.Wk[t] += T(lr * clip(gWk[t]))
		params.Wv[t] += T(lr * clip(gWv[t]))
	}
	if gWo != nil {
		for i := 0; i < len(gWo); i++ {
			params.Wo[i] += T(lr * clip(gWo[i]))
		}
	}

	// === 4) Scatter dX back to previous layer error tensor ===
	// err[layerIdx-1][srcY][srcX] += dX[i]
	iTok := 0
	for y := 0; y < prev.Height; y++ {
		for x := 0; x < prev.Width; x++ {
			err[layerIdx-1][y][x] += T(dX[iTok])
			iTok++
		}
	}
}
