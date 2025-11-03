// attn.go
package paragon

import (
	"math"
	"math/rand"
)

/* ─────────────────────────────────────────────────────────────────────────────
   Attention config & params (multi-head)
   ───────────────────────────────────────────────────────────────────────────── */

type AttnConfig[T Numeric] struct {
	// Core
	Heads    int     // number of heads (defaults to 1 if <=0)
	DK       int     // per-head size (dk)
	UseWo    bool    // apply output projection ( (heads*dk) x Hcurr )
	Share    string  // "layer" | "per-slice"
	Dropout  float32 // reserved for train-time (no-op here)
	PosEnc2D bool    // add tiny 2D positional encoding to tokens
	UseNorm  bool    // unit-norm the concatenated zbar (pre-Wo)

	// Tunables (default if zero):
	PosEncAmp float64 // default 1e-2
	NormEps   float64 // default 1e-6

}

type AttnParams[T Numeric] struct {
	// Per-head projections; shapes:
	//  Wq[h],Wk[h],Wv[h] ∈ R^{dk}   (scalar token -> dk)
	//  Wo ∈ R^{(heads*dk) x Hcurr}  (post-concat projection), if UseWo
	Wq    [][]T
	Wk    [][]T
	Wv    [][]T
	Wo    []T // len = heads*dk*Hcurr (row-major: outIdx*Hd + t)
	DK    int
	Heads int
}

/* ─────────────────────────────────────────────────────────────────────────────
   Small utils
   ───────────────────────────────────────────────────────────────────────────── */

func f64[T Numeric](v T) float64 { return float64(any(v).(T)) }
func tOf[T Numeric](x float64) T { return T(x) }

func randInitVec[T Numeric](n int, scale float64) []T {
	out := make([]T, n)
	for i := 0; i < n; i++ {
		out[i] = tOf[T]((rand.Float64()*2 - 1) * scale)
	}
	return out
}

func defPosEncAmp[T Numeric](cfg *AttnConfig[T]) float64 {
	if cfg == nil || cfg.PosEncAmp == 0 {
		return 1e-2
	}
	return cfg.PosEncAmp
}

func defNormEps[T Numeric](cfg *AttnConfig[T]) float64 {
	if cfg == nil || cfg.NormEps == 0 {
		return 1e-6
	}
	return cfg.NormEps
}

func defHeads[T Numeric](cfg *AttnConfig[T]) int {
	if cfg == nil || cfg.Heads <= 0 {
		return 1
	}
	return cfg.Heads
}

func addTinyPosEnc2DWithAmp[T Numeric](x []float64, prev *Grid[T], amp float64) {
	if prev.Width == 0 || prev.Height == 0 || len(x) != prev.Width*prev.Height || amp == 0 {
		return
	}
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
	for i := range s {
		row := s[i]
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

// Correct dot product (your old version didn’t multiply b)
func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

/* ─────────────────────────────────────────────────────────────────────────────
   Param placement / init
   ───────────────────────────────────────────────────────────────────────────── */

func ensureAttnParams[T Numeric](curr *Grid[T], prev *Grid[T], col int) *AttnParams[T] {
	cfg := curr.Attn
	if cfg == nil || cfg.DK <= 0 {
		return nil
	}
	H := defHeads(cfg)
	dk := cfg.DK

	mk := func() *AttnParams[T] {
		p := &AttnParams[T]{DK: dk, Heads: H}
		p.Wq = make([][]T, H)
		p.Wk = make([][]T, H)
		p.Wv = make([][]T, H)
		scale := 1.0 / math.Sqrt(float64(dk))
		for h := 0; h < H; h++ {
			p.Wq[h] = randInitVec[T](dk, scale)
			p.Wk[h] = randInitVec[T](dk, scale)
			p.Wv[h] = randInitVec[T](dk, scale)
		}
		if cfg.UseWo {
			p.Wo = randInitVec[T](H*dk*curr.Height, scale)
		}
		return p
	}

	if cfg.Share == "per-slice" {
		if curr.attnSlices == nil {
			curr.attnSlices = map[int]*AttnParams[T]{}
		}
		if p, ok := curr.attnSlices[col]; ok {
			return p
		}
		p := mk()
		curr.attnSlices[col] = p
		return p
	}

	// default: "layer"
	if curr.attnLayer != nil {
		return curr.attnLayer
	}
	p := mk()
	curr.attnLayer = p
	return p
}

/* ─────────────────────────────────────────────────────────────────────────────
   Forward (multi-head)
   ───────────────────────────────────────────────────────────────────────────── */

func (n *Network[T]) forwardAttnColumn(layerIdx int, col int, isReplay bool) {
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
	H := params.Heads
	dk := params.DK
	if N == 0 || dk == 0 || H <= 0 {
		return
	}
	Hd := H * dk // concat size

	// 1) Tokens
	X := make([]float64, N)
	{
		idx := 0
		for y := 0; y < prev.Height; y++ {
			for x := 0; x < prev.Width; x++ {
				X[idx] = f64(prev.Neurons[y][x].Value)
				idx++
			}
		}
		if cfg.PosEnc2D {
			addTinyPosEnc2DWithAmp(X, prev, defPosEncAmp(cfg))
		}
	}

	// 2) Multi-head projections & attention
	zbarCat := make([]float64, Hd) // concat of per-head pooled features
	headOffset := 0

	for h := 0; h < H; h++ {
		// Q,K,V ∈ [N][dk]
		Q := make([][]float64, N)
		K := make([][]float64, N)
		V := make([][]float64, N)
		wqh := params.Wq[h]
		wkh := params.Wk[h]
		wvh := params.Wv[h]

		for i := 0; i < N; i++ {
			q := make([]float64, dk)
			k := make([]float64, dk)
			v := make([]float64, dk)
			x := X[i]
			for j := 0; j < dk; j++ {
				q[j] = x * f64(wqh[j])
				k[j] = x * f64(wkh[j])
				v[j] = x * f64(wvh[j])
			}
			Q[i], K[i], V[i] = q, k, v
		}

		// Scores / softmax
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

		// Z = A V  → pool to zbar_h
		zbarH := make([]float64, dk)
		for i := 0; i < N; i++ {
			zi := make([]float64, dk)
			Ai := A[i]
			for j := 0; j < N; j++ {
				a := Ai[j]
				if a == 0 {
					continue
				}
				vj := V[j]
				for t := 0; t < dk; t++ {
					zi[t] += a * vj[t]
				}
			}
			for t := 0; t < dk; t++ {
				zbarH[t] += zi[t]
			}
		}
		invN := 1.0 / float64(N)
		for t := 0; t < dk; t++ {
			zbarH[t] *= invN
		}

		// concat
		copy(zbarCat[headOffset:headOffset+dk], zbarH)
		headOffset += dk
	}

	// Optional norm over concatenated vector
	if cfg.UseNorm {
		eps := defNormEps(cfg)
		mu := 0.0
		for t := 0; t < Hd; t++ {
			mu += zbarCat[t]
		}
		mu /= float64(Hd)
		vv := 0.0
		for t := 0; t < Hd; t++ {
			d := zbarCat[t] - mu
			vv += d * d
		}
		vv /= float64(Hd)
		invStd := 1.0 / math.Sqrt(vv+eps)
		for t := 0; t < Hd; t++ {
			zbarCat[t] = (zbarCat[t] - mu) * invStd
		}
	}

	// 3) Project to column
	out := make([]float64, curr.Height)
	if cfg.UseWo && len(params.Wo) == Hd*curr.Height {
		for y := 0; y < curr.Height; y++ {
			sum := 0.0
			base := y * Hd
			for t := 0; t < Hd; t++ {
				sum += zbarCat[t] * f64(params.Wo[base+t])
			}
			out[y] = sum
		}
	} else {
		// simple fallback
		sum := 0.0
		for t := 0; t < Hd; t++ {
			sum += zbarCat[t]
		}
		for y := 0; y < curr.Height; y++ {
			alpha := float64(y) / float64(max(1, curr.Height-1))
			out[y] = (1.0-alpha)*sum + 0.5*alpha*sum
		}
	}

	// 4) Activation (+ optional replay gain)

	for y := 0; y < curr.Height; y++ {
		val := tOf[T](out[y])
		val = ApplyActivationGeneric(val, curr.Neurons[y][col].Activation)

		curr.Neurons[y][col].Value = val
	}
}

/* ─────────────────────────────────────────────────────────────────────────────
   Backward (multi-head)
   ───────────────────────────────────────────────────────────────────────────── */

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
	H := params.Heads
	dk := params.DK
	if N == 0 || dk == 0 || H <= 0 {
		return
	}
	Hd := H * dk

	dy := make([]float64, curr.Height)
	for y := 0; y < curr.Height; y++ {
		v := float64(err[layerIdx][y][col])
		dy[y] = v
	}

	// Recompute intermediates (tokens)
	X := make([]float64, N)
	{
		idx := 0
		for y := 0; y < prev.Height; y++ {
			for x := 0; x < prev.Width; x++ {
				X[idx] = f64(prev.Neurons[y][x].Value)
				idx++
			}
		}
		if cfg.PosEnc2D {
			addTinyPosEnc2DWithAmp(X, prev, defPosEncAmp(cfg))
		}
	}

	// Per-head caches (to avoid re-alloc inside loops)
	type headCache struct {
		Q [][]float64
		K [][]float64
		V [][]float64
		A [][]float64
		Z [][]float64
		// pooled head feature
		zbar []float64
	}
	heads := make([]headCache, H)

	// Forward recompute per head
	for h := 0; h < H; h++ {
		wqh := params.Wq[h]
		wkh := params.Wk[h]
		wvh := params.Wv[h]

		Q := make([][]float64, N)
		K := make([][]float64, N)
		V := make([][]float64, N)
		for i := 0; i < N; i++ {
			q := make([]float64, dk)
			k := make([]float64, dk)
			v := make([]float64, dk)
			x := X[i]
			for t := 0; t < dk; t++ {
				q[t] = x * f64(wqh[t])
				k[t] = x * f64(wkh[t])
				v[t] = x * f64(wvh[t])
			}
			Q[i], K[i], V[i] = q, k, v
		}
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

		Z := make([][]float64, N)
		zbar := make([]float64, dk)
		for i := 0; i < N; i++ {
			zi := make([]float64, dk)
			Ai := A[i]
			for j := 0; j < N; j++ {
				a := Ai[j]
				if a == 0 {
					continue
				}
				vj := V[j]
				for t := 0; t < dk; t++ {
					zi[t] += a * vj[t]
				}
			}
			Z[i] = zi
			for t := 0; t < dk; t++ {
				zbar[t] += zi[t]
			}
		}
		invN := 1.0 / float64(N)
		for t := 0; t < dk; t++ {
			zbar[t] *= invN
		}
		heads[h] = headCache{Q: Q, K: K, V: V, A: A, Z: Z, zbar: zbar}
	}

	// Concat zbar for norm + projection
	zbarCat := make([]float64, Hd)
	off := 0
	for h := 0; h < H; h++ {
		copy(zbarCat[off:off+dk], heads[h].zbar)
		off += dk
	}

	// Norm (cache)
	useNorm := cfg.UseNorm
	eps := defNormEps(cfg)
	var mu, invStd float64
	var yhat []float64
	if useNorm {
		mu = 0.0
		for t := 0; t < Hd; t++ {
			mu += zbarCat[t]
		}
		mu /= float64(Hd)
		vv := 0.0
		for t := 0; t < Hd; t++ {
			d := zbarCat[t] - mu
			vv += d * d
		}
		vv /= float64(Hd)
		invStd = 1.0 / math.Sqrt(vv+eps)
		yhat = make([]float64, Hd)
		for t := 0; t < Hd; t++ {
			yhat[t] = (zbarCat[t] - mu) * invStd
		}
	}

	// Grad buffers
	dZbarCat := make([]float64, Hd) // gradient wrt (concat) pre-norm vector (or pre-Wo if no norm)
	var gWo []float64
	if cfg.UseWo && len(params.Wo) == Hd*curr.Height {
		gWo = make([]float64, Hd*curr.Height)
	}

	// Back from projection
	if cfg.UseWo && gWo != nil {
		for y := 0; y < curr.Height; y++ {
			gy := dy[y]
			base := y * Hd
			for t := 0; t < Hd; t++ {
				gWo[base+t] += zbarCat[t] * gy
				dZbarCat[t] += f64(params.Wo[base+t]) * gy
			}
		}
	} else {
		// fallback projection used in forward
		sumScale := 0.0
		den := float64(max(1, curr.Height-1))
		for y := 0; y < curr.Height; y++ {
			alpha := float64(y) / den
			sumScale += dy[y] * (1.0 - 0.5*alpha)
		}
		for t := 0; t < Hd; t++ {
			dZbarCat[t] += sumScale
		}
	}

	// Back through norm if used: yhat = Norm(zbarCat)
	if useNorm {
		g := dZbarCat
		meanG := 0.0
		meanGY := 0.0
		for t := 0; t < Hd; t++ {
			meanG += g[t]
			meanGY += g[t] * yhat[t]
		}
		meanG /= float64(Hd)
		meanGY /= float64(Hd)
		for t := 0; t < Hd; t++ {
			dZbarCat[t] = (g[t] - meanG - yhat[t]*meanGY) * invStd
		}
	}

	// Split dZbarCat to per-head dZbar
	dZbarPerHead := make([][]float64, H)
	off = 0
	for h := 0; h < H; h++ {
		dZbarPerHead[h] = make([]float64, dk)
		copy(dZbarPerHead[h], dZbarCat[off:off+dk])
		off += dk
	}

	// Now back through each head: Zbar_h = mean_i Z_h[i]
	// Accumulate grads for Wq/Wk/Wv and X
	dQ := make([][]float64, N)
	dK := make([][]float64, N)
	for i := 0; i < N; i++ {
		dQ[i] = make([]float64, dk) // per head scratch; will reuse
		dK[i] = make([]float64, dk)
	}
	dV := make([][]float64, N)
	for j := 0; j < N; j++ {
		dV[j] = make([]float64, dk)
	}
	dS := make([][]float64, N) // per head softmax grad; reused
	for i := 0; i < N; i++ {
		dS[i] = make([]float64, N)
	}
	dX := make([]float64, N)

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

	// Grad buffers for params
	gWq := make([][]float64, H)
	gWk := make([][]float64, H)
	gWv := make([][]float64, H)
	for h := 0; h < H; h++ {
		gWq[h] = make([]float64, dk)
		gWk[h] = make([]float64, dk)
		gWv[h] = make([]float64, dk)
	}

	invN := 1.0 / float64(N)

	for h := 0; h < H; h++ {
		Q := heads[h].Q
		K := heads[h].K
		V := heads[h].V
		A := heads[h].A
		Z := heads[h].Z
		dZbar := dZbarPerHead[h]

		// 1) dZ[i] += dZbar / N
		for i := 0; i < N; i++ {
			for t := 0; t < dk; t++ {
				Z[i][t] = dZbar[t] * invN // reuse Z as dZ to save alloc
			}
		}

		// 2) Back through Z = A V
		//    dV += Aᵀ dZ ; dA += dZ Vᵀ
		for i := 0; i < N; i++ {
			zi := Z[i] // actually dZ now
			Ai := A[i]
			for j := 0; j < N; j++ {
				aij := Ai[j]
				if aij == 0 {
					continue
				}
				for t := 0; t < dk; t++ {
					dV[j][t] += aij * zi[t]
				}
				dS[i][j] = dot(zi, V[j]) // store into dS for next step
			}
		}

		// 3) Back through softmax rows: dS -> (dA)
		for i := 0; i < N; i++ {
			row := dS[i] // currently holds dA[i][*]
			sum := 0.0
			for j := 0; j < N; j++ {
				sum += row[j] * A[i][j]
			}
			for j := 0; j < N; j++ {
				row[j] = (row[j] - sum) * A[i][j] // now row = dS[i][*]
			}
		}

		// 4) Back through S = (QKᵀ)/√dk
		invScale := 1.0 / math.Sqrt(float64(dk))
		for i := 0; i < N; i++ {
			for j := 0; j < N; j++ {
				g := dS[i][j] * invScale
				if g == 0 {
					continue
				}
				for t := 0; t < dk; t++ {
					dQ[i][t] += g * K[j][t]
					dK[j][t] += g * Q[i][t]
				}
			}
		}

		// 5) Back to params and X (per head)
		for i := 0; i < N; i++ {
			x := X[i]
			for t := 0; t < dk; t++ {
				gWq[h][t] += x * dQ[i][t]
				gWk[h][t] += x * dK[i][t]
				gWv[h][t] += x * dV[i][t]
				dX[i] += dQ[i][t]*f64(params.Wq[h][t]) +
					dK[i][t]*f64(params.Wk[h][t]) +
					dV[i][t]*f64(params.Wv[h][t])
			}
		}

		// zero per-head scratch for next head
		for i := 0; i < N; i++ {
			for t := 0; t < dk; t++ {
				dQ[i][t] = 0
				dK[i][t] = 0
				dV[i][t] = 0
			}
		}
		for i := 0; i < N; i++ {
			for j := 0; j < N; j++ {
				dS[i][j] = 0
			}
		}
	}

	// Apply clipping + SGD updates
	for h := 0; h < H; h++ {
		for t := 0; t < dk; t++ {
			params.Wq[h][t] += T(lr * clip(gWq[h][t]))
			params.Wk[h][t] += T(lr * clip(gWk[h][t]))
			params.Wv[h][t] += T(lr * clip(gWv[h][t]))
		}
	}
	if gWo != nil {
		for i := 0; i < len(gWo); i++ {
			params.Wo[i] += T(lr * clip(gWo[i]))
		}
	}

	// Scatter to prev layer error
	iTok := 0
	for y := 0; y < prev.Height; y++ {
		for x := 0; x < prev.Width; x++ {
			err[layerIdx-1][y][x] += T(dX[iTok])
			iTok++
		}
	}
}
