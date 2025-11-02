// attn.go
package paragon

import (
	"math"
	"math/rand"
)

/* ─────────────────────────────────────────────────────────────────────────────
   Attention config & params
   ───────────────────────────────────────────────────────────────────────────── */

type AttnConfig[T Numeric] struct {
	DK       int     // head size
	UseWo    bool    // output projection (dk x Hcurr)
	Share    string  // "layer" | "per-slice"
	Dropout  float32 // reserved for train-time
	PosEnc2D bool    // add tiny 2D positional encoding to tokens
	UseNorm  bool    // unit-norm zbar before Wo

	// Tunables (default if zero):
	PosEncAmp float64 // default 1e-2
	NormEps   float64 // default 1e-6

	// Replay controls (fully optional):
	UseReplay   bool    // if true, allow replay gain path
	ForceReplay bool    // if true, apply replay gain even when caller didn't flag isReplay
	ReplayGain  float64 // default 1.1 (only used if UseReplay)
}

type AttnParams[T Numeric] struct {
	Wq []T // len=dk
	Wk []T // len=dk
	Wv []T // len=dk
	Wo []T // len=dk*Hcurr (iff UseWo)
	DK int
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

func defReplayGain[T Numeric](cfg *AttnConfig[T]) float64 {
	if cfg == nil || cfg.ReplayGain == 0 {
		return 1.1
	}
	return cfg.ReplayGain
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

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i]
	}
	return s
}

/* ─────────────────────────────────────────────────────────────────────────────
   Param placement
   ───────────────────────────────────────────────────────────────────────────── */

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

/* ─────────────────────────────────────────────────────────────────────────────
   Forward
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
	dk := params.DK
	if N == 0 || dk == 0 {
		return
	}

	// 1) Tokens
	X := make([]float64, N)
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

	// 2) Q,K,V
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

	// 3) Attention
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

	// 4) Z = A V
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

	// 5) Pool
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

	// Optional norm
	if cfg.UseNorm {
		eps := defNormEps(cfg)
		mu := 0.0
		for t := 0; t < dk; t++ {
			mu += zbar[t]
		}
		mu /= float64(dk)
		vv := 0.0
		for t := 0; t < dk; t++ {
			d := zbar[t] - mu
			vv += d * d
		}
		vv /= float64(dk)
		invStd := 1.0 / math.Sqrt(vv+eps)
		for t := 0; t < dk; t++ {
			zbar[t] = (zbar[t] - mu) * invStd
		}
	}

	// 6) Project
	out := make([]float64, curr.Height)
	if cfg.UseWo && len(params.Wo) == dk*curr.Height {
		for y := 0; y < curr.Height; y++ {
			sum := 0.0
			base := y * dk
			for t := 0; t < dk; t++ {
				sum += zbar[t] * f64(params.Wo[base+t])
			}
			out[y] = sum
		}
	} else {
		sum := 0.0
		for t := 0; t < dk; t++ {
			sum += zbar[t]
		}
		for y := 0; y < curr.Height; y++ {
			alpha := float64(y) / float64(max(1, curr.Height-1))
			out[y] = (1.0-alpha)*sum + 0.5*alpha*sum
		}
	}

	// 7) Activation (+ optional replay gain)
	useReplay := cfg.UseReplay && (isReplay || cfg.ForceReplay)
	replayGain := defReplayGain(cfg)
	for y := 0; y < curr.Height; y++ {
		val := tOf[T](out[y])
		val = ApplyActivationGeneric(val, curr.Neurons[y][col].Activation)
		if useReplay {
			val = tOf[T](float64(val) * replayGain)
		}
		curr.Neurons[y][col].Value = val
	}
}

/* ─────────────────────────────────────────────────────────────────────────────
   Backward
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
	dk := params.DK
	if N == 0 || dk == 0 {
		return
	}

	// upstream grad
	useReplay := cfg.UseReplay && (isReplay || cfg.ForceReplay)
	replayGain := defReplayGain(cfg)

	dy := make([]float64, curr.Height)
	for y := 0; y < curr.Height; y++ {
		v := float64(err[layerIdx][y][col])
		if useReplay {
			v *= replayGain
		}
		dy[y] = v
	}

	// Recompute intermediates (same as forward)
	X := make([]float64, N)
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

	Q := make([][]float64, N)
	K := make([][]float64, N)
	V := make([][]float64, N)
	for i := 0; i < N; i++ {
		q := make([]float64, dk)
		k := make([]float64, dk)
		v := make([]float64, dk)
		x := X[i]
		for t := 0; t < dk; t++ {
			q[t] = x * f64(params.Wq[t])
			k[t] = x * f64(params.Wk[t])
			v[t] = x * f64(params.Wv[t])
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

	// optional norm (cache stats)
	useNorm := cfg.UseNorm
	eps := defNormEps(cfg)

	var mu, invStd float64
	var yhat []float64
	if useNorm {
		mu = 0.0
		for t := 0; t < dk; t++ {
			mu += zbar[t]
		}
		mu /= float64(dk)
		vv := 0.0
		for t := 0; t < dk; t++ {
			d := zbar[t] - mu
			vv += d * d
		}
		vv /= float64(dk)
		invStd = 1.0 / math.Sqrt(vv+eps)
		yhat = make([]float64, dk)
		for t := 0; t < dk; t++ {
			yhat[t] = (zbar[t] - mu) * invStd
		}
	}

	// grads
	dZ := make([][]float64, N)
	for i := 0; i < N; i++ {
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

	gWq := make([]float64, dk)
	gWk := make([]float64, dk)
	gWv := make([]float64, dk)
	var gWo []float64
	if cfg.UseWo && len(params.Wo) == dk*curr.Height {
		gWo = make([]float64, dk*curr.Height)
	}

	// back from projection
	dZbar := make([]float64, dk)
	if cfg.UseWo && gWo != nil {
		for y := 0; y < curr.Height; y++ {
			base := y * dk
			gy := dy[y]
			for t := 0; t < dk; t++ {
				gWo[base+t] += zbar[t] * gy
				dZbar[t] += f64(params.Wo[base+t]) * gy
			}
		}
	} else {
		sumScale := 0.0
		den := float64(max(1, curr.Height-1))
		for y := 0; y < curr.Height; y++ {
			alpha := float64(y) / den
			sumScale += dy[y] * (1.0 - 0.5*alpha)
		}
		for t := 0; t < dk; t++ {
			dZbar[t] += sumScale
		}
	}

	// back through norm (if used)
	if useNorm {
		g := dZbar
		meanG := 0.0
		meanGY := 0.0
		for t := 0; t < dk; t++ {
			meanG += g[t]
			meanGY += g[t] * yhat[t]
		}
		meanG /= float64(dk)
		meanGY /= float64(dk)
		for t := 0; t < dk; t++ {
			dZbar[t] = (g[t] - meanG - yhat[t]*meanGY) * invStd
		}
	}

	// zbar = mean_i Z[i]
	for i := 0; i < N; i++ {
		for t := 0; t < dk; t++ {
			dZ[i][t] += dZbar[t] * invN
		}
	}

	// Z = A V
	for i := 0; i < N; i++ {
		zi := dZ[i]
		Ai := A[i]
		for j := 0; j < N; j++ {
			aij := Ai[j]
			if aij == 0 {
				continue
			}
			for t := 0; t < dk; t++ {
				dV[j][t] += aij * zi[t]
			}
			dA[i][j] += dot(zi, V[j])
		}
	}

	// A = softmax(S) row-wise
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

	// S = (Q Kᵀ)/√dk
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			g := dS[i][j] * (1.0 / math.Sqrt(float64(dk)))
			if g == 0 {
				continue
			}
			for t := 0; t < dk; t++ {
				dQ[i][t] += g * K[j][t]
				dK[j][t] += g * Q[i][t]
			}
		}
	}

	// back to params & X
	for i := 0; i < N; i++ {
		x := X[i]
		for t := 0; t < dk; t++ {
			gWq[t] += x * dQ[i][t]
			gWk[t] += x * dK[i][t]
			gWv[t] += x * dV[i][t]
			dX[i] += dQ[i][t]*f64(params.Wq[t]) +
				dK[i][t]*f64(params.Wk[t]) +
				dV[i][t]*f64(params.Wv[t])
		}
	}

	// clip + SGD
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

	// scatter to prev err
	iTok := 0
	for y := 0; y < prev.Height; y++ {
		for x := 0; x < prev.Width; x++ {
			err[layerIdx-1][y][x] += T(dX[iTok])
			iTok++
		}
	}
}
