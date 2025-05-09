// diffusion.go
package paragon

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// DiffusionConfig defines parameters for the diffusion process
type DiffusionConfig struct {
	NumTimesteps int     // Number of diffusion steps, e.g. 50
	MaxLength    int     // Maximum sequence length, e.g. 64
	LearningRate float64 // Learning rate for training
	Epochs       int     // Number of training epochs
	Temperature  float64 // Temperature for sampling
	TopK         int     // Top-k sampling parameter

	// Below are new fields for improved discrete diffusion
	MaskScheduleStart float64 // fraction of tokens to mask at t=0
	MaskScheduleEnd   float64 // fraction of tokens to mask at t=NumTimesteps-1
}

// DiffusionModel encapsulates a network with diffusion capabilities
type DiffusionModel struct {
	Network       *Network
	Config        DiffusionConfig
	Tokenizer     *CustomTokenizer
	SpecialTokens map[int]bool

	// A per-step fraction of tokens to mask. E.g. fraction[0] = 0.1, fraction[1] = 0.15, ...
	// We'll fill this in once on model creation.
	MaskFraction []float64
}

// CustomTokenizer (moved here for modularity)
type CustomTokenizer struct {
	Vocab         map[string]int
	ReverseVocab  map[int]string
	VocabSize     int
	SpecialTokens map[int]bool
}

func NewCustomTokenizer(sentences []string) *CustomTokenizer {
	t := &CustomTokenizer{
		Vocab:         make(map[string]int),
		ReverseVocab:  make(map[int]string),
		SpecialTokens: map[int]bool{},
	}
	specials := []string{"[PAD]", "[MASK]", "[CLS]", "[SEP]"}
	for i, tok := range specials {
		t.Vocab[tok] = i
		t.ReverseVocab[i] = tok
		t.SpecialTokens[i] = true
	}
	nextID := len(specials)
	for _, s := range sentences {
		words := strings.Fields(strings.ToLower(s))
		for _, w := range words {
			if _, exists := t.Vocab[w]; !exists {
				t.Vocab[w] = nextID
				t.ReverseVocab[nextID] = w
				nextID++
			}
		}
	}
	t.VocabSize = nextID
	return t
}

func (t *CustomTokenizer) Encode(text string) []int {
	words := strings.Fields(strings.ToLower(text))
	ids := make([]int, len(words))
	for i, w := range words {
		if id, exists := t.Vocab[w]; exists {
			ids[i] = id
		} else {
			ids[i] = t.Vocab["[PAD]"]
		}
	}
	return ids
}

func (t *CustomTokenizer) Decode(ids []int) string {
	words := make([]string, 0, len(ids))
	for _, id := range ids {
		if word, exists := t.ReverseVocab[id]; exists && !t.SpecialTokens[id] {
			words = append(words, word)
		}
	}
	return strings.Join(words, " ")
}

// NewDiffusionModel initializes a diffusion model with a network & improved mask schedule
func NewDiffusionModel(network *Network, config DiffusionConfig, sentences []string) *DiffusionModel {
	tokenizer := NewCustomTokenizer(sentences)

	// Create the model
	d := &DiffusionModel{
		Network:       network,
		Config:        config,
		Tokenizer:     tokenizer,
		SpecialTokens: tokenizer.SpecialTokens,
		MaskFraction:  make([]float64, config.NumTimesteps),
	}

	// Fill in the mask schedule from start to end (linear, or tweak if you like)
	// For example, if MaskScheduleStart=0.2, MaskScheduleEnd=0.8, then over timesteps we go linearly from 0.2 -> 0.8
	for t := 0; t < config.NumTimesteps; t++ {
		frac := config.MaskScheduleStart + (config.MaskScheduleEnd-config.MaskScheduleStart)*float64(t)/float64(config.NumTimesteps-1)
		if frac < 0 {
			frac = 0
		}
		if frac > 1 {
			frac = 1
		}
		d.MaskFraction[t] = frac
	}

	return d
}

func (d *DiffusionModel) AddNoise(tokens []int, t int) []int {
	noiseLevel := math.Min(0.8, float64(t+1)/float64(d.Config.NumTimesteps))
	noisyTokens := make([]int, d.Config.MaxLength)
	padTokenID := d.Tokenizer.Vocab["[PAD]"]
	maskTokenID := d.Tokenizer.Vocab["[MASK]"]
	if len(tokens) > d.Config.MaxLength {
		copy(noisyTokens, tokens[:d.Config.MaxLength])
	} else {
		copy(noisyTokens, tokens)
		for i := len(tokens); i < d.Config.MaxLength; i++ {
			noisyTokens[i] = padTokenID
		}
	}
	for i := 0; i < d.Config.MaxLength; i++ {
		if rand.Float64() < noiseLevel && noisyTokens[i] != padTokenID {
			noisyTokens[i] = maskTokenID
		}
	}
	return noisyTokens
}

func (d *DiffusionModel) AddNoiseMasked(tokens []int, tVal float64) []int {
	noiseLevel := tVal // No capping at 0.8, full range [0,1]
	noisyTokens := make([]int, d.Config.MaxLength)
	padTokenID := d.Tokenizer.Vocab["[PAD]"]
	maskTokenID := d.Tokenizer.Vocab["[MASK]"]
	if len(tokens) > d.Config.MaxLength {
		copy(noisyTokens, tokens[:d.Config.MaxLength])
	} else {
		copy(noisyTokens, tokens)
		for i := len(tokens); i < d.Config.MaxLength; i++ {
			noisyTokens[i] = padTokenID
		}
	}
	for i := 0; i < d.Config.MaxLength; i++ {
		if noisyTokens[i] != padTokenID && rand.Float64() < noiseLevel {
			noisyTokens[i] = maskTokenID
		}
	}
	return noisyTokens
}

// diffusion.go (partial update)
func (d *DiffusionModel) Train(sentences []string) {
	data := make([][]int, len(sentences))
	for i, s := range sentences {
		ids := d.Tokenizer.Encode(s)
		if len(ids) > d.Config.MaxLength {
			data[i] = ids[:d.Config.MaxLength]
		} else {
			data[i] = make([]int, d.Config.MaxLength)
			copy(data[i], ids)
			for j := len(ids); j < d.Config.MaxLength; j++ {
				data[i][j] = d.Tokenizer.Vocab["[PAD]"]
			}
		}
	}

	for epoch := 0; epoch < d.Config.Epochs; epoch++ {
		totalLoss := 0.0
		lr := d.Config.LearningRate * (1 - float64(epoch)/float64(d.Config.Epochs)) // Decay
		for _, tokens := range data {
			t := rand.Intn(d.Config.NumTimesteps)
			noisyTokens := d.AddNoise(tokens, t)
			input := make([][]float64, 1)
			input[0] = make([]float64, d.Config.MaxLength)
			for i, tok := range noisyTokens {
				input[0][i] = float64(tok)
			}
			output := d.Network.ForwardTransformer(input)

			loss := 0.0
			for i := 0; i < d.Config.MaxLength; i++ {
				probs := Softmax(output[i])
				target := tokens[i]
				loss -= math.Log(math.Max(probs[target], 1e-10))
			}
			totalLoss += loss / float64(d.Config.MaxLength)

			errorTerms := make([][]float64, d.Config.MaxLength)
			for i := 0; i < d.Config.MaxLength; i++ {
				probs := Softmax(output[i])
				errorTerms[i] = make([]float64, len(probs))
				for j := 0; j < len(probs); j++ {
					delta := probs[j]
					if j == tokens[i] {
						delta -= 1
					}
					if delta > 5.0 {
						delta = 5.0 // Gradient clipping
					} else if delta < -5.0 {
						delta = -5.0
					}
					errorTerms[i][j] = delta
				}
			}
			d.Network.Backward(errorTerms, lr)
		}
		if epoch%40 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss/float64(len(sentences)))
		}
	}
}

// diffusion.go (only Generate function updated)
func (d *DiffusionModel) Generate() string {
	// current[] holds discrete token IDs, length=MaxLength
	current := make([]int, d.Config.MaxLength)

	// Random init of tokens
	for i := range current {
		current[i] = rand.Intn(d.Tokenizer.VocabSize)
	}
	fmt.Println("Initial random tokens:", d.Tokenizer.Decode(current))

	// Step-by-step from t=NumTimesteps-1 down to t=0
	for t := d.Config.NumTimesteps - 1; t >= 0; t-- {
		// 1) Build one-hot input [MaxLength][VocabSize]
		oneHot2D := make([][]float64, d.Config.MaxLength)
		for i, tok := range current {
			row := make([]float64, d.Tokenizer.VocabSize)
			if tok >= 0 && tok < d.Tokenizer.VocabSize {
				row[tok] = 1.0
			}
			oneHot2D[i] = row
		}

		// 2) Forward pass
		output2D := d.Network.ForwardTransformer(oneHot2D) // shape [1][MaxLength*VocabSize]
		logits := output2D[0]                              // length = 10*VocabSize

		// 3) For each position i in [0..MaxLength-1], sample a new token
		for i := 0; i < d.Config.MaxLength; i++ {
			start := i * d.Tokenizer.VocabSize
			end := start + d.Tokenizer.VocabSize
			probs := Softmax(logits[start:end])

			// Example: top-k or random sampling
			// Let’s do a simple max-prob pick for clarity:
			bestToken := 0
			bestProb := probs[0]
			for m := 1; m < len(probs); m++ {
				if probs[m] > bestProb {
					bestProb = probs[m]
					bestToken = m
				}
			}
			current[i] = bestToken
		}

		// Optionally, you can do “re-masking” or other diffusion steps.
		// This snippet just does a single pass each step for demonstration.
	}

	// Decode final tokens
	finalStr := d.Tokenizer.Decode(current)
	return finalStr
}

func trainMaskedDiffusion(model *DiffusionModel, sentences []string, tokenizer *CustomTokenizer,
	dConfig DiffusionConfig, tConfig TransformerConfig) {

	batchSize := 10
	multithreading := true
	cpuPercent := 0.8
	numCores := runtime.NumCPU()
	numThreads := int(float64(numCores) * cpuPercent)
	if numThreads < 1 {
		numThreads = 1
	}
	numBatches := (len(sentences) + batchSize - 1) / batchSize
	fmt.Printf("Using %d threads (%d%% of %d cores), %d batches\n", numThreads, int(cpuPercent*100), numCores, numBatches)

	for epoch := 0; epoch < dConfig.Epochs; epoch++ {
		startTime := time.Now()
		// Learning rate schedule as before:
		lr := dConfig.LearningRate * (1 + math.Cos(float64(epoch)*math.Pi/float64(dConfig.Epochs))) / 2

		// Prepare data (tokenize and pad)
		data := make([][]int, len(sentences))
		for i, s := range sentences {
			ids := tokenizer.Encode(s)
			if len(ids) > dConfig.MaxLength {
				data[i] = ids[:dConfig.MaxLength]
			} else {
				data[i] = make([]int, dConfig.MaxLength)
				copy(data[i], ids)
				for j := len(ids); j < dConfig.MaxLength; j++ {
					data[i][j] = tokenizer.Vocab["[PAD]"]
				}
			}
		}

		// Prepare for multithreaded batch processing
		var wg sync.WaitGroup
		sem := make(chan struct{}, numThreads)
		totalLoss := 0.0
		accumulatedErrorTerms := make([][]float64, len(sentences))
		for i := range accumulatedErrorTerms {
			accumulatedErrorTerms[i] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
		}
		lossChan := make(chan float64, numBatches)
		errorTermsChan := make(chan struct {
			terms    [][]float64
			batchIdx int
		}, numBatches)

		for i := 0; i < len(sentences); i += batchSize {
			end := i + batchSize
			if end > len(sentences) {
				end = len(sentences)
			}
			batchData := data[i:end]
			batchIdx := i / batchSize

			if multithreading {
				wg.Add(1)
				sem <- struct{}{}
				go func(startIdx int, batch [][]int, idx int) {
					defer wg.Done()
					defer func() { <-sem }()

					// For each sample in the batch, sample a continuous t in [0,1]
					batchInputs := make([][]float64, len(batch))
					batchTargets := make([][]int, len(batch))
					noisyBatch := make([][]int, len(batch))
					for j, tokens := range batch {
						tVal := rand.Float64()
						noisyTokens := model.AddNoiseMasked(tokens, tVal)
						noisyBatch[j] = noisyTokens
						batchInputs[j] = make([]float64, dConfig.MaxLength)
						for k, tok := range noisyTokens {
							batchInputs[j][k] = float64(tok)
						}
						batchTargets[j] = tokens
					}

					// Forward pass through the transformer
					batchOutputs := make([][][]float64, len(batch))
					for j, input := range batchInputs {
						singleInput := [][]float64{input}
						batchOutputs[j] = model.Network.ForwardTransformer(singleInput)
					}

					// Compute loss only on positions where the noisy input is [MASK]
					loss := 0.0
					batchErrorTerms := make([][]float64, len(batch))
					for j := 0; j < len(batch); j++ {
						batchErrorTerms[j] = make([]float64, dConfig.MaxLength*tConfig.VocabSize)
						for k := 0; k < dConfig.MaxLength; k++ {
							if noisyBatch[j][k] == tokenizer.Vocab["[MASK]"] {
								startIdx := k * tConfig.VocabSize
								endIdx := (k + 1) * tConfig.VocabSize
								probs := Softmax(batchOutputs[j][0][startIdx:endIdx])
								target := batchTargets[j][k]
								loss -= math.Log(math.Max(probs[target], 1e-10))
								// Compute error term (with gradient clipping)
								for m := 0; m < tConfig.VocabSize; m++ {
									delta := probs[m]
									if m == target {
										delta -= 1
									}
									if delta > 5.0 {
										delta = 5.0
									} else if delta < -5.0 {
										delta = -5.0
									}
									batchErrorTerms[j][startIdx+m] = delta
								}
							} else {
								// For nonmasked tokens, set error term to zero.
								for m := 0; m < tConfig.VocabSize; m++ {
									batchErrorTerms[j][k*tConfig.VocabSize+m] = 0
								}
							}
						}
					}
					lossChan <- loss / float64(len(batch))
					errorTermsChan <- struct {
						terms    [][]float64
						batchIdx int
					}{batchErrorTerms, idx}
				}(i, batchData, batchIdx)
			} else {
				// (Non–multithreaded branch omitted for brevity)
			}
		}
		go func() {
			wg.Wait()
			close(lossChan)
			close(errorTermsChan)
		}()
		for l := range lossChan {
			totalLoss += l
		}
		for et := range errorTermsChan {
			start := et.batchIdx * batchSize
			for j, terms := range et.terms {
				if start+j < len(accumulatedErrorTerms) {
					accumulatedErrorTerms[start+j] = terms
				}
			}
		}
		model.Network.Backward(accumulatedErrorTerms, lr)
		totalLoss /= float64(numBatches)
		if epoch%10 == 0 {
			fmt.Printf("%s Epoch %d, Loss: %.4f, Time: %v\n", time.Now().String(), epoch, totalLoss, time.Since(startTime))
			fmt.Println("Generating sample text...")
			sample := model.GenerateMasked()
			fmt.Println("Sample generation:", sample)
		}
	}
}

func (d *DiffusionModel) GenerateMasked() string {
	maskTokenID := d.Tokenizer.Vocab["[MASK]"]
	//fmt.Println("maskTokenID:", maskTokenID) // Debug: Should be 4
	current := make([]int, d.Config.MaxLength)
	for i := range current {
		current[i] = maskTokenID
	}
	//fmt.Println("Initial fully masked sequence:", d.Tokenizer.Decode(current))
	steps := d.Config.NumTimesteps
	for s := steps; s > 0; s-- {
		input := make([][]float64, d.Config.MaxLength)
		for k := 0; k < d.Config.MaxLength; k++ {
			input[k] = make([]float64, d.Tokenizer.VocabSize)
			tok := current[k]
			if tok >= 0 && tok < d.Tokenizer.VocabSize {
				input[k][tok] = 1.0
			}
		}
		outputFlat := d.Network.ForwardTransformer(input)
		output := make([][]float64, d.Config.MaxLength)
		for i := 0; i < d.Config.MaxLength; i++ {
			start := i * d.Tokenizer.VocabSize
			end := (i + 1) * d.Tokenizer.VocabSize
			output[i] = outputFlat[0][start:end]
		}
		maskedPositions := []int{}
		for i := 0; i < d.Config.MaxLength; i++ {
			if current[i] == maskTokenID {
				maskedPositions = append(maskedPositions, i)
				probs := Softmax(output[i])
				//fmt.Printf("Step %d, Pos %d, Probs: %v\n", s, i, probs)
				topK := make([]struct {
					idx  int
					prob float64
				}, d.Tokenizer.VocabSize)
				for j := 0; j < d.Tokenizer.VocabSize; j++ {
					prob := probs[j] / d.Config.Temperature
					if j == maskTokenID { // Exclude [MASK] from sampling
						prob = 0
					}
					topK[j] = struct {
						idx  int
						prob float64
					}{j, prob}
				}
				sort.Slice(topK, func(i, j int) bool { return topK[i].prob > topK[j].prob })
				topK = topK[:d.Config.TopK]
				sum := 0.0
				for _, p := range topK {
					sum += p.prob
				}
				if sum == 0 { // Fallback if all probs are 0
					sum = 1.0
					for j := range topK {
						topK[j].prob = 1.0 / float64(d.Config.TopK)
					}
				} else {
					for j := range topK {
						topK[j].prob /= sum
					}
				}
				r := rand.Float64()
				cumulative := 0.0
				selected := topK[0].idx // Default to highest prob if loop fails
				for _, tk := range topK {
					cumulative += tk.prob
					if r <= cumulative {
						selected = tk.idx
						break
					}
				}
				current[i] = selected
				//fmt.Printf("Step %d, Pos %d, Selected: %d\n", s, i, current[i])
			}
		}
		if s > 1 {
			pRemask := float64(s-1) / float64(s)
			for _, i := range maskedPositions {
				if rand.Float64() < pRemask {
					current[i] = maskTokenID
				}
			}
		}
		//fmt.Printf("Generation step %d: %s\n", s, d.Tokenizer.Decode(current))
	}
	// Replace remaining [MASK] with 0
	for i := range current {
		if current[i] == maskTokenID {
			current[i] = 0
		}
	}
	//fmt.Println("Final current before decode:", current)
	return d.Tokenizer.Decode(current)
}

// BetterAddNoise masks a fixed fraction of tokens (according to MaskFraction[t]) instead of a random fraction
func (d *DiffusionModel) BetterAddNoise(x0 []int, t int) []int {
	noisy := make([]int, len(x0))
	copy(noisy, x0)

	padID := d.Tokenizer.Vocab["[PAD]"]
	maskID := d.Tokenizer.Vocab["[MASK]"]
	fraction := d.MaskFraction[t] // fraction of non-pad tokens to mask
	if fraction <= 0 {
		return noisy
	}
	// Collect indices of non-pad tokens
	var idxes []int
	for i, tok := range x0 {
		if tok != padID {
			idxes = append(idxes, i)
		}
	}
	// Shuffle them
	rand.Shuffle(len(idxes), func(i, j int) {
		idxes[i], idxes[j] = idxes[j], idxes[i]
	})
	// Number of tokens to mask
	k := int(math.Round(float64(len(idxes)) * fraction))
	for i := 0; i < k; i++ {
		noisy[idxes[i]] = maskID
	}
	return noisy
}

// TrainBetterDiffusion uses the improved discrete diffusion approach
// 1) We pick a random step t in [0, NumTimesteps-1]
// 2) We apply BetterAddNoise(...) to get x_t
// 3) We feed x_t into the network, compute cross-entropy w.r.t. x0 only on masked positions
// 4) Single-step update
func (d *DiffusionModel) TrainBetterDiffusion(samples [][]int) {
	data := make([][]int, len(samples))
	copy(data, samples)

	for epoch := 0; epoch < d.Config.Epochs; epoch++ {
		totalLoss := 0.0
		lr := d.Config.LearningRate * (1.0 - float64(epoch)/float64(d.Config.Epochs))

		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, x0 := range data {
			t := rand.Intn(d.Config.NumTimesteps)
			xt := d.BetterAddNoise(x0, t)

			// BUILD A [MaxLength][VocabSize] array
			batchInput := make([][]float64, d.Config.MaxLength)
			for i, tok := range xt {
				row := make([]float64, d.Tokenizer.VocabSize)
				if tok >= 0 && tok < d.Tokenizer.VocabSize {
					row[tok] = 1.0
				}
				batchInput[i] = row
			}

			// forward
			output2D := d.Network.ForwardTransformer(batchInput) // shape: [1][MaxLength * VocabSize] in paragon
			preds := output2D[0]                                 // length = MaxLength * VocabSize

			var loss float64
			errorTerms := make([]float64, d.Config.MaxLength*d.Tokenizer.VocabSize)

			maskID := d.Tokenizer.Vocab["[MASK]"]
			for i, tok := range xt {
				if tok == maskID {
					start := i * d.Tokenizer.VocabSize
					end := start + d.Tokenizer.VocabSize
					probs := Softmax(preds[start:end])
					target := x0[i]
					loss -= math.Log(math.Max(probs[target], 1e-10))
					for m := 0; m < d.Tokenizer.VocabSize; m++ {
						delta := probs[m]
						if m == target {
							delta -= 1.0
						}
						// clip
						if delta > 5.0 {
							delta = 5.0
						} else if delta < -5.0 {
							delta = -5.0
						}
						errorTerms[start+m] = delta
					}
				}
			}

			totalLoss += loss

			// reshape error terms to [MaxLength][VocabSize]
			shaped := make([][]float64, d.Config.MaxLength)
			for i := 0; i < d.Config.MaxLength; i++ {
				st := i * d.Tokenizer.VocabSize
				shaped[i] = errorTerms[st : st+d.Tokenizer.VocabSize]
			}
			d.Network.Backward(shaped, lr)
		}

		avgLoss := totalLoss / float64(len(data))
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, avgLoss)
		}
	}
}

// GenerateBetter does a *one-pass* reverse diffusion without re-masking. We start from t=NumTimesteps-1
// and reduce to t=0, but we do *not* forcibly mask anything again. Instead, we let the model refine
// step by step, each time denoising from x_t -> x_{t-1} as best it can.
func (d *DiffusionModel) GenerateBetter() []int {
	maskID := d.Tokenizer.Vocab["[MASK]"]
	xcur := make([]int, d.Config.MaxLength)
	for i := range xcur {
		xcur[i] = maskID
	}

	for t := d.Config.NumTimesteps - 1; t >= 0; t-- {
		batchInput := make([][]float64, d.Config.MaxLength)
		for i, tok := range xcur {
			row := make([]float64, d.Tokenizer.VocabSize)
			if tok >= 0 && tok < d.Tokenizer.VocabSize {
				row[tok] = 1.0
			}
			batchInput[i] = row
		}

		output2D := d.Network.ForwardTransformer(batchInput)
		preds := output2D[0]

		maskedPositions := []int{}
		for i, tok := range xcur {
			if tok == maskID {
				maskedPositions = append(maskedPositions, i)
				start := i * d.Tokenizer.VocabSize
				end := start + d.Tokenizer.VocabSize
				probs := Softmax(preds[start:end])
				if d.Config.Temperature > 1e-12 {
					for j := range probs {
						probs[j] /= d.Config.Temperature
					}
				}
				probs[maskID] = 0
				sum := 0.0
				for _, p := range probs {
					sum += p
				}
				if sum < 1e-12 {
					xcur[i] = 0
					continue
				}
				for j := range probs {
					probs[j] /= sum
				}
				topKSlice := make([]struct {
					idx  int
					prob float64
				}, len(probs))
				for j, p := range probs {
					topKSlice[j] = struct {
						idx  int
						prob float64
					}{j, p}
				}
				sort.Slice(topKSlice, func(a, b int) bool {
					return topKSlice[a].prob > topKSlice[b].prob
				})
				if d.Config.TopK < 1 {
					d.Config.TopK = 1
				}
				if d.Config.TopK > len(topKSlice) {
					d.Config.TopK = len(topKSlice)
				}
				topKSlice = topKSlice[:d.Config.TopK]
				r := rand.Float64()
				cumul := 0.0
				chosen := topKSlice[0].idx
				for _, pair := range topKSlice {
					cumul += pair.prob
					if r <= cumul {
						chosen = pair.idx
						break
					}
				}
				xcur[i] = chosen
			}
		}

		// Add re-masking if not the last step
		if t > 0 {
			pRemask := float64(t) / float64(d.Config.NumTimesteps)
			for _, i := range maskedPositions {
				if rand.Float64() < pRemask {
					xcur[i] = maskID
				}
			}
		}
	}

	// Replace any remaining [MASK] with 0
	for i, tok := range xcur {
		if tok == maskID {
			xcur[i] = 0
		}
	}
	return xcur
}
