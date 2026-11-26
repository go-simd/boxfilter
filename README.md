# boxfilter

Pure-Go SIMD kernels for the two inner loops of an **area-averaging (box)
downscale of an RGBA8 picture on premultiplied alpha**, in exact integer
arithmetic.

```go
import "github.com/go-simd/boxfilter"

// vertical pass: acc += w * premultiply(row), four uint32 per pixel
boxfilter.AccumPremul(acc, srcRow, w)

// horizontal pass: dst pixel i = sum of src pixels i*n .. i*n+n-1
boxfilter.SumGroups(dst, acc, n)
```

* `AccumPremul(acc []uint32, src []byte, w uint32)`: for every pixel,
  `acc += w*(R*A, G*A, B*A, A)`.
* `SumGroups(dst, src []uint32, n int)`: channel-wise sum of runs of `n` pixels.

* `SumBlocks(dst []uint32, src []byte, stride, n, m int)`: the whole integer-factor
  downscale in one pass: destination pixel `j` is the sum of the premultiplied `n x m`
  block at pixel `j*n` of row 0, rows `stride` bytes apart. The sums stay in vector
  registers; no accumulator row is written or read back. A kernel reads no byte
  outside the blocks.

Sums wrap modulo 2^32; the caller keeps 65025 times the total weight below 2^32
(a total weight of at most 66051). Integer arithmetic is exact, so every kernel
is **bit-identical** to the scalar reference, checked by differential tests,
fuzzing and a 100% coverage gate.

## Architectures

| target | kernel |
| --- | --- |
| arm64, Go 1.27+ | NEON (`VTBL` premultiplier, `VUMULL`, `VUMLAL`) |
| amd64 | AVX2 (`VPSHUFB`, `VPMULLW`, `VPMULLD`), detected at run time |
| everything else | portable scalar |

On arm64 the weight must fit 16 bits for the kernel (larger weights take the
scalar path). The amd64 kernel was validated on CI runners, not on the
development machine.

## Performance

Apple M4 Max, `-cpu 1`, one row of N pixels, `AccumPremul(acc, src, 1)`:

| pixels | scalar | NEON | speed-up |
| ---: | ---: | ---: | ---: |
| 64 | 70.9 ns | 18.2 ns | 3.9x |
| 256 | 277.6 ns | 47.5 ns | 5.8x |
| 1000 | 1096 ns | 170.9 ns | 6.4x |

Against Pillow 13.0.0 (the reference tool), through `painter.DrawImage`, which
uses `SumBlocks` for whole-number ratios. Apple M4 Max, `-cpu 1`, 1-min load
4.9 to 5.2 (gate 8); Go time / Pillow time, below 1 is faster; full record in
[docs/perf-table.md](docs/perf-table.md), reproduced by `bench/` (hash-pinned
Python requirements):

| picture | Pillow call | ratio |
| --- | --- | ---: |
| 1000x700 to 250x175, translucent | resize(BOX) on RGBa | 0.49 |
| 256x256 to 24x24, translucent | resize(BOX) on RGBa | 0.66 |
| 64x64 to 16x16, translucent | resize(BOX) on RGBa | 0.38 |
| 1000x700 to 250x175, opaque | reduce(4) on RGBA | 0.45 |
| 1000x700 to 250x175, opaque | reduce(4) on RGBa | 0.97 |
| 1000x700 to 250x175, opaque | reduce(4) on RGBX | 0.97 |

Pillow's `reduce` on RGBA does not premultiply, so RGBa and RGBX are the
semantics-equal comparisons; the amd64 kernel is checked on CI runners only.

Used by [go-widgets/painter](https://github.com/go-widgets/painter) for its
downscaling `DrawImage`.
