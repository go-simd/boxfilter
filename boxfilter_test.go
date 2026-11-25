// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package boxfilter

import (
	"math/rand"
	"slices"
	"testing"
)

func randBytes(rng *rand.Rand, n int) []byte {
	b := make([]byte, n)
	rng.Read(b)
	return b
}

func randU32(rng *rand.Rand, n int) []uint32 {
	u := make([]uint32, n)
	for i := range u {
		u[i] = rng.Uint32()
	}
	return u
}

// premul is the definition, written from the spec in the doc comment with no
// slicing tricks, that every other implementation is held to.
func premul(acc []uint32, src []byte, w uint32, pixels int) []uint32 {
	out := slices.Clone(acc)
	for i := 0; i < pixels; i++ {
		r, g, b, a := uint32(src[4*i]), uint32(src[4*i+1]), uint32(src[4*i+2]), uint32(src[4*i+3])
		out[4*i] += w * r * a
		out[4*i+1] += w * g * a
		out[4*i+2] += w * b * a
		out[4*i+3] += w * a
	}
	return out
}

func TestAccumPremulMatchesTheDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, w := range []uint32{0, 1, 2, 3, 255, 65535, 65536, 1 << 31, 0xFFFFFFFF} {
		for n := 0; n <= 70; n++ {
			src := randBytes(rng, 4*n)
			acc := randU32(rng, 4*n)
			want := premul(acc, src, w, n)
			AccumPremul(acc, src, w)
			if !slices.Equal(acc, want) {
				t.Fatalf("w=%d n=%d: kernel differs from the definition", w, n)
			}
		}
	}
}

// The pixel count is the shorter of the two slices; the rest is untouched.
func TestAccumPremulLengthMismatch(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	src := randBytes(rng, 4*40+3) // 40 pixels and a stray 3 bytes
	acc := randU32(rng, 4*50+2)
	want := premul(acc, src, 7, 40)
	AccumPremul(acc, src, 7)
	if !slices.Equal(acc, want) {
		t.Fatal("longer acc or ragged src changed the result")
	}
	acc2 := randU32(rng, 4*20)
	want2 := premul(acc2, src, 7, 20)
	AccumPremul(acc2, src, 7)
	if !slices.Equal(acc2, want2) {
		t.Fatal("shorter acc was overrun or under-filled")
	}
}

func TestSumGroupsMatchesTheDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for n := 1; n <= 9; n++ {
		for groups := 0; groups <= 9; groups++ {
			src := randU32(rng, 4*n*groups)
			dst := randU32(rng, 4*groups)
			want := make([]uint32, 4*groups)
			for g := 0; g < groups; g++ {
				for j := 0; j < n; j++ {
					for c := 0; c < 4; c++ {
						want[4*g+c] += src[4*(g*n+j)+c]
					}
				}
			}
			SumGroups(dst, src, n)
			if !slices.Equal(dst, want) {
				t.Fatalf("n=%d groups=%d: differs", n, groups)
			}
		}
	}
}

func TestSumGroupsShortAndDegenerate(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	src := randU32(rng, 4*10)
	dst := randU32(rng, 4*7)
	keep := slices.Clone(dst)
	for _, n := range []int{0, -3} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("n=%d panicked: %v", n, r)
				}
			}()
			SumGroups(dst, src, n)
		}()
	}
	if !slices.Equal(dst, keep) {
		t.Fatal("n < 1 must do nothing")
	}
	SumGroups(dst, src, 3) // src holds 3 whole groups of 3 pixels, dst has room for 7
	for g := 0; g < 3; g++ {
		for c := 0; c < 4; c++ {
			want := src[4*(3*g)+c] + src[4*(3*g+1)+c] + src[4*(3*g+2)+c]
			if dst[4*g+c] != want {
				t.Fatalf("group %d channel %d: %d, want %d", g, c, dst[4*g+c], want)
			}
		}
	}
	if !slices.Equal(dst[12:], keep[12:]) {
		t.Fatal("destinations beyond the source were written")
	}
}

// FuzzAccumPremul: any bytes, weight and length agree with the definition.
func FuzzAccumPremul(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 255, 255, 255, 255}, uint32(1))
	f.Add(make([]byte, 64*3+8), uint32(65535))
	f.Fuzz(func(t *testing.T, src []byte, w uint32) {
		n := len(src) / 4
		acc := make([]uint32, 4*n+4)
		for i := range acc {
			acc[i] = uint32(i) * 2654435761
		}
		want := premul(acc, src, w, n)
		AccumPremul(acc, src, w)
		if !slices.Equal(acc, want) {
			t.Fatalf("w=%d n=%d differs", w, n)
		}
	})
}

// blocks is the definition of SumBlocks, written from the doc comment: one
// destination pixel per whole block, in a picture srcW pixels wide.
func blocks(src []byte, stride, n, m, groups int) []uint32 {
	want := make([]uint32, 4*groups)
	for g := 0; g < groups; g++ {
		for y := 0; y < m; y++ {
			for x := 0; x < n; x++ {
				o := y*stride + 4*(g*n+x)
				a := uint32(src[o+3])
				want[4*g] += uint32(src[o]) * a
				want[4*g+1] += uint32(src[o+1]) * a
				want[4*g+2] += uint32(src[o+2]) * a
				want[4*g+3] += a
			}
		}
	}
	return want
}

// Every block shape, tails of 1, 2 and 3 pixels included, a row wider than the
// blocks, and dst slots beyond the picture left alone.
func TestSumBlocksMatchesTheDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for n := 1; n <= 9; n++ {
		for m := 1; m <= 5; m++ {
			for groups := 0; groups <= 6; groups++ {
				for _, extra := range []int{0, 3} { // pixels past the last block in each row
					stride := 4 * (n*groups + extra)
					if stride < 4*n {
						stride = 4 * n
					}
					src := randBytes(rng, (m-1)*stride+4*n*groups)
					dst := randU32(rng, 4*groups+8)
					keep := slices.Clone(dst)
					SumBlocks(dst, src, stride, n, m)
					want := blocks(src, stride, n, m, min(groups, stride/(4*n)))
					got := min(groups, stride/(4*n))
					if !slices.Equal(dst[:4*got], want) || !slices.Equal(dst[4*got:], keep[4*got:]) {
						t.Fatalf("n=%d m=%d groups=%d extra=%d: differs from the definition", n, m, groups, extra)
					}
				}
			}
		}
	}
}

// The scalar reference is the oracle the kernels are held to: it must itself
// equal the definition, on every shape (it is also the whole implementation
// where no kernel exists).
func TestSumBlocksRefMatchesTheDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for n := 1; n <= 6; n++ {
		for m := 1; m <= 4; m++ {
			stride := 4 * (n*3 + 1)
			src := randBytes(rng, (m-1)*stride+4*n*3)
			dst := make([]uint32, 12)
			sumBlocksRef(dst, src, stride, n, m)
			if !slices.Equal(dst, blocks(src, stride, n, m, 3)) {
				t.Fatalf("n=%d m=%d: reference differs from the definition", n, m)
			}
		}
	}
}

// A block is read whole or not at all: the pixel count is the least of what dst
// holds, what a row holds and what the LAST row still holds.
func TestSumBlocksLimits(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	src := randBytes(rng, 4*(3*10)+4*7) // rows of 10 pixels, the last short of 3
	stride := 4 * 10
	dst := make([]uint32, 4*20)
	SumBlocks(dst, src, stride, 3, 4) // 10/3 = 3 per row, the last row holds 7 pixels: 2 whole blocks
	want := blocks(src, stride, 3, 4, 2)
	if !slices.Equal(dst[:8], want) || !slices.Equal(dst[8:], make([]uint32, 4*18)) {
		t.Fatal("the pixel count is not the least of dst, the row and the last row")
	}
	// A destination shorter than the picture.
	short := make([]uint32, 4*1+3)
	SumBlocks(short, src, stride, 3, 4)
	if !slices.Equal(short[:4], want[:4]) || short[4] != 0 {
		t.Fatal("a short dst was overrun or under-filled")
	}
}

func TestSumBlocksDegenerate(t *testing.T) {
	src := randBytes(rand.New(rand.NewSource(9)), 4*40)
	dst := make([]uint32, 4*8)
	for _, c := range [][3]int{{0, 1, 16}, {-1, 1, 16}, {2, 0, 16}, {2, -1, 16}, {4, 1, 12}, {2, 1000, 16}} {
		SumBlocks(dst, src, c[2], c[0], c[1]) // n, m, stride
		if !slices.Equal(dst, make([]uint32, 4*8)) {
			t.Fatalf("n=%d m=%d stride=%d wrote to dst", c[0], c[1], c[2])
		}
	}
	SumBlocks(nil, src, 16, 2, 1)
	SumBlocks(dst, nil, 16, 2, 1)
	if !slices.Equal(dst, make([]uint32, 4*8)) {
		t.Fatal("an empty src wrote to dst")
	}
}

// FuzzSumBlocks: any bytes and block shape agree with the definition, and no byte
// outside the blocks is read (the source is cut to exactly what the blocks cover).
func FuzzSumBlocks(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 255, 255, 255, 255, 9, 8, 7, 6}, uint8(3), uint8(1), uint8(0))
	f.Add(make([]byte, 400), uint8(5), uint8(4), uint8(2))
	f.Fuzz(func(t *testing.T, src []byte, n, m, extra uint8) {
		nn, mm := int(n%11)+1, int(m%6)+1
		stride := 4 * (nn*(len(src)/(4*nn*mm)) + int(extra%4))
		if stride < 4*nn {
			stride = 4 * nn
		}
		groups := stride / (4 * nn)
		if need := (mm-1)*stride + 4*nn*groups; len(src) > need {
			src = src[:need]
		}
		dst := make([]uint32, 4*groups)
		SumBlocks(dst, src, stride, nn, mm)
		avail := 0
		if len(src) >= (mm-1)*stride {
			avail = min(groups, (len(src)-(mm-1)*stride)/(4*nn))
		}
		want := blocks(src, stride, nn, mm, avail)
		if !slices.Equal(dst[:4*avail], want) {
			t.Fatalf("n=%d m=%d stride=%d len=%d differs", nn, mm, stride, len(src))
		}
	})
}

func BenchmarkSumBlocks(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	// 1000 wide, 4 x 4 blocks: one destination row of a 1000x700 to 250x175 downscale.
	const w, groups = 1000, 250
	for _, c := range [][2]int{{4, 4}, {2, 2}, {3, 3}, {8, 8}} {
		n, m := c[0], c[1]
		src := randBytes(rng, 4*w*m)
		dst := make([]uint32, 4*(w/n))
		name := itoa(n) + "x" + itoa(m)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(4 * w * m))
			for i := 0; i < b.N; i++ {
				SumBlocks(dst, src, 4*w, n, m)
			}
		})
		b.Run(name+"-twopass", func(b *testing.B) { // AccumPremul over m rows, then SumGroups
			acc, g := make([]uint32, 4*w), make([]uint32, 4*(w/n))
			b.SetBytes(int64(4 * w * m))
			for i := 0; i < b.N; i++ {
				clear(acc)
				for y := 0; y < m; y++ {
					AccumPremul(acc, src[4*w*y:4*w*(y+1)], 1)
				}
				SumGroups(g, acc, n)
			}
		})
	}
	_ = groups
}

func BenchmarkAccumPremul(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{24, 64, 256, 1000} {
		src, acc := randBytes(rng, 4*n), make([]uint32, 4*n)
		b.Run(itoa(n)+"px", func(b *testing.B) {
			b.SetBytes(int64(4 * n))
			for i := 0; i < b.N; i++ {
				AccumPremul(acc, src, 1)
			}
		})
		b.Run(itoa(n)+"px-ref", func(b *testing.B) {
			b.SetBytes(int64(4 * n))
			for i := 0; i < b.N; i++ {
				accumPremulRef(acc, src, 1)
			}
		})
	}
}

func BenchmarkSumGroups(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{2, 4, 8} {
		const groups = 250
		src, dst := randU32(rng, 4*n*groups), make([]uint32, 4*groups)
		b.Run("n"+itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SumGroups(dst, src, n)
			}
		})
		b.Run("n"+itoa(n)+"-ref", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sumGroupsRef(dst, src, n)
			}
		})
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for ; n > 0; n /= 10 {
		d = append([]byte{byte('0' + n%10)}, d...)
	}
	return string(d)
}

// The scalar references are held to the definition on their own, so they stay
// right on the architectures (and CPUs) where a kernel takes the work away.
func TestReferencesMatchTheDefinition(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for n := 0; n <= 20; n++ {
		src, acc := randBytes(rng, 4*n), randU32(rng, 4*n)
		want := premul(acc, src, 9, n)
		accumPremulRef(acc, src, 9)
		if !slices.Equal(acc, want) {
			t.Fatalf("accumPremulRef n=%d", n)
		}
	}
	for n := 1; n <= 5; n++ {
		src := randU32(rng, 4*n*3)
		dst := make([]uint32, 12)
		sumGroupsRef(dst, src, n)
		for g := 0; g < 3; g++ {
			for c := 0; c < 4; c++ {
				var want uint32
				for j := 0; j < n; j++ {
					want += src[4*(g*n+j)+c]
				}
				if dst[4*g+c] != want {
					t.Fatalf("sumGroupsRef n=%d g=%d c=%d", n, g, c)
				}
			}
		}
	}
}
