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
