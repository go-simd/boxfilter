// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

// Package boxfilter holds the two inner loops of an area-averaging (box)
// downscale of an RGBA8 picture on PREMULTIPLIED alpha, in exact integer
// arithmetic, with SIMD kernels (NEON on arm64, AVX2 on amd64) and a portable
// scalar fallback that every kernel is bit-identical to.
//
// A box downscale is two separable passes. The vertical pass adds the rows a
// destination row covers into a row of uint32 sums, each source pixel
// premultiplied on the way in: that is AccumPremul, and it is where nearly all
// the time goes (one multiply per colour channel per source pixel). The
// horizontal pass then sums runs of those accumulated pixels, which for an
// integer ratio is a plain sum of n neighbours: SumGroups.
//
// The arithmetic wraps modulo 2^32 like any uint32 sum. The CALLER bounds it: a
// premultiplied channel is at most 255*255 = 65025, so the sums stay exact while
// 65025 times the total weight stays below 2^32, i.e. a total weight of at most
// 66051.
package boxfilter

// AccumPremul adds w times the premultiplied form of every pixel of src (RGBA,
// 8 bits per channel, straight alpha) to acc, four uint32 per pixel in the same
// R, G, B, A order:
//
//	acc[4i+0] += w * (R*A)   acc[4i+1] += w * (G*A)
//	acc[4i+2] += w * (B*A)   acc[4i+3] += w * A
//
// It handles min(len(src), len(acc))/4 pixels and ignores the rest.
func AccumPremul(acc []uint32, src []byte, w uint32) {
	n := min(len(src)/4, len(acc)/4)
	done := accumPremul(acc, src, n, w)
	accumPremulRef(acc[4*done:4*n], src[4*done:4*n], w)
}

// SumBlocks is the whole of a box downscale by an INTEGER factor in one pass over
// the source: destination pixel j (four uint32, R, G, B, A order) is the sum of the
// premultiplied pixels of the n x m block whose top-left pixel is pixel j*n of row
// 0 of src, rows being stride bytes apart:
//
//	dst[4j+0] = sum over the block of R*A   dst[4j+1] = sum of G*A
//	dst[4j+2] = sum over the block of B*A   dst[4j+3] = sum of A
//
// It is AccumPremul over m rows (every weight 1) followed by SumGroups with the
// same n, but the sums stay in vector registers for the whole block and only the
// result is stored: nothing is written to, or read back from, an accumulator row.
// A kernel reads no byte outside the blocks.
//
// It fills as many whole destination pixels as dst holds, as each of the m rows
// holds (stride/(4n) per row) and as src covers (the last row included); n < 1,
// m < 1 or stride < 4n does nothing. Like the other functions the sums wrap modulo
// 2^32: exact while 65025*n*m < 2^32.
func SumBlocks(dst []uint32, src []byte, stride, n, m int) {
	if n < 1 || m < 1 || stride < 4*n || (m-1) > len(src)/stride {
		return
	}
	groups := min(len(dst)/4, stride/(4*n), (len(src)-(m-1)*stride)/(4*n))
	done := sumBlocks(dst, src, stride, n, m, groups)
	sumBlocksRef(dst[4*done:4*groups], src[4*done*n:], stride, n, m)
}

// SumGroups sums runs of n pixels: dst pixel i (four uint32) is the channel-wise
// sum of src pixels i*n .. i*n+n-1. It fills as many whole destination pixels as
// both slices hold, and does nothing for n < 1.
func SumGroups(dst, src []uint32, n int) {
	if n < 1 {
		return
	}
	groups := min(len(dst)/4, len(src)/4/n)
	done := sumGroups(dst, src, groups, n)
	sumGroupsRef(dst[4*done:4*groups], src[4*done*n:4*groups*n], n)
}
