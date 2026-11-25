// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build arm64 && go1.27

package boxfilter

// accumPremulNEON adds w times the premultiplied pixels of src to acc, sixteen
// pixels per unit: len(src) is a multiple of 16, len(acc) >= len(src), and
// w < 2^16.
//
//go:noescape
func accumPremulNEON(acc []uint32, src []byte, w uint64)

// sumGroupsNEON sums n >= 1 source pixels per destination pixel for len(dst)/4
// destinations; src holds at least len(dst)/4*n pixels.
//
//go:noescape
func sumGroupsNEON(dst, src []uint32, n int)

// sumBlocksNEON writes len(dst)/4 destination pixels, each the sum of an n x m
// block of premultiplied pixels (n, m >= 1), the first block starting at src[0]
// and the rows stride bytes apart; it reads no byte outside the blocks.
//
//go:noescape
func sumBlocksNEON(dst []uint32, src []byte, stride, n, m int)

func sumBlocks(dst []uint32, src []byte, stride, n, m, groups int) (done int) {
	if groups == 0 {
		return 0
	}
	sumBlocksNEON(dst[:4*groups], src, stride, n, m)
	return groups
}

func accumPremul(acc []uint32, src []byte, n int, w uint32) (done int) {
	units := n / 4
	if units == 0 || w > 0xFFFF {
		return 0
	}
	accumPremulNEON(acc[:units*16], src[:units*16], uint64(w))
	return units * 4
}

func sumGroups(dst, src []uint32, groups, n int) (done int) {
	sumGroupsNEON(dst[:4*groups], src[:4*groups*n], n)
	return groups
}
