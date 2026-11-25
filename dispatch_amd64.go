// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build amd64

package boxfilter

import "golang.org/x/sys/cpu"

// hasAVX2 gates the kernels; a variable so a test can drive the scalar fallback.
var hasAVX2 = cpu.X86.HasAVX2

// accumPremulAVX2 adds w times the premultiplied pixels of src to acc, four
// pixels per iteration: len(src) is a multiple of 16 and len(acc) >= len(src).
//
//go:noescape
func accumPremulAVX2(acc []uint32, src []byte, w uint64)

// sumGroupsAVX2 sums n >= 1 source pixels per destination pixel for len(dst)/4
// destinations; src holds at least len(dst)/4*n pixels.
//
//go:noescape
func sumGroupsAVX2(dst, src []uint32, n int)

func accumPremul(acc []uint32, src []byte, n int, w uint32) (done int) {
	units := n / 4
	if units == 0 || !hasAVX2 {
		return 0
	}
	accumPremulAVX2(acc[:units*16], src[:units*16], uint64(w))
	return units * 4
}

func sumGroups(dst, src []uint32, groups, n int) (done int) {
	if groups == 0 || !hasAVX2 {
		return 0
	}
	sumGroupsAVX2(dst[:4*groups], src[:4*groups*n], n)
	return groups
}
