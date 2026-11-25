// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

package boxfilter

// accumPremulRef is the scalar definition of AccumPremul over whole pixels:
// len(acc) == len(src) and both are a multiple of 4.
func accumPremulRef(acc []uint32, src []byte, w uint32) {
	for i := 0; i+4 <= len(src); i += 4 {
		a := uint32(src[i+3])
		acc[i] += w * uint32(src[i]) * a
		acc[i+1] += w * uint32(src[i+1]) * a
		acc[i+2] += w * uint32(src[i+2]) * a
		acc[i+3] += w * a
	}
}

// sumGroupsRef is the scalar definition of SumGroups: len(dst) is 4 per group
// and len(src) is 4*n per group.
func sumGroupsRef(dst, src []uint32, n int) {
	for g := 0; g+4 <= len(dst); g += 4 {
		var r, gr, b, a uint32
		for j := 0; j < n; j++ {
			o := (g/4*n + j) * 4
			r += src[o]
			gr += src[o+1]
			b += src[o+2]
			a += src[o+3]
		}
		dst[g], dst[g+1], dst[g+2], dst[g+3] = r, gr, b, a
	}
}
