// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build !amd64 && !(arm64 && go1.27)

package boxfilter

// No SIMD kernel on this architecture (arm64 before Go 1.27 cannot assemble the
// integer NEON multiplies): the scalar reference does all the work.

func accumPremul(acc []uint32, src []byte, n int, w uint32) (done int) { return 0 }

func sumBlocks(dst []uint32, src []byte, stride, n, m, groups int) (done int) { return 0 }

func sumGroups(dst, src []uint32, groups, n int) (done int) { return 0 }
