// Copyright (c) 2026 the go-simd/boxfilter authors. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license that can be
// found in the LICENSE file at the root of this repository.

//go:build amd64

package boxfilter

import "testing"

// Without AVX2 the scalar reference does all the work and nothing is lost.
func TestWithoutAVX2TheReferenceRuns(t *testing.T) {
	was := hasAVX2
	defer func() { hasAVX2 = was }()
	hasAVX2 = false
	TestAccumPremulMatchesTheDefinition(t)
	TestSumGroupsMatchesTheDefinition(t)
	TestSumBlocksMatchesTheDefinition(t)
}

func TestAVX2KernelIsTheOneRunning(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU (or under this emulator): the kernel is not exercised here")
	}
	acc := make([]uint32, 16)
	if done := accumPremul(acc, make([]byte, 16), 4, 1); done != 4 {
		t.Fatalf("kernel handled %d pixels, want 4", done)
	}
	if done := sumBlocks(make([]uint32, 4), make([]byte, 32), 16, 2, 2, 1); done != 1 {
		t.Fatalf("block kernel handled %d groups, want 1", done)
	}
}
