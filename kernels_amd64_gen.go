//go:build ignore

// Command gen produces kernels_amd64.s with go-asmgen: AVX2 kernels for the
// premultiplied box-filter accumulation and the pixel-group sum.
//
// # accumPremulAVX2, 4 pixels per iteration
//
// One 16-byte load is four RGBA pixels. VPSHUFB copies each pixel's alpha byte
// over its three colour bytes (index 3, 7, 11, 15; 0x80 zeroes the fourth), the
// bytes are zero-extended to words, and the fourth word of each pixel is set to 1
// with a VPOR: the premultiplier (a, a, a, 1). VPMULLW against the zero-extended
// pixel words gives the premultiplied channels r*a, g*a, b*a and a, each at most
// 255*255 so the low 16 bits are the whole product. The words are widened to
// dwords, multiplied by the broadcast weight (VPMULLD, wrapping like the uint32
// sum it feeds) and added to the accumulator row, which holds four uint32 per
// pixel in the same order, so pixels never need to be deinterleaved.
//
// AVX2 is detected at run time (the Go wrapper falls back to the scalar
// reference without it).
//
// # sumGroupsAVX2
//
// One destination pixel is one 128-bit register: n loads and VPADDDs, then one
// store; two destination pixels at a time to hide the add latency.
//
// Run: go run -modfile=<gen go.mod> kernels_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(accumKernel().Func())
	f.Add(sumKernel().Func())
	if err := os.WriteFile("kernels_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_amd64.s")
}

// accumKernel: SI src, DI acc, CX units (4 pixels) left, DX w. X10 the VPSHUFB
// mask, Y11 words (0,0,0,1) repeating, Y12 the weight broadcast to every dword.
func accumKernel() *amd64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("acc"), abi.Slice("src"), abi.Scalar("w", abi.Uint64)},
		nil,
	)
	b := amd64.NewFunc("accumPremulAVX2", sig, 0)
	b.LoadArg("acc_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX").LoadArg("w", "DX").
		Raw("SHRQ $4, CX").
		Raw("TESTQ CX, CX").Raw("JZ done").
		Raw("MOVQ $0x80070707_80030303, AX").Raw("MOVQ $0x800F0F0F_800B0B0B, BX").
		Raw("VMOVQ AX, X10").Raw("VPINSRQ $1, BX, X10, X10").
		Raw("VPCMPEQQ Y11, Y11, Y11").Raw("VPSRLQ $63, Y11, Y11").Raw("VPSLLQ $48, Y11, Y11").
		Raw("VMOVQ DX, X12").Raw("VPBROADCASTD X12, Y12").
		Label("loop").
		Raw("VMOVDQU (SI), X0").
		Raw("VPSHUFB X10, X0, X1").
		Raw("VPMOVZXBW X1, Y1").
		Raw("VPOR Y11, Y1, Y1").
		Raw("VPMOVZXBW X0, Y0").
		Raw("VPMULLW Y1, Y0, Y0").
		Raw("VPMOVZXWD X0, Y2").
		Raw("VEXTRACTI128 $1, Y0, X3").
		Raw("VPMOVZXWD X3, Y3").
		Raw("VPMULLD Y12, Y2, Y2").
		Raw("VPMULLD Y12, Y3, Y3").
		Raw("VPADDD (DI), Y2, Y2").
		Raw("VPADDD 32(DI), Y3, Y3").
		Raw("VMOVDQU Y2, (DI)").
		Raw("VMOVDQU Y3, 32(DI)").
		Raw("ADDQ $16, SI").Raw("ADDQ $64, DI").
		Raw("DECQ CX").Raw("JNZ loop").
		Raw("VZEROUPPER").
		Label("done").Ret()
	return b
}

// sumKernel: DI dst, SI src, CX groups left, BX n, R8 bytes of one group.
func sumKernel() *amd64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		nil,
	)
	b := amd64.NewFunc("sumGroupsAVX2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("dst_len", "CX").LoadArg("src_base", "SI").LoadArg("n", "BX").
		Raw("SHRQ $2, CX").
		Raw("MOVQ BX, R8").Raw("SHLQ $4, R8").
		Label("pair").
		Raw("CMPQ CX, $2").Raw("JL single").
		Raw("VPXOR X0, X0, X0").Raw("VPXOR X1, X1, X1").
		Raw("LEAQ (SI)(R8*1), R9"). // second group's source
		Raw("MOVQ BX, R10").
		Label("pj").
		Raw("VPADDD (SI), X0, X0").
		Raw("VPADDD (R9), X1, X1").
		Raw("ADDQ $16, SI").Raw("ADDQ $16, R9").
		Raw("DECQ R10").Raw("JNZ pj").
		Raw("VMOVDQU X0, (DI)").Raw("VMOVDQU X1, 16(DI)").
		Raw("ADDQ $32, DI").
		Raw("MOVQ R9, SI").
		Raw("SUBQ $2, CX").Raw("JMP pair").
		Label("single").
		Raw("TESTQ CX, CX").Raw("JZ done").
		Raw("VPXOR X0, X0, X0").
		Raw("MOVQ BX, R10").
		Label("sj").
		Raw("VPADDD (SI), X0, X0").
		Raw("ADDQ $16, SI").
		Raw("DECQ R10").Raw("JNZ sj").
		Raw("VMOVDQU X0, (DI)").
		Label("done").Ret()
	return b
}
