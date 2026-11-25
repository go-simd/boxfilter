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
	f.Add(blocksKernel().Func())
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

// blocksUnit emits the work for ONE register of up to four pixels in X0:
// premultiply as accumKernel does, widen the words to dwords and add them into
// the accumulators Y8 (pixels 0 and 1 of the unit) and Y9 (pixels 2 and 3), whose
// two 128-bit halves are folded once per block.
func blocksUnit(b *amd64.Builder) {
	b.Raw("VPSHUFB X10, X0, X1").
		Raw("VPMOVZXBW X1, Y1").
		Raw("VPOR Y11, Y1, Y1").
		Raw("VPMOVZXBW X0, Y0").
		Raw("VPMULLW Y1, Y0, Y0").
		Raw("VPMOVZXWD X0, Y2").
		Raw("VEXTRACTI128 $1, Y0, X3").
		Raw("VPMOVZXWD X3, Y3").
		Raw("VPADDD Y2, Y8, Y8").
		Raw("VPADDD Y3, Y9, Y9")
}

// blocksKernel: DI dst, SI src (the start of the first group in row 0), DX
// stride in bytes, CX groups left, R8 rows m, R9 whole units of four pixels per
// row (n/4), R10 the 0..3 pixels left over, R11 the bytes of a group's row (4n),
// R12 the group's first pixel in the current row, BX the load cursor, R13 rows
// left, AX units left. X10 the VPSHUFB mask, Y11 words (0,0,0,1) repeating.
//
// One destination pixel is the sum of an n x m block of premultiplied pixels,
// held in two vector accumulators for the whole block: the source is read once
// and only the result is stored. The 1..3 pixels left over are loaded into a
// zeroed register with 4, 8 or 8+4 byte loads, so no byte beyond the block is
// read, and the zero pixels add nothing.
func blocksKernel() *amd64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("stride", abi.Int64), abi.Scalar("n", abi.Int64), abi.Scalar("m", abi.Int64)},
		nil,
	)
	b := amd64.NewFunc("sumBlocksAVX2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("dst_len", "CX").LoadArg("src_base", "SI").
		LoadArg("stride", "DX").LoadArg("n", "R9").LoadArg("m", "R8").
		Raw("SHRQ $2, CX").
		Raw("TESTQ CX, CX").Raw("JZ done").
		Raw("MOVQ R9, R10").Raw("ANDQ $3, R10").Raw("MOVQ R9, R11").Raw("SHLQ $2, R11").Raw("SHRQ $2, R9").
		Raw("MOVQ $0x80070707_80030303, AX").Raw("MOVQ $0x800F0F0F_800B0B0B, BX").
		Raw("VMOVQ AX, X10").Raw("VPINSRQ $1, BX, X10, X10").
		Raw("VPCMPEQQ Y11, Y11, Y11").Raw("VPSRLQ $63, Y11, Y11").Raw("VPSLLQ $48, Y11, Y11").
		Label("group").
		Raw("VPXOR Y8, Y8, Y8").Raw("VPXOR Y9, Y9, Y9").
		Raw("MOVQ SI, R12").Raw("MOVQ R8, R13").
		Label("row").
		Raw("MOVQ R12, BX").Raw("MOVQ R9, AX").
		Raw("TESTQ AX, AX").Raw("JZ tail").
		Label("unit").
		Raw("VMOVDQU (BX), X0")
	blocksUnit(b)
	b.Raw("ADDQ $16, BX").Raw("DECQ AX").Raw("JNZ unit").
		Label("tail").
		Raw("TESTQ R10, R10").Raw("JZ rowend").
		Raw("VPXOR X0, X0, X0").
		Raw("CMPQ R10, $1").Raw("JNE two").
		Raw("VMOVD (BX), X0").Raw("JMP last").
		Label("two").
		Raw("VMOVQ (BX), X0").
		Raw("CMPQ R10, $2").Raw("JE last").
		Raw("VPINSRD $2, 8(BX), X0, X0").
		Label("last")
	blocksUnit(b)
	b.Label("rowend").
		Raw("ADDQ DX, R12").
		Raw("DECQ R13").Raw("JNZ row").
		Raw("VPADDD Y9, Y8, Y8").
		Raw("VEXTRACTI128 $1, Y8, X1").Raw("VPADDD X1, X8, X8").
		Raw("VMOVDQU X8, (DI)").
		Raw("ADDQ $16, DI").Raw("ADDQ R11, SI").
		Raw("DECQ CX").Raw("JNZ group").
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
