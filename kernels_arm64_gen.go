//go:build ignore

// Command gen produces kernels_arm64.s with go-asmgen: NEON kernels for the
// premultiplied box-filter accumulation and the pixel-group sum.
//
// # accumPremulNEON, 4 pixels per unit, 8 per iteration
//
// One 16-byte load is four RGBA pixels. The premultiplier of a pixel is the byte
// vector (a, a, a, 1): the premultiplied channels are r*a, g*a, b*a and a*1. It
// is built with ONE two-register table lookup, VTBL over {pixels, one}: index 3,
// 7, 11 or 15 picks the alpha byte of that pixel, index 16 picks byte 0 of the
// second table register, which holds the constant 1. VUMULL/VUMULL2 then multiply
// the pixel bytes by the premultiplier (byte x byte -> halfword, at most
// 255*255, exact), giving the premultiplied channels of two pixels per register
// with the four channels of a pixel adjacent. The accumulator row is uint32, four
// per pixel, in the same order, so one VLD1 of four registers holds the four
// pixels' sums and the weighted add is ONE instruction per pixel: VUMLAL (VUMLAL2
// for the upper pixel of the register), acc.4S += prod.4H * w.4H, a widening
// multiply-accumulate exact in 32 bits. That limits w to 16 bits (the Go wrapper
// sends a larger weight to the scalar reference). No deinterleaving load or store
// is used: VLD4/VST4 on the accumulator measured as the bottleneck.
//
// NEON is the ARMv8-A baseline, so there is no feature gate; the integer NEON
// mnemonics are only assemblable on Go 1.27+, hence //go:build arm64 && go1.27.
//
// # sumGroupsNEON
//
// One destination pixel is one 128-bit register: n loads and VADDs of the source
// pixels' .S4 vectors, then one store. Two destination pixels are summed at a
// time to hide the add latency.
//
// Run: go run -modfile=<gen go.mod> kernels_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(accumKernel().Func())
	f.Add(sumKernel().Func())
	f.Add(blocksKernel().Func())
	out := strings.Replace(f.String(), "//go:build arm64\n", "//go:build arm64 && go1.27\n", 1)
	if err := os.WriteFile("kernels_arm64.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_arm64.s")
}

// unit emits the work for 4 pixels: data in V<d> (V<d+1> holds the constant 1,
// the second VTBL table register), premultiplier V<m>, products V<p>/V<p+1>,
// accumulator registers V<a>..V<a+3>.
func unit(b *arm64.Builder, d, m, p, a int) {
	b.Raw(fmt.Sprintf("VTBL V30.B16, [V%d.B16, V%d.B16], V%d.B16", d, d+1, m)).
		Raw(fmt.Sprintf("VUMULL V%d.B8, V%d.B8, V%d.H8", d, m, p)).
		Raw(fmt.Sprintf("VUMULL2 V%d.B16, V%d.B16, V%d.H8", d, m, p+1)).
		Raw(fmt.Sprintf("VLD1 (R1), [V%d.S4, V%d.S4, V%d.S4, V%d.S4]", a, a+1, a+2, a+3)).
		Raw(fmt.Sprintf("VUMLAL V%d.H4, V31.H4, V%d.S4", p, a)).
		Raw(fmt.Sprintf("VUMLAL2 V%d.H8, V31.H8, V%d.S4", p, a+1)).
		Raw(fmt.Sprintf("VUMLAL V%d.H4, V31.H4, V%d.S4", p+1, a+2)).
		Raw(fmt.Sprintf("VUMLAL2 V%d.H8, V31.H8, V%d.S4", p+1, a+3)).
		Raw(fmt.Sprintf("VST1.P [V%d.S4, V%d.S4, V%d.S4, V%d.S4], 64(R1)", a, a+1, a+2, a+3))
}

// accumKernel: R0 src, R1 acc, R4 units (4 pixels) left, R2 w. V30 the VTBL
// index (3,3,3,16, 7,7,7,16, 11,11,11,16, 15,15,15,16), V1 and V3 the constant
// 1 in byte 0, V31 the weight in every halfword lane.
func accumKernel() *arm64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("acc"), abi.Slice("src"), abi.Scalar("w", abi.Uint64)},
		nil,
	)
	b := arm64.NewFunc("accumPremulNEON", sig, 0)
	b.LoadArg("acc_base", "R1").LoadArg("src_base", "R0").LoadArg("src_len", "R4").LoadArg("w", "R2").
		Raw("LSR $4, R4, R4").
		Raw("CBZ R4, done").
		Raw("VDUP R2, V31.H8").
		Raw("MOVD $0x10070707_10030303, R5").Raw("MOVD $0x100F0F0F_100B0B0B, R6").
		Raw("VMOV R5, V30.D[0]").Raw("VMOV R6, V30.D[1]").
		Raw("VEOR V1.B16, V1.B16, V1.B16").Raw("VEOR V3.B16, V3.B16, V3.B16").
		Raw("MOVD $1, R5").Raw("VMOV R5, V1.D[0]").Raw("VMOV R5, V3.D[0]").
		Label("pair").
		Raw("CMP $2, R4").Raw("BLT single").
		Raw("VLD1.P 16(R0), [V0.B16]").Raw("VLD1.P 16(R0), [V2.B16]")
	unit(b, 0, 4, 6, 8)
	unit(b, 2, 5, 12, 16)
	b.Raw("SUB $2, R4, R4").Raw("B pair").
		Label("single").
		Raw("CBZ R4, done").
		Raw("VLD1.P 16(R0), [V0.B16]")
	unit(b, 0, 4, 6, 8)
	b.Label("done").Ret()
	return b
}

// blocksUnit emits the work for ONE register of up to four pixels in V0 (V1 holds
// the constant 1, the second VTBL table register): premultiply, then add the four
// pixels into four independent accumulators V8..V11 (a chain of one dependent add
// per unit, not four) with the widening add UADDW, exact in 32 bits and with no
// multiply: every weight of a block is 1.
func blocksUnit(b *arm64.Builder) {
	b.Raw("VTBL V30.B16, [V0.B16, V1.B16], V4.B16").
		Raw("VUMULL V0.B8, V4.B8, V6.H8").
		Raw("VUMULL2 V0.B16, V4.B16, V7.H8").
		Raw("VUADDW V6.H4, V8.S4, V8.S4").
		Raw("VUADDW2 V6.H8, V9.S4, V9.S4").
		Raw("VUADDW V7.H4, V10.S4, V10.S4").
		Raw("VUADDW2 V7.H8, V11.S4, V11.S4")
}

// blocksKernel: R0 dst, R1 src (start of the first group in row 0), R2 stride in
// bytes, R3 n pixels per row of a block, R4 m rows, R5 groups (destination pixels)
// left, R6 whole units of four pixels per row (n/4), R7 the 0..3 pixels left over,
// R12 the bytes of a group's row (4n), R8 the first pixel of this group in the
// current row, R10 the load cursor, R9 rows left, R11 units left.
//
// One destination pixel is the sum of an n x m block of premultiplied pixels,
// held in four vector accumulators for the whole block: the source is read once
// and nothing but the result is stored. The 1..3 pixels left over (n not a
// multiple of four) are loaded into a zeroed register with lane loads, so no byte
// beyond the block is read, and the zero pixels add nothing.
func blocksKernel() *arm64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("stride", abi.Int64), abi.Scalar("n", abi.Int64), abi.Scalar("m", abi.Int64)},
		nil,
	)
	b := arm64.NewFunc("sumBlocksNEON", sig, 0)
	b.LoadArg("dst_base", "R0").LoadArg("dst_len", "R5").LoadArg("src_base", "R1").
		LoadArg("stride", "R2").LoadArg("n", "R3").LoadArg("m", "R4").
		Raw("LSR $2, R5, R5").
		Raw("CBZ R5, done").
		Raw("LSR $2, R3, R6").Raw("AND $3, R3, R7").Raw("LSL $2, R3, R12").
		Raw("MOVD $0x10070707_10030303, R13").Raw("MOVD $0x100F0F0F_100B0B0B, R14").
		Raw("VMOV R13, V30.D[0]").Raw("VMOV R14, V30.D[1]").
		Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("MOVD $1, R13").Raw("VMOV R13, V1.D[0]").
		Label("group").
		Raw("VEOR V8.B16, V8.B16, V8.B16").Raw("VEOR V9.B16, V9.B16, V9.B16").
		Raw("VEOR V10.B16, V10.B16, V10.B16").Raw("VEOR V11.B16, V11.B16, V11.B16").
		Raw("MOVD R1, R8").Raw("MOVD R4, R9").
		Label("row").
		Raw("MOVD R8, R10").Raw("MOVD R6, R11").
		Raw("CBZ R11, tail").
		Label("unit").
		Raw("VLD1.P 16(R10), [V0.B16]")
	blocksUnit(b)
	b.Raw("SUBS $1, R11, R11").Raw("BNE unit").
		Label("tail").
		Raw("CBZ R7, rowend").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("TST $2, R7").Raw("BEQ one").
		Raw("VLD1.P 8(R10), V0.D[0]").
		Label("one").
		Raw("TST $1, R7").Raw("BEQ last").
		Raw("TST $2, R7").Raw("BNE lane2").
		Raw("VLD1 (R10), V0.S[0]").Raw("B last").
		Label("lane2").
		Raw("VLD1 (R10), V0.S[2]").
		Label("last")
	blocksUnit(b)
	b.Label("rowend").
		Raw("ADD R2, R8, R8").
		Raw("SUBS $1, R9, R9").Raw("BNE row").
		Raw("VADD V8.S4, V9.S4, V8.S4").Raw("VADD V10.S4, V11.S4, V10.S4").Raw("VADD V8.S4, V10.S4, V8.S4").
		Raw("VST1.P [V8.S4], 16(R0)").
		Raw("ADD R12, R1, R1").
		Raw("SUBS $1, R5, R5").Raw("BNE group").
		Label("done").Ret()
	return b
}

// sumKernel: R0 dst, R1 src, R2 groups left, R3 n. Two destination pixels per
// outer iteration (V0, V1 accumulators), then at most one more.
func sumKernel() *arm64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		nil,
	)
	b := arm64.NewFunc("sumGroupsNEON", sig, 0)
	b.LoadArg("dst_base", "R0").LoadArg("dst_len", "R2").LoadArg("src_base", "R1").LoadArg("n", "R3").
		Raw("LSR $2, R2, R2"). // groups = len(dst)/4
		Raw("LSL $4, R3, R5"). // R5 = n*16: bytes of one group
		Label("pair").
		Raw("CMP $2, R2").Raw("BLT single").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("ADD R5, R1, R6"). // second group's source
		Raw("MOVD R3, R4").
		Label("pj").
		Raw("VLD1.P 16(R1), [V2.S4]").
		Raw("VLD1.P 16(R6), [V3.S4]").
		Raw("VADD V2.S4, V0.S4, V0.S4").
		Raw("VADD V3.S4, V1.S4, V1.S4").
		Raw("SUBS $1, R4, R4").Raw("BNE pj").
		Raw("VST1.P [V0.S4], 16(R0)").
		Raw("VST1.P [V1.S4], 16(R0)").
		Raw("MOVD R6, R1"). // continue after the second group
		Raw("SUB $2, R2, R2").Raw("B pair").
		Label("single").
		Raw("CBZ R2, done").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("MOVD R3, R4").
		Label("sj").
		Raw("VLD1.P 16(R1), [V2.S4]").
		Raw("VADD V2.S4, V0.S4, V0.S4").
		Raw("SUBS $1, R4, R4").Raw("BNE sj").
		Raw("VST1.P [V0.S4], 16(R0)").
		Label("done").Ret()
	return b
}
