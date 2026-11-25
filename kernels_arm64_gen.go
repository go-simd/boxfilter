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
