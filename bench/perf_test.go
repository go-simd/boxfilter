package bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os/exec"
	"testing"

	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
	"github.com/go-widgets/painter"
)

// The Go side is painter.PixelPainter.DrawImage, the caller of the two kernels
// of this module (AccumPremul does nearly all the work); the references are
// Pillow's two area-averaging downscales on the SAME bytes:
//
//   - Image.resize(BOX) on a premultiplied RGBa image: the same arithmetic
//     (premultiplied area mean), non-integer ratios included;
//   - Image.reduce(n) on an OPAQUE RGBA image: Pillow's integer-factor fast path,
//     which does not premultiply, so it equals ours only where alpha is 255.
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, denominator
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestSameData .                      the two sides time the same bytes (needs uv)

// pixel is byte i of the test picture: a fixed hash of i, alpha forced to 255
// when opaque. The Python setup below computes the same bytes.
func pixel(i int, opaque bool) byte {
	if opaque && i%4 == 3 {
		return 255
	}
	return byte((uint64(i) * 2654435761) >> 13)
}

func picture(w, h int, opaque bool) []byte {
	b := make([]byte, w*h*4)
	for i := range b {
		b[i] = pixel(i, opaque)
	}
	return b
}

// pyData is the Python expression of picture(w, h, opaque) as a uint8 array.
func pyData(w, h int, opaque bool) string {
	if opaque {
		return fmt.Sprintf("np.where(np.arange(%d)%%4==3, 255, ((np.arange(%d,dtype=np.uint64)*2654435761)>>13)&255).astype(np.uint8)", w*h*4, w*h*4)
	}
	return fmt.Sprintf("(((np.arange(%d,dtype=np.uint64)*2654435761)>>13)&255).astype(np.uint8)", w*h*4)
}

// pySetup is what refbench hashes: the picture, as the one argument.
func pySetup(w, h int, opaque bool) string { return "(" + pyData(w, h, opaque) + ",)" }

// pyCall builds the Pillow image OUTSIDE the timed loop (the worker evaluates
// the call expression once, before its clock starts) and returns the function
// that is timed: op applied to the image.
func pyCall(w, h int, opaque bool, mode, op string) string {
	conv := ""
	if mode != "RGBA" {
		conv = fmt.Sprintf(".convert(%q)", mode)
	}
	return fmt.Sprintf(`(lambda I: (lambda im: lambda a: %s)(I.frombytes("RGBA",(%d,%d),%s.tobytes())%s))(__import__("PIL.Image",fromlist=["_"]))`,
		op, w, h, pyData(w, h, opaque), conv)
}

var sink byte

type row struct {
	name          string
	sw, sh        int
	dw, dh        int
	opaque        bool
	ref, mode, op string
}

func rows() []row {
	var rs []row
	for _, s := range []struct{ sw, sh, dw, dh int }{{256, 256, 24, 24}, {64, 64, 16, 16}, {1000, 700, 250, 175}} {
		rs = append(rs, row{
			name: fmt.Sprintf("DrawImage %dx%d to %dx%d translucent", s.sw, s.sh, s.dw, s.dh),
			sw:   s.sw, sh: s.sh, dw: s.dw, dh: s.dh,
			ref: "Pillow resize(BOX) on RGBa", mode: "RGBa",
			op: fmt.Sprintf("im.resize((%d,%d), I.BOX)", s.dw, s.dh),
		})
	}
	for _, s := range []struct{ sw, sh, dw, dh int }{{64, 64, 16, 16}, {1000, 700, 250, 175}} {
		rs = append(rs, row{
			name: fmt.Sprintf("DrawImage %dx%d to %dx%d opaque", s.sw, s.sh, s.dw, s.dh),
			sw:   s.sw, sh: s.sh, dw: s.dw, dh: s.dh, opaque: true,
			ref: "Pillow reduce(4) on RGBA", mode: "RGBA",
			op: fmt.Sprintf("im.reduce(%d)", s.sw/s.dw),
		})
	}
	// The same reduce(4) on the other four-byte modes Pillow has: reduce does
	// not premultiply, so on RGBa (already premultiplied) and RGBX (no alpha)
	// it is the same loop; they show whether the mode changes Pillow's speed.
	for _, mode := range []string{"RGBa", "RGBX"} {
		rs = append(rs, row{
			name: "DrawImage 1000x700 to 250x175 opaque vs " + mode,
			sw:   1000, sh: 700, dw: 250, dh: 175, opaque: true,
			ref: "Pillow reduce(4) on " + mode, mode: mode, op: "im.reduce(4)",
		})
	}
	return rs
}

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range rows() {
		src := picture(r.sw, r.sh, r.opaque)
		p := painter.NewPixelPainter(make([]byte, r.dw*r.dh*4), r.dw, r.dh)
		dst := painter.Rect{W: r.dw, H: r.dh}
		c := refbench.Case{
			Name:      r.name,
			Func:      "AccumPremul",
			Reference: r.ref,
			Parity:    refbench.ParityTolerance, // within 0.5 of the exact mean (painter's TestBoxExactWithinStatedBound); Pillow rounds its own way
			Go: func(iters int) {
				for i := 0; i < iters; i++ {
					p.DrawImage(dst, src, r.sw, r.sh)
				}
				sink = p.Buf[0]
			},
		}
		if w != nil {
			// The "n" of the worker is unused: the sizes are in the expressions.
			c.Ref = w.Timer(pySetup(r.sw, r.sh, r.opaque), pyCall(r.sw, r.sh, r.opaque, r.mode, r.op), r.sw)
		}
		out = append(out, c)
	}
	return out
}

// measuring says whether rbtest's -perf.run or -perf.resume is set: only then
// is the Python worker (and uv) needed; CI mode needs no reference.
func measuring() bool {
	for _, name := range []string{"perf.run", "perf.resume"} {
		if f := flag.Lookup(name); f != nil && f.Value.String() == "true" {
			return true
		}
	}
	return false
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if measuring() {
		uv, err := exec.LookPath("uv")
		if err != nil {
			t.Fatalf("uv is needed to run the pinned Python reference: %v", err)
		}
		w, err = refbench.StartPython(context.Background(), uv, "refs.txt")
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
	}
	opt := refbench.Options{Repo: "go-simd/boxfilter"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, opt)
}

// TestSameData: both sides time the same bytes (the SHA-256 of the picture is
// the same in Go and in the Python worker).
func TestSameData(t *testing.T) {
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv not installed")
	}
	w, err := refbench.StartPython(context.Background(), uv, "refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	for _, r := range rows() {
		got, err := w.DataHash(pySetup(r.sw, r.sh, r.opaque), r.sw)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(picture(r.sw, r.sh, r.opaque))
		if want := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("%s: python %s, go %s", r.name, got, want)
		}
	}
}
