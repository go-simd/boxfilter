# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/perf_test.go` or is named
here with its reason.

The cases time `painter.PixelPainter.DrawImage` (go-widgets/painter), the
caller of both kernels, against Pillow's area-averaging downscales: a kernel
alone has no Pillow counterpart (Pillow's `reduce` is one fused loop), so the
honest unit is the whole downscale. `AccumPremul` carries those rows because it
does nearly all of the work.

- `SumGroups`: the horizontal pass of an integer ratio; it is inside every `DrawImage ... to ...` row whose horizontal ratio is an integer (64 to 16, 1000 to 250) and has no Pillow counterpart on its own.
