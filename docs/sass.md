# SASS: what the ladder actually compiles to

Every other kind of evidence in this repository needs a GPU in a known state:
locked clocks, a cooldown, a throttle gate, five process repeats. What a kernel
compiles to needs none of that. It is a function of the source, the toolkit and
the target architecture, and nothing else, so it is the one measurement here that
anybody can reproduce byte for byte and that can be pinned with a golden file.

That is what `experiments/sass/` is: one normalized disassembly per ladder rung,
committed like the ncu text pages, plus a golden copy of the top kernel that a
gate diffs against.

## Producing them

```sh
make sass            # every rung into experiments/sass/
make sass-diff       # the gate: top kernel against the golden
make register-study  # the register ceiling table
```

Or directly, against a build tree anywhere:

```sh
bash benchmarks/capture_sass.sh --build-dir /var/tmp/ckl-build
bash benchmarks/capture_sass.sh --list
bash benchmarks/capture_sass.sh --rung cutlass --stdout
python3 scripts/sass_diff.py --build-dir /var/tmp/ckl-build
python3 benchmarks/register_study.py --build-dir /var/tmp/ckl-build
```

The rungs captured are `naive`, `tiled`, `register`, `cp_async`, `wmma_fp16`,
`wmma_bf16`, `mma_ptx`, `mma_ldm`, `mma_opt`, `mma_opt_bias`, the six tile family
shapes, `splitk`, `streamk` and `cutlass`.

Two of them are narrower than their name suggests, and the rung table in
`benchmarks/capture_sass.sh` says so. The split-K mainloop is the tile family
kernel with a partials pointer, already captured as `tile_128x128x32`, so the
`splitk` capture is the fixup pass, which is the only code split-K adds. And the
tile family and stream-K captures are the tail free instantiation of each shape;
the predicated one is a separate template instantiation with its own entry.

## Normalization, and why each rule is there

A raw `cuobjdump -sass` dump cannot be diffed. Six rules make it comparable
across rebuilds and across machines, and no more than six, because every rule
past what is needed is a way for the gate to stop failing when it should.

1. **One architecture.** A multiarch build carries the same kernel once per
   target, so `cuobjdump --arch sm_120` asks for one ELF section.
2. **One kernel per block.** Everything before the requested `Function` line and
   everything after the next one goes. That also drops the fatbin header, the PTX
   section, and the `identifier =` line, which holds an absolute source path and
   would differ on every machine.
3. **The unnamed namespace module hash is folded.** nvcc spells an anonymous
   namespace as `<length>_GLOBAL__N__<8 hex>_<digits>_<file>_<8 hex>`, with both
   hashes derived from the translation unit. Without this, editing anything in a
   file would rename its kernels and the diff would report a rename instead of
   the instruction that moved.
4. **The instruction address column goes.** `/*0410*/` shifts on every later line
   when one instruction is inserted anywhere earlier.
5. **The encoded instruction words go**, along with the continuation lines that
   carry nothing else. `/* 0x000fc40000000f00 */` is the mnemonic plus the
   scheduling control bits, and the control bits move without the instruction
   stream changing.
6. **`.headerflags` goes and whitespace runs collapse to one space.**
   `cuobjdump` aligns its columns to the widest operand in the kernel, so an
   unrelated edit reflows the whole file.

What survives is the mnemonic, its predicate, its operands and its order, with
branch targets intact. Nothing that describes what the machine will do is
removed.

Above that sits a four line header naming the rung, the architecture, and the
toolkit that assembled it. SASS is a function of the source, the architecture and
the assembler, so leaving the assembler out would let a toolkit upgrade look like
a kernel change. The captures in this tree were assembled by nvcc 13.3.73 for
sm_120, and `scripts/sass_diff.py` says so plainly when the toolkit it finds is
not the one the golden names, rather than printing a diff of the whole kernel.

## The gate

`scripts/sass_diff.py` recaptures `mma_opt` from a fresh build and compares it to
`experiments/sass/golden/mma_opt.sass`, exiting non-zero with a unified diff on
any difference. It is wired into `make sass-diff`.

Ground rule 8 says a gate that cannot fail is not a gate, so both halves were
demonstrated before this page was written. Rebuilding the translation unit with
no source change is a clean pass. Adding one factor to one line of the epilogue,

```text
*p0 = make_float2(alpha * acc[mi][ni][0] * 1.5f, alpha * acc[mi][ni][1]);
```

fails it, with `FMUL R60, R60, 1.5` appearing in the diff alongside the register
allocation that moved around it. Reverting the line makes it pass again.

When a kernel change is intended, regenerate the golden with
`python3 scripts/sass_diff.py --update` and commit it in the same change as the
kernel, so the review sees exactly what moved in the instruction stream.

The gate is not yet a CI job, and there is one thing to settle before it can be.
It needs no GPU, so the existing container build job is the natural home for it,
but that job runs `nvidia/cuda:13.0.1-devel-ubuntu24.04` and this golden was
assembled by nvcc 13.3.73. Two assemblers, two answers, and the job would fail on
the header line before it looked at an instruction. Either the container moves to
the toolkit the golden names, or the golden is regenerated on the container's
toolkit and this machine becomes the one that disagrees. That is a CI ownership
decision, not a decision this page can make, so until it is made the gate runs
locally through `make sass-diff`.

## Reading a capture

A normalized line is the mnemonic, its optional predicate, and its operands:

```text
LDGSTS.E.BYPASS.128 [R2], desc[UR10][R4.64] ;
LDSM.16.M88.4 R24, [R117+0x2000] ;
LDSM.16.MT88.4 R8, [R118] ;
HMMA.16816.F32 R32, R24, R8, R32 ;
BAR.SYNC.DEFER_BLOCKING 0x0 ;
```

- `LDGSTS` is `cp.async`: global to shared with no register round trip.
  `BYPASS` is the `.cg` cache hint, `128` the access width in bits.
- `LDSM.16.M88.4` is `ldmatrix.sync.aligned.m8n8.x4`: one instruction fills four
  fragments for a warp out of shared memory. `MT88` is the `.trans` form, which
  is how the B operand arrives in the layout `mma.sync` wants.
- `LDGDEPBAR` and `DEPBAR.LE` are the `cp.async` commit and wait: they order the
  in flight copies, and they are not block wide barriers.
- `HMMA.16816.F32` is `mma.sync.aligned.m16n8k16.f32.f16.f16.f32`, the largest
  dense FP16 shape on this part.
- `STS` and `LDS` are ordinary shared stores and loads. In a tensor mainloop they
  mean the data took a register detour that `LDGSTS` and `LDSM` exist to avoid.

## The instruction mix

Counted from the committed captures, so these are compile time facts and not
measurements. Each count is per kernel body, which for the mainloop rungs is one
K stage of the software pipeline.

| Rung | Instructions | HMMA | LDSM | LDGSTS | LDG | STS | LDS | STG | BAR |
|---|---|---|---|---|---|---|---|---|---|
| wmma_fp16 | 560 | 56 | 28 | 0 | 30 | 14 | 0 | 32 | 14 |
| mma_ptx | 392 | 8 | 0 | 0 | 34 | 2 | 24 | 64 | 2 |
| mma_ldm | 584 | 56 | 42 | 0 | 46 | 14 | 0 | 64 | 14 |
| mma_opt | 776 | 32 | 12 | 12 | 32 | 0 | 9 | 64 | 1 |
| tile_128x128x32 | 920 | 32 | 12 | 12 | 32 | 0 | 9 | 96 | 1 |
| tile_128x128x64 | 1120 | 64 | 24 | 24 | 32 | 0 | 9 | 96 | 1 |
| streamk | 1912 | 64 | 24 | 24 | 98 | 0 | 26 | 160 | 4 |
| splitk (fixup) | 184 | 0 | 0 | 0 | 30 | 0 | 0 | 1 | 0 |
| cutlass | 1840 | 64 | 24 | 24 | 42 | 128 | 74 | 65 | 40 |

Every HMMA in the table is `HMMA.16816.F32`. Four things in it are worth saying
out loud, because each one is a design decision showing up in the machine code.

**The ladder's story is visible.** `mma_ptx` reads fragments scalar out of shared
memory: 24 `LDS` and no `LDSM` at all, for 8 HMMA. `mma_ldm` replaces those with
42 `LDSM` and gets 56 HMMA out of the same shape. `mma_opt` adds `cp.async`, and
the 12 `LDGSTS` are the first instructions in this ladder that move global data
into shared memory without touching a register.

**The barrier count collapses.** `wmma_fp16` and `mma_ldm` run 14 `BAR` per body;
`mma_opt` and the whole tile family run 1. That is the three stage pipeline
paying off: with three or more stages, the buffer written at iteration s was last
read at iteration s-1, which the iteration-s barrier already fences, so one
barrier per K stage is enough. `cutlass` runs 40, most of them in its epilogue.

**`mma_opt` and the CUTLASS reference line amortize `ldmatrix` identically.**
Both sit at 32 HMMA per 12 LDSM and 64 per 24 respectively, which is the same
2.67 ratio. Whatever separates the two kernels in wall clock, it is not how many
`mma.sync` each `ldmatrix` feeds.

**Their epilogues are opposites.** `mma_opt` has zero `STS`: it stores 64 bit
pairs to global straight out of the `mma` fragment layout. CUTLASS has 128 `STS`
and 74 `LDS`, because it stages the accumulator through shared memory to
rearrange it into fully coalesced global stores, and 65 `STG` against
`mma_opt`'s 64 is what that buys. Which trade wins here is exactly the kind of
question the gap analysis in `docs/cutlass.md` is set up to answer, and it is
pending a profile.

## The top kernel's schedule

`gemm_mma_opt_kernel<false>`, 776 instructions, 122 registers, 0 bytes spilled,
1 barrier, 49152 bytes of dynamic shared memory, 256 threads.

The body is one K stage of a three stage pipeline over a 128 by 128 block tile
with a K step of 32. Eight warps in a 2 by 4 layout each own a 64 by 32 region,
which is a 4 by 4 grid of 16 by 8 output tiles over two K substeps of 16. So per
stage each warp issues:

- 12 `LDGSTS` covering the next stage's A and B tiles, issued before the math on
  the current stage rather than after it, which is the whole point of `cp.async`;
- 12 `LDSM.16` filling fragments out of shared memory: per K substep, 4 plain
  ones for the warp's 4 A tiles and 2 transposed ones for its 4 B tiles, since
  one `ldmatrix.x4.trans` covers two adjacent 16 by 8 output tiles, twice over
  for the two substeps;
- 32 `HMMA.16816.F32`, one per output tile per substep;
- 1 `BAR.SYNC`, plus the `LDGDEPBAR` and `DEPBAR.LE` pair that waits on the
  `cp.async` group rather than on the block.

The epilogue is 64 `STG.E.64`: 32 output tiles of 4 accumulators each, paired
into 64 bit stores. That pairing is why `c` has to be 8 byte aligned, which the
launcher checks rather than assuming.

Where ptxas put the registers, and what it costs to move them, is the next
section. Where the warps actually stall is pending an ncu round: none of the
issue stall counters can be read off a disassembly, and the numbers in
`experiments/results/ncu/` do not yet cover this question. That round is owner
work at locked clocks, and this page will say what it found rather than what it
expected.

## The register allocation study

`benchmarks/register_study.py` reassembles the top kernel across register
ceilings and blocks per SM floors, and writes
`experiments/results/register_study.csv`. Every column but throughput is a
compile time fact; throughput reads "pending" and stays that way until the owner
runs it at locked clocks.

There is a wrinkle worth knowing about, because it is easy to run this study and
get a table of nothing. `nvcc -maxrregcount` is ignored for a kernel carrying
`__launch_bounds__`, and so is `ptxas -maxrregcount` when the PTX carries
`.maxntid`. Launch bounds win, by design. The kernel declares
`__launch_bounds__(256)`, so a naive sweep returns 122 registers in every row.
The study therefore emits PTX once with the build's own flags, and edits one
performance tuning directive in a copy of it: the register axis drops `.maxntid`
so the ceiling is the only constraint, and the launch bounds axis keeps it and
adds `.minnctapersm`. Unconstrained, both variants still assemble to 122
registers with no spill, which is the row each axis opens with.

| Axis | maxrregcount | minnctapersm | Registers | Spill stores | Spill loads | Blocks by registers | Blocks per SM | Limiter |
|---|---|---|---|---|---|---|---|---|
| maxrregcount | none | 1 | 122 | 0 | 0 | 2 | 2 | shared memory |
| maxrregcount | 64 | 1 | 64 | 544 | 544 | 4 | 2 | shared memory |
| maxrregcount | 80 | 1 | 80 | 172 | 164 | 3 | 2 | shared memory |
| maxrregcount | 96 | 1 | 96 | 12 | 12 | 2 | 2 | shared memory |
| maxrregcount | 111 | 1 | 109 | 0 | 0 | 2 | 2 | shared memory |
| maxrregcount | 128 | 1 | 122 | 0 | 0 | 2 | 2 | shared memory |
| maxrregcount | 168 | 1 | 122 | 0 | 0 | 2 | 2 | shared memory |
| launch bounds | none | 1 | 122 | 0 | 0 | 2 | 2 | shared memory |
| launch bounds | none | 2 | 122 | 0 | 0 | 2 | 2 | shared memory |
| launch bounds | none | 3 | 80 | 172 | 164 | 3 | 2 | shared memory |
| launch bounds | none | 4 | 64 | 544 | 544 | 4 | 2 | shared memory |

Three readings, all of them compile time facts rather than performance claims.

**109 registers are free.** A ceiling of 111 lands the kernel at 109 with no
spill at all, and 96 costs only 12 bytes each way. So the kernel does not need
122; ptxas simply has no reason to use fewer when nothing asks.

**Nothing is bought by asking.** The occupancy arithmetic on this part is
65536 registers, 100 KB of shared memory and 48 warps per SM. At 256 threads and
49152 bytes of shared memory per block, shared memory alone allows two blocks per
SM, and it stays the binding constraint at every register count in the table. At
64 registers the register file would allow four blocks; the shared memory budget
still allows two. Cutting registers buys spill traffic and no residency.

**The blocks per SM floor is the same lever from the other end.** Asking for
three blocks per SM produces exactly the 80 register, 172 byte spill row that
`-maxrregcount=80` produces, because ptxas gets there the same way. And it would
not deliver three blocks either: shared memory would still cap it at two.

The lever that would move occupancy on this kernel is the shared memory
footprint, which means the stage count or the tile shape, not the register
ceiling. The tile family in `docs/gemm.md` is that experiment; `tile_128x64x64`
and `tile_64x128x64` trade the block tile for a different footprint, and their
own register counts are in the table there.
