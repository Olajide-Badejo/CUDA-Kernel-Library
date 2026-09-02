# Top level entry points. Run inside WSL2 where the CUDA toolchain lives.
# Targets grow as phases land; the definition of done is that `make setup`
# followed by `make all` reproduces every number from a clean clone.

BUILD_DIR ?= build
BUILD_TYPE ?= Release
GENERATOR ?= Ninja
CTEST_LABELS ?=

# report, roofline, sweep collide with directory names, so they must be phony or
# make treats the directory as an up to date target and does nothing.
.PHONY: all setup configure build test bench roofline sweep sweep-quick tile-sweep \
        summary report check-style dash provenance sass sass-diff register-study \
        clean help

help:
	@echo "Targets:"
	@echo "  setup        configure the build tree"
	@echo "  build        compile all targets"
	@echo "  test         run correctness tests (needs a GPU)"
	@echo "  bench        run the default GEMM benchmark (needs a GPU)"
	@echo "  sweep        full measurement protocol sweep (needs a GPU, locked clocks)"
	@echo "  tile-sweep   the tile family decision table (needs a GPU)"
	@echo "  summary      rebuild summary.csv from rows already on file"
	@echo "  sass         capture every ladder rung's SASS into experiments/sass"
	@echo "  sass-diff    fail if the top kernel's SASS left the committed golden"
	@echo "  register-study  reassemble the top kernel under register ceilings"
	@echo "  check-style  run the dash and provenance gates"
	@echo "  all          build then test then check-style"
	@echo "  clean        remove the build tree"

setup configure:
	cmake -S . -B $(BUILD_DIR) -G $(GENERATOR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

bench: build
	./$(BUILD_DIR)/benchmarks/bench_gemm

# Measure the roofline (ceilings plus the ladder) and render the figure.
roofline: build
	./$(BUILD_DIR)/tools/ckl_roofline
	python3 scripts/plot_roofline.py

# Full resumable sweep across every family under the Section 13 protocol: locked
# clocks, five independent process repeats per configuration with a bootstrap
# interval, vendor baselines measured once per shape, shuffled order with
# cooldowns, and a throttle gate that fails the run. Refreshes the canonical
# summary.csv from this commit's rows only. Needs a GPU and clock control; see
# docs/benchmarking.md for what to do when the clock cannot be locked.
sweep: build
	python3 benchmarks/sweep.py

sweep-quick: build
	python3 benchmarks/sweep.py --quick

# The tile family decision table the dispatch heuristic reads. Same protocol,
# imported from the sweep driver rather than copied into it.
tile-sweep: build
	python3 benchmarks/tile_sweep.py

# Rebuild the canonical summary from rows already on file, without measuring.
# SUMMARY_COMMIT selects which commit's rows it is built from; the default is
# HEAD, and a summary never mixes commits.
summary:
	python3 benchmarks/sweep.py --refresh-only $(if $(SUMMARY_COMMIT),--commit $(SUMMARY_COMMIT),)

# Instruction level evidence. All three read the built objects rather than the
# GPU, so they run on any machine with the toolkit and give the same answer.
sass: build
	bash benchmarks/capture_sass.sh --build-dir $(BUILD_DIR)

# The gate: the top kernel's SASS against the committed golden. Non zero exit and
# a unified diff on any difference. Regenerate the golden deliberately with
# `python3 scripts/sass_diff.py --update` when a kernel change is intended.
sass-diff: build
	python3 scripts/sass_diff.py --build-dir $(BUILD_DIR)

# The register allocation study: the top kernel reassembled across maxrregcount
# ceilings and blocks per SM floors, with registers, spills and occupancy filled
# in and the throughput column left pending until the owner runs it at locked
# clocks.
register-study: build
	python3 benchmarks/register_study.py --build-dir $(BUILD_DIR)

check-style: dash provenance

dash:
	python3 scripts/check_no_dashes.py .

# Every commit hash recorded in a results file has to resolve in git, or be
# listed as a known dead v1 hash in experiments/results/legacy_hashes.txt.
provenance:
	python3 scripts/check_provenance.py

# Regenerate figures and tables from the canonical results, then build both PDFs.
# Does not depend on the CUDA build: it works from the committed summary.csv and
# figures, so it runs on a machine without a GPU or toolkit (and in CI). Dash
# check runs last so a stray dash in the prose fails the report build.
report:
	python3 scripts/gen_report_assets.py
	cd report && latexmk -pdf -interaction=nonstopmode -output-directory=build main.tex
	cd report_debug && latexmk -pdf -interaction=nonstopmode -output-directory=build debug_report.tex
	mkdir -p reports
	cp report/build/main.pdf reports/main_report.pdf
	cp report_debug/build/debug_report.pdf reports/debug_report.pdf
	python3 scripts/check_no_dashes.py .

# `all` is the reproduction target: build, test, sweep, report, style gate, from a
# clean tree. The sweep needs a GPU; the report builds from the committed summary.
all: build test sweep report check-style

clean:
	rm -rf $(BUILD_DIR) report/build report_debug/build
