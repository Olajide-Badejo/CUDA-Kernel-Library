# Top level entry points. Run inside WSL2 where the CUDA toolchain lives.
# Targets grow as phases land; the definition of done is that `make setup`
# followed by `make all` reproduces every number from a clean clone.

BUILD_DIR ?= build
BUILD_TYPE ?= Release
GENERATOR ?= Ninja
CTEST_LABELS ?=

# Appended to every configure this file runs. It exists because the host
# compiler is not always the default one: on a machine whose default GCC is 15,
# nvcc 13.3 cannot parse libstdc++'s `if consteval`, and the fix is
#   make build CMAKE_EXTRA="-DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14"
# See the host compiler section of docs/building.md. The CI containers ship
# GCC 13 and need nothing here.
CMAKE_EXTRA ?=

# The style tools, overridable so a checkout can point at a pip venv:
#   make check-style CLANG_FORMAT=.venv/bin/clang-format CLANG_TIDY=.venv/bin/clang-tidy
# The versions of record are in requirements-dev.txt, and they are exact. The
# tree is formatted for clang-format 20.1.7; version 19 accepts 13 files that
# 20.1.7 rewrites, so "whatever clang-format is on this machine" is not a gate.
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
RUFF ?= ruff
YAMLLINT ?= yamllint

# clang-tidy reads a compile database, and CMake's Ninja generator writes GCC's
# module scanning flags into every C++ command line. clang rejects those, so the
# analysis build is its own tree configured with scanning off. Nothing is
# compiled in it; the headers clang-tidy needs are generated at configure time.
TIDY_BUILD_DIR ?= $(BUILD_DIR)/tidy

# Reproducible figures. matplotlib stamps the wall clock into a PDF unless this
# is set, and the report job diffs report/figures byte for byte. The value is a
# fixed instant (2025-01-01T00:00:00Z) rather than a commit date, because the
# same data has to draw the same bytes on every machine and every branch.
SOURCE_DATE_EPOCH ?= 1735689600
export SOURCE_DATE_EPOCH

# Everything clang-format and clang-tidy have an opinion about.
STYLE_SOURCES := $(shell find include src tests benchmarks examples tools -type f \
    \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \
       -o -name '*.cu' -o -name '*.cuh' \) 2>/dev/null)

# report, roofline, sweep collide with directory names, so they must be phony or
# make treats the directory as an up to date target and does nothing.
.PHONY: all setup configure build test bench roofline sweep sweep-quick tile-sweep \
        summary report check-style style format format-check tidy tidy-configure \
        ruff yamllint dash provenance doxygen sass sass-diff register-study \
        perf-regression clean help

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
	@echo "  perf-regression  tonight's quick sweep against the committed baseline"
	@echo "  format       rewrite every source file the way clang-format wants it"
	@echo "  format-check clang-format, read only, non zero on any difference"
	@echo "  tidy         clang-tidy over the host translation units"
	@echo "  doxygen      build the API documentation, zero warnings"
	@echo "  check-style  the whole style gate, exactly what the CI style job runs"
	@echo "  all          build then test then check-style"
	@echo "  clean        remove the build tree"

setup configure:
	cmake -S . -B $(BUILD_DIR) -G $(GENERATOR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_EXTRA)

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

# The style gate, in the order the CI style job runs it. Every step here fails
# the build on its own; none of them is advisory.
check-style: format-check ruff yamllint dash provenance tidy

# An alias, because `make style` is what fingers type.
style: check-style

format:
	$(CLANG_FORMAT) -i $(STYLE_SOURCES)

format-check:
	@$(CLANG_FORMAT) --version
	$(CLANG_FORMAT) --dry-run --Werror $(STYLE_SOURCES)

ruff:
	$(RUFF) check .

yamllint:
	$(YAMLLINT) .github/workflows

tidy-configure:
	cmake -S . -B $(TIDY_BUILD_DIR) -G $(GENERATOR) \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
	    -DCKL_WERROR=OFF $(CMAKE_EXTRA)

tidy: tidy-configure
	CLANG_TIDY=$(CLANG_TIDY) python3 scripts/run_clang_tidy.py -p $(TIDY_BUILD_DIR)

# Zero warnings, because the Doxyfile sets WARN_AS_ERROR. A header that loses
# its tags fails here rather than shipping an empty page.
doxygen:
	doxygen Doxyfile

# Tonight's quick sweep against the committed baseline. Needs a GPU, and needs
# experiments/results/perf_baseline.csv to exist; without it the script says so
# and exits non-zero rather than passing on no evidence.
perf-regression: build
	python3 scripts/perf_regression.py --sweep-arg --bench \
	    --sweep-arg $(BUILD_DIR)/benchmarks/bench_all

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
