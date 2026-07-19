# Top level entry points. Run inside WSL2 where the CUDA toolchain lives.
# Targets grow as phases land; the definition of done is that `make setup`
# followed by `make all` reproduces every number from a clean clone.

BUILD_DIR ?= build
BUILD_TYPE ?= Release
GENERATOR ?= Ninja
CTEST_LABELS ?=

# report, roofline, sweep collide with directory names, so they must be phony or
# make treats the directory as an up to date target and does nothing.
.PHONY: all setup configure build test bench roofline sweep sweep-quick report \
        check-style dash clean help

help:
	@echo "Targets:"
	@echo "  setup        configure the build tree"
	@echo "  build        compile all targets"
	@echo "  test         run correctness tests (needs a GPU)"
	@echo "  bench        run the default GEMM benchmark (needs a GPU)"
	@echo "  check-style  run the dash gate (clang-format, clang-tidy, ruff added later)"
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
	./$(BUILD_DIR)/src/profiler/roofline
	python3 scripts/plot_roofline.py

# Full resumable sweep across every family; refreshes the canonical summary.csv.
sweep: build
	python3 benchmarks/sweep.py

sweep-quick: build
	python3 benchmarks/sweep.py --quick

check-style: dash

dash:
	python3 scripts/check_no_dashes.py .

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
