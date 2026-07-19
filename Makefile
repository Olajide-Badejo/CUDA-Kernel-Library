# Top level entry points. Run inside WSL2 where the CUDA toolchain lives.
# Targets grow as phases land; the definition of done is that `make setup`
# followed by `make all` reproduces every number from a clean clone.

BUILD_DIR ?= build
BUILD_TYPE ?= Release
GENERATOR ?= Ninja
CTEST_LABELS ?=

.PHONY: all setup configure build test bench check-style dash clean help

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

# `all` is the reproduction target. It gains the sweep and the report as those
# phases land; today it builds, tests, and runs the style gate.
all: build test check-style

clean:
	rm -rf $(BUILD_DIR)
