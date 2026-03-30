.DEFAULT_GOAL := build

# Packages that can be built/tested without GPU toolchains (CUDA, ROCm).
NO_GPU_PKGS := -p inferrs -p inferrs-backend-vulkan

.PHONY: build all fmt clippy test

build:
	cargo build $(NO_GPU_PKGS)

all: fmt clippy test build

fmt:
	cargo fmt --check $(NO_GPU_PKGS)

clippy:
	cargo clippy $(NO_GPU_PKGS) -- -D warnings

test:
	cargo test $(NO_GPU_PKGS)
