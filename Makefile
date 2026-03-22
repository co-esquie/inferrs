.DEFAULT_GOAL := build

.PHONY: build all fmt clippy test

build:
	cargo build

all: fmt clippy test build

fmt:
	cargo fmt --check

clippy:
	cargo clippy -- -D warnings

test:
	cargo test
