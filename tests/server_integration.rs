//! Integration tests for the inferrs HTTP server.
//!
//! These tests start the binary against a real model and verify that
//! `/v1/chat/completions` returns a well-formed OpenAI-compatible response.
//!
//! # Running
//!
//! The tests require network access to download models from HuggingFace Hub
//! the first time they run (subsequent runs use the local cache).
//!
//! ```
//! cargo test --test server_integration -- --nocapture
//! ```
//!
//! Each test starts `inferrs serve <model>` on a random free port, waits for
//! the server to become healthy, sends a chat completion request, and asserts
//! the response structure.
