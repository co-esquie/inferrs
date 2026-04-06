//! `inferrs pull` — pre-download a HuggingFace model to the local cache.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Clone)]
pub struct PullArgs {
    /// HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)
    pub model: String,

    /// Git branch or tag on HuggingFace Hub
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Quantize weights and cache the result as a GGUF file.
    ///
    /// Accepted formats (case-insensitive): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    /// Q2K, Q3K, Q4K (Q4_K_M), Q5K, Q6K.
    ///
    /// When used as a plain flag (`--quantize`) the default Q4_K_M (= Q4K) is used.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"), require_equals(true),
          value_name = "FORMAT")]
    pub quantize: Option<String>,
}

pub fn run(args: PullArgs) -> Result<()> {
    let quant_dtype = args
        .quantize
        .as_deref()
        .map(crate::quantize::parse_format)
        .transpose()?;

    let files = crate::hub::download_and_maybe_quantize(&args.model, &args.revision, quant_dtype)?;

    println!("Pulled {}", args.model);
    println!("  config:    {}", files.config_path.display());
    println!("  tokenizer: {}", files.tokenizer_path.display());
    for w in &files.weight_paths {
        println!("  weights:   {}", w.display());
    }
    if let Some(gguf) = &files.gguf_path {
        println!("  gguf:      {}", gguf.display());
    }

    Ok(())
}
