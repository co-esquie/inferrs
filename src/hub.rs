//! HuggingFace Hub model downloading.

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Files needed to load a model.
pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    pub weight_paths: Vec<PathBuf>,
}

/// Download model files from HuggingFace Hub.
pub fn download_model(model_id: &str, revision: &str) -> Result<ModelFiles> {
    tracing::info!("Downloading model {} (revision: {})", model_id, revision);

    let api = Api::new().context("Failed to create HuggingFace API client")?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    // Download config.json
    let config_path = repo
        .get("config.json")
        .context("Failed to download config.json")?;
    tracing::info!("Downloaded config.json");

    // Download tokenizer.json
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    tracing::info!("Downloaded tokenizer.json");

    // Try to download tokenizer_config.json (optional)
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
    if tokenizer_config_path.is_some() {
        tracing::info!("Downloaded tokenizer_config.json");
    }

    // Download safetensors weight files
    let weight_paths = download_safetensors(&repo)?;

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths,
    })
}

fn download_safetensors(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try model.safetensors first (single file models)
    if let Ok(path) = repo.get("model.safetensors") {
        tracing::info!("Downloaded model.safetensors");
        return Ok(vec![path]);
    }

    // Try model.safetensors.index.json for sharded models
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("No model.safetensors or model.safetensors.index.json found")?;

    let index_content =
        std::fs::read_to_string(&index_path).context("Failed to read safetensors index")?;
    let index: serde_json::Value =
        serde_json::from_str(&index_content).context("Failed to parse safetensors index")?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("No weight_map in safetensors index")?;

    // Collect unique filenames
    let mut filenames: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    filenames.sort();
    filenames.dedup();

    let mut paths = Vec::new();
    for filename in &filenames {
        let path = repo
            .get(filename)
            .with_context(|| format!("Failed to download {filename}"))?;
        tracing::info!("Downloaded {}", filename);
        paths.push(path);
    }

    Ok(paths)
}
