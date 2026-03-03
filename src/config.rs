use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_interval")]
    pub update_interval_ms: u64,
    #[serde(default)]
    pub accent_color_idx: usize,
    #[serde(default)]
    pub sort_column: usize,
    #[serde(default = "default_true")]
    pub sort_ascending: bool,
}

fn default_interval() -> u64 {
    1000
}

fn default_true() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            update_interval_ms: 1000,
            accent_color_idx: 0,
            sort_column: 0,
            sort_ascending: true,
        }
    }
}

impl Config {
    fn config_path() -> Option<PathBuf> {
        dirs_fallback().map(|d| d.join("gptop.json"))
    }

    pub fn load() -> Self {
        Self::config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        if let Some(path) = Self::config_path() {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let json = serde_json::to_string_pretty(self)?;
            std::fs::write(path, json)?;
        }
        Ok(())
    }
}

fn dirs_fallback() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".config"))
}
