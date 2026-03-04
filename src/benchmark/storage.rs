use super::types::{
    BenchmarkIndex, BenchmarkIndexEntry, BenchmarkReport, ComparisonResult, KernelType,
};
use anyhow::Result;
use chrono::Local;
use std::path::{Path, PathBuf};

pub struct BenchmarkStorage {
    base_dir: PathBuf,
}

impl BenchmarkStorage {
    pub fn new() -> Result<Self> {
        let base_dir = Self::benchmark_dir()?;
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    fn benchmark_dir() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".config"))
            })
            .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?;

        Ok(config_dir.join("gptop").join("benchmarks"))
    }

    fn index_path(&self) -> PathBuf {
        self.base_dir.join("index.json")
    }

    fn report_path(&self, id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.json", id))
    }

    fn generate_id() -> String {
        let now = Local::now();
        now.format("%Y-%m-%d-%H%M%S").to_string()
    }

    pub fn save(&self, report: &mut BenchmarkReport) -> Result<()> {
        // Generate ID if not set
        if report.metadata.id.is_empty() {
            report.metadata.id = Self::generate_id();
        }

        let id = &report.metadata.id;
        let path = self.report_path(id);

        // Serialize and save
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(&path, json)?;

        // Update index
        self.add_to_index(BenchmarkIndexEntry {
            id: id.clone(),
            timestamp: report.metadata.timestamp,
            hostname: report.metadata.hostname.clone(),
            benchmark_type: report.metadata.benchmark_type.clone(),
            duration_seconds: report.metadata.duration_seconds,
            gpu_name: report.system_info.gpu_name.clone(),
            grade: report.summary.grade.clone(),
            score: report.summary.score,
        })?;

        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<BenchmarkReport> {
        let path = self.report_path(id);
        let content = std::fs::read_to_string(&path)?;
        let report: BenchmarkReport = serde_json::from_str(&content)?;
        Ok(report)
    }

    pub fn load_index(&self) -> Result<BenchmarkIndex> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(BenchmarkIndex::default());
        }
        let content = std::fs::read_to_string(&path)?;
        let index: BenchmarkIndex = serde_json::from_str(&content)?;
        Ok(index)
    }

    fn add_to_index(&self, entry: BenchmarkIndexEntry) -> Result<()> {
        let mut index = self.load_index()?;

        // Remove any existing entry with same ID (shouldn't happen with timestamps)
        index.entries.retain(|e| e.id != entry.id);

        // Add new entry at the beginning
        index.entries.insert(0, entry);

        // Keep only last 100 entries to prevent index bloat
        if index.entries.len() > 100 {
            index.entries.truncate(100);
            // Optionally: clean up old report files here
        }

        // Save index
        let json = serde_json::to_string_pretty(&index)?;
        std::fs::write(self.index_path(), json)?;

        Ok(())
    }

    pub fn list(&self, limit: Option<usize>) -> Result<Vec<BenchmarkIndexEntry>> {
        let index = self.load_index()?;
        let entries = if let Some(limit) = limit {
            index.entries.into_iter().take(limit).collect()
        } else {
            index.entries
        };
        Ok(entries)
    }

    pub fn find_by_prefix(&self, prefix: &str) -> Result<Option<String>> {
        let index = self.load_index()?;

        // Try exact match first
        for entry in &index.entries {
            if entry.id == prefix {
                return Ok(Some(entry.id.clone()));
            }
        }

        // Try prefix match
        let matches: Vec<_> = index
            .entries
            .iter()
            .filter(|e| e.id.starts_with(prefix))
            .collect();

        if matches.len() == 1 {
            Ok(Some(matches[0].id.clone()))
        } else if matches.is_empty() {
            Ok(None)
        } else {
            Err(anyhow::anyhow!(
                "Multiple benchmarks match prefix '{}': {:?}",
                prefix,
                matches.iter().map(|m| &m.id).collect::<Vec<_>>()
            ))
        }
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.report_path(id);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }

        // Update index
        let mut index = self.load_index()?;
        index.entries.retain(|e| e.id != id);
        let json = serde_json::to_string_pretty(&index)?;
        std::fs::write(self.index_path(), json)?;

        Ok(())
    }

    pub fn export_comparison(&self, comparison: &ComparisonResult, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(comparison)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_id_format() {
        let id = BenchmarkStorage::generate_id();
        // Should be in format: YYYY-MM-DD-HHMMSS
        assert_eq!(id.len(), 17);
        assert!(id.contains('-'));
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let storage = BenchmarkStorage {
            base_dir: temp_dir.path().to_path_buf(),
        };

        let report = create_test_report();
        let mut report_with_id = report.clone();

        storage.save(&mut report_with_id).unwrap();
        let loaded = storage.load(&report_with_id.metadata.id).unwrap();

        assert_eq!(loaded.metadata.hostname, report.metadata.hostname);
        assert_eq!(loaded.summary.score, report.summary.score);
    }

    fn create_test_report() -> BenchmarkReport {
        BenchmarkReport {
            metadata: BenchmarkMetadata {
                id: String::new(),
                timestamp: Local::now(),
                hostname: "test-host".to_string(),
                gptop_version: "0.2.0".to_string(),
                duration_seconds: 60,
                benchmark_type: BenchmarkType::Comprehensive,
                kernels_run: vec![KernelType::MatMulSmall],
            },
            system_info: SystemInfo {
                os: "macos".to_string(),
                gpu_vendor: "Apple".to_string(),
                gpu_name: "Apple M3 Max GPU".to_string(),
                gpu_cores: Some(40),
                total_memory: 68719476736,
            },
            devices: vec![],
            summary: BenchmarkSummary {
                grade: Grade::B,
                score: 85.5,
                strengths: vec!["Good compute performance".to_string()],
                weaknesses: vec![],
                recommendations: vec![],
            },
        }
    }
}
