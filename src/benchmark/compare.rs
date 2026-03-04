use super::types::*;

pub fn compare_reports(baseline: &BenchmarkReport, current: &BenchmarkReport) -> ComparisonResult {
    let delta = compute_delta(baseline, current);
    let improvements = identify_improvements(&delta);
    let regressions = identify_regressions(&delta);
    let unchanged = identify_unchanged(&delta);
    let verdict = determine_verdict(&delta, &improvements, &regressions);

    ComparisonResult {
        baseline_id: baseline.metadata.id.clone(),
        current_id: current.metadata.id.clone(),
        baseline_timestamp: baseline.metadata.timestamp,
        current_timestamp: current.metadata.timestamp,
        delta_pct: delta,
        improvements,
        regressions,
        unchanged,
        verdict,
    }
}

fn compute_delta(baseline: &BenchmarkReport, current: &BenchmarkReport) -> DeltaMetrics {
    // For simplicity, compare first device only (or aggregate all)
    let baseline_metrics = baseline
        .devices
        .first()
        .map(|d| &d.overall_metrics)
        .cloned()
        .unwrap_or_default();

    let current_metrics = current
        .devices
        .first()
        .map(|d| &d.overall_metrics)
        .cloned()
        .unwrap_or_default();

    DeltaMetrics {
        sustained_tflops: compute_pct_change(
            baseline_metrics.sustained_tflops,
            current_metrics.sustained_tflops,
        ),
        peak_tflops: compute_pct_change(baseline_metrics.peak_tflops, current_metrics.peak_tflops),
        avg_efficiency: compute_pct_change(
            baseline_metrics.avg_efficiency,
            current_metrics.avg_efficiency,
        ),
        avg_power: compute_pct_change(
            baseline_metrics.avg_power_watts as f64,
            current_metrics.avg_power_watts as f64,
        ),
        peak_temp: compute_pct_change(
            baseline_metrics.peak_temp_celsius as f64,
            current_metrics.peak_temp_celsius as f64,
        ),
        memory_bandwidth: match (
            baseline_metrics.memory_bandwidth_gbps,
            current_metrics.memory_bandwidth_gbps,
        ) {
            (Some(b), Some(c)) => Some(compute_pct_change(b, c)),
            _ => None,
        },
        throttle_events: current_metrics.total_throttle_events as i32
            - baseline_metrics.total_throttle_events as i32,
    }
}

fn compute_pct_change(baseline: f64, current: f64) -> f64 {
    if baseline == 0.0 {
        if current == 0.0 {
            0.0
        } else {
            100.0
        }
    } else {
        ((current - baseline) / baseline) * 100.0
    }
}

fn identify_improvements(delta: &DeltaMetrics) -> Vec<String> {
    let mut improvements = Vec::new();

    if delta.sustained_tflops > 5.0 {
        improvements.push(format!("Sustained compute +{:.1}%", delta.sustained_tflops));
    }

    if delta.peak_tflops > 5.0 {
        improvements.push(format!("Peak compute +{:.1}%", delta.peak_tflops));
    }

    if delta.avg_efficiency > 5.0 {
        improvements.push(format!("Power efficiency +{:.1}%", delta.avg_efficiency));
    }

    if delta.avg_power < -5.0 {
        improvements.push(format!("Power consumption -{:.1}%", delta.avg_power.abs()));
    }

    if delta.peak_temp < -5.0 {
        improvements.push(format!("Peak temperature -{:.1}%", delta.peak_temp.abs()));
    }

    if delta.throttle_events < 0 {
        improvements.push(format!(
            "Throttling events reduced by {}",
            delta.throttle_events.abs()
        ));
    }

    if let Some(bw) = delta.memory_bandwidth {
        if bw > 5.0 {
            improvements.push(format!("Memory bandwidth +{:.1}%", bw));
        }
    }

    improvements
}

fn identify_regressions(delta: &DeltaMetrics) -> Vec<String> {
    let mut regressions = Vec::new();

    if delta.sustained_tflops < -5.0 {
        regressions.push(format!(
            "Sustained compute -{:.1}%",
            delta.sustained_tflops.abs()
        ));
    }

    if delta.peak_tflops < -5.0 {
        regressions.push(format!("Peak compute -{:.1}%", delta.peak_tflops.abs()));
    }

    if delta.avg_efficiency < -5.0 {
        regressions.push(format!(
            "Power efficiency -{:.1}%",
            delta.avg_efficiency.abs()
        ));
    }

    if delta.avg_power > 5.0 {
        regressions.push(format!("Power consumption +{:.1}%", delta.avg_power));
    }

    if delta.peak_temp > 5.0 {
        regressions.push(format!("Peak temperature +{:.1}%", delta.peak_temp));
    }

    if delta.throttle_events > 0 {
        regressions.push(format!(
            "Throttling events increased by {}",
            delta.throttle_events
        ));
    }

    if let Some(bw) = delta.memory_bandwidth {
        if bw < -5.0 {
            regressions.push(format!("Memory bandwidth -{:.1}%", bw.abs()));
        }
    }

    regressions
}

fn identify_unchanged(delta: &DeltaMetrics) -> Vec<String> {
    let mut unchanged = Vec::new();

    if delta.sustained_tflops.abs() <= 5.0 {
        unchanged.push("Sustained compute".to_string());
    }

    if delta.peak_tflops.abs() <= 5.0 {
        unchanged.push("Peak compute".to_string());
    }

    if delta.avg_efficiency.abs() <= 5.0 {
        unchanged.push("Power efficiency".to_string());
    }

    if delta.avg_power.abs() <= 5.0 {
        unchanged.push("Power consumption".to_string());
    }

    if delta.peak_temp.abs() <= 5.0 {
        unchanged.push("Temperature".to_string());
    }

    unchanged
}

fn determine_verdict(
    delta: &DeltaMetrics,
    improvements: &[String],
    regressions: &[String],
) -> ComparisonVerdict {
    let improvement_count = improvements.len() as i32;
    let regression_count = regressions.len() as i32;
    let net_score = improvement_count - regression_count;

    // Check for significant changes (>10%)
    let significant_improvement = delta.sustained_tflops > 10.0
        || delta.peak_tflops > 10.0
        || delta.avg_efficiency > 10.0
        || delta.peak_temp < -10.0;

    let significant_regression = delta.sustained_tflops < -10.0
        || delta.peak_tflops < -10.0
        || delta.avg_efficiency < -10.0
        || delta.peak_temp > 10.0
        || delta.throttle_events > 0;

    if significant_improvement && !significant_regression && net_score > 0 {
        ComparisonVerdict::SignificantImprovement
    } else if significant_regression && !significant_improvement && net_score < 0 {
        ComparisonVerdict::SignificantRegression
    } else if net_score > 0 {
        ComparisonVerdict::Improvement
    } else if net_score < 0 {
        ComparisonVerdict::Regression
    } else {
        ComparisonVerdict::Neutral
    }
}
