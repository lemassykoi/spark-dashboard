use super::histogram::{fraction_le, percentile};
use super::prometheus::{parse_prometheus_text, GaugeSeries, ParsedMetrics};
use super::vllm::{
    classify_models_error_status, fetch_hf_model_info, normalize_model_id, resolve_model_name,
    ModelsEndpointReply, OpenAIModelsResponse,
};
use super::warmup::WarmupTracker;
use super::{
    EngineAdapter, EngineMetrics, EngineStatus, EngineType, HistogramBucket, LatencyPercentiles,
    MetricGroup, ModelInfo, ModelMetadataError, ModelResolution, E2E_SLO_MS, ITL_SLO_MS,
    TTFT_SLO_MS,
};
use async_trait::async_trait;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// The launch flag that turns SGLang's `/metrics` endpoint on. Without it the
/// server answers everything else but serves no metrics — the state the
/// dashboard names on the wire (`metrics_off`) instead of reporting every
/// group as "not reported", which would describe the wrong fact.
pub const METRICS_LAUNCH_FLAG: &str = "--enable-metrics";

/// The request counter warmup progress is measured on: every SGLang request
/// adds one observation to the time-to-first-token histogram, as on vLLM.
const WARMUP_TRIGGER_KEY: &str = "sglang_time_to_first_token_seconds_count";

/// Read the warmup-skip threshold from the environment, shared convention
/// with the vLLM adapter (`SPARK_WARMUP_SKIP_REQUESTS`).
fn warmup_skip_from_env() -> u64 {
    std::env::var("SPARK_WARMUP_SKIP_REQUESTS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1)
}

pub struct SglangAdapter {
    client: reqwest::Client,
    endpoint: String,
    /// Optional bearer token. SGLang gates `/v1/models` (and everything
    /// except `/health`, `/ready`, `/metrics`) when launched with an API key;
    /// the open endpoints ignore a presented token harmlessly.
    api_key: Option<String>,
    /// Model identity recovered from the launch command line
    /// (`--model-path` / `--served-model-name`), the same fallback role it
    /// plays in the vLLM adapter.
    served_model: Option<String>,
    prev_gen_tokens: Mutex<Option<(f64, Instant)>>,
    prev_prompt_tokens: Mutex<Option<(f64, Instant)>>,
    avg_accum: Mutex<(f64, u64)>,
    avg_prompt_accum: Mutex<(f64, u64)>,
    warmup: Mutex<WarmupTracker>,
    hf_model_cache: Mutex<Option<ModelInfo>>,
    last_hf_error: Mutex<Option<Instant>>,
    /// Latched by each metrics scrape: true when `/metrics` answered but
    /// carried no `sglang` series (the un-instrumented launch), false when
    /// real series were seen. A scrape that could not reach the server at all
    /// leaves the flag untouched — an unreachable engine is "offline", not
    /// "metrics off".
    metrics_off: AtomicBool,
}

impl SglangAdapter {
    pub fn new(
        client: reqwest::Client,
        endpoint: String,
        served_model: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            endpoint,
            api_key,
            served_model,
            prev_gen_tokens: Mutex::new(None),
            prev_prompt_tokens: Mutex::new(None),
            avg_accum: Mutex::new((0.0, 0)),
            avg_prompt_accum: Mutex::new((0.0, 0)),
            warmup: Mutex::new(WarmupTracker::new(
                warmup_skip_from_env(),
                WARMUP_TRIGGER_KEY,
            )),
            hf_model_cache: Mutex::new(None),
            last_hf_error: Mutex::new(None),
            metrics_off: AtomicBool::new(false),
        }
    }

    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => rb.bearer_auth(key),
            None => rb,
        }
    }
}

/// Parse an SGLang scrape with the shared parser, which sums counters and
/// histograms across the label series SGLang splits them into
/// (`is_streaming`, `cache_source`, and one series per rank), and keep each
/// gauge's series for `engine_metrics` to combine per what it measures.
///
/// One SGLang-specific step: under `--enable-priority-scheduling` the
/// scheduler logs each queue gauge twice, its total under `priority=""` and a
/// breakdown under `priority="<n>"` beside it. Only the totals partition the
/// engine, so the breakdown series are dropped before anything sums them.
fn parse_sglang(body: &str) -> Option<ParsedMetrics> {
    let mut parsed = parse_prometheus_text(body)?;
    for series in parsed.gauges.values_mut() {
        series.retain(|g| g.labels.get("priority").is_none_or(|p| p.is_empty()));
    }
    Some(parsed)
}

/// Whether a parsed scrape carries any sample at all.
fn has_samples(parsed: &ParsedMetrics) -> bool {
    !(parsed.counters.is_empty() && parsed.gauges.is_empty() && parsed.histograms.is_empty())
}

/// SGLang's acceptance gauges, combined across the ranks that report them.
///
/// Each data-parallel rank logs its own `spec_accept_length` and
/// `spec_accept_rate`, both over its last decode-log interval, and logs them
/// as `0` until it has verified a draft — as a server without a drafter does
/// for good. A rank that has verified one reports a length of at least 1, so
/// the engine-wide figures are the means over the ranks whose length is
/// positive, each rank's rate read off the series with the same labels.
/// Returns `(length, rate)`; `None` while no rank has verified a draft.
fn spec_acceptance(parsed: &ParsedMetrics) -> Option<(f64, Option<f64>)> {
    let ranks: Vec<&GaugeSeries> = parsed
        .gauges
        .get("sglang_spec_accept_length")
        .into_iter()
        .flatten()
        .filter(|g| g.value > 0.0)
        .collect();
    if ranks.is_empty() {
        return None;
    }
    let length = ranks.iter().map(|g| g.value).sum::<f64>() / ranks.len() as f64;
    let rates: Vec<f64> = parsed
        .gauges
        .get("sglang_spec_accept_rate")
        .into_iter()
        .flatten()
        .filter(|g| ranks.iter().any(|r| r.labels == g.labels))
        .map(|g| g.value)
        .collect();
    let rate = (!rates.is_empty()).then(|| rates.iter().sum::<f64>() / rates.len() as f64);
    Some((length, rate))
}

#[async_trait]
impl EngineAdapter for SglangAdapter {
    fn engine_type(&self) -> EngineType {
        EngineType::Sglang
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// SGLang reports every group its endpoint can fill; fields a given
    /// version or launch does not expose (TPOT without its own histogram,
    /// swap counters, per-iteration batch size, spec counters) stay `None`
    /// beside them — a field the engine never emits is "not reported" at
    /// field level, which the panels already render as absence, exactly as
    /// vLLM's optional sections do.
    fn metric_groups(&self) -> BTreeSet<MetricGroup> {
        MetricGroup::all()
    }

    fn metrics_launch_flag(&self) -> Option<&'static str> {
        Some(METRICS_LAUNCH_FLAG)
    }

    fn metrics_off(&self) -> bool {
        self.metrics_off.load(Ordering::Relaxed)
    }

    async fn health_check(&self) -> EngineStatus {
        // `/ready` is SGLang's cheap readiness probe. `/health` is not: it
        // runs a one-token generation through the scheduler, so it stalls for
        // as long as the queue ahead of it — a live but busy engine would
        // read as failing health under exactly the load that matters.
        match self
            .auth(
                self.client
                    .get(format!("{}/ready", self.endpoint))
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => EngineStatus::Running,
            Ok(r) => EngineStatus::Error(format!("HTTP {}", r.status())),
            Err(e) => EngineStatus::Error(e.to_string()),
        }
    }

    async fn get_model_info(&self) -> ModelResolution {
        // Same OpenAI-compatible endpoint and the same precedence and error
        // classification the vLLM adapter uses (shared helpers in `vllm.rs`),
        // because SGLang's `/v1/models` answers identically — including the
        // 401 when the engine has an API key the dashboard was not given.
        let reply = match self
            .auth(
                self.client
                    .get(format!("{}/v1/models", self.endpoint))
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<OpenAIModelsResponse>().await {
                    Ok(models) => match models.data.first() {
                        Some(m) => ModelsEndpointReply::Resolved(m.id.clone()),
                        None => ModelsEndpointReply::Empty,
                    },
                    Err(_) => ModelsEndpointReply::Failed(ModelMetadataError::Unavailable),
                }
            }
            Ok(resp) => {
                ModelsEndpointReply::Failed(classify_models_error_status(resp.status().as_u16()))
            }
            Err(_) => ModelsEndpointReply::Failed(ModelMetadataError::Unavailable),
        };

        let (name, metadata_error) = resolve_model_name(&reply, self.served_model.as_deref());
        let Some(name) = name else {
            return ModelResolution {
                model: None,
                metadata_error,
            };
        };

        let name = normalize_model_id(&name);
        let model = match fetch_hf_model_info(
            &self.endpoint,
            &name,
            &self.hf_model_cache,
            &self.last_hf_error,
        )
        .await
        {
            Some(enriched) => enriched,
            None => ModelInfo {
                name,
                parameter_size: None,
                quantization: None,
                precision: None,
                tensor_type: None,
                model_type: None,
                pipeline_tag: None,
            },
        };

        ModelResolution {
            model: Some(model),
            metadata_error,
        }
    }

    async fn get_metrics(&self) -> Option<EngineMetrics> {
        let resp = self
            .auth(
                self.client
                    .get(format!("{}/metrics", self.endpoint))
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
            .ok()?;
        let status = resp.status();
        let body = resp.text().await.ok()?;
        // The latch decision is `metrics_off_signal`, pure below so the ways
        // a scrape can answer are testable without a server. A failure to
        // reach the server never gets here — the `send()` above stops it —
        // because offline is not the same as off.
        let parsed = status
            .is_success()
            .then(|| parse_sglang(&body))
            .flatten()
            .filter(has_samples);
        if let Some(off) =
            metrics_off_signal(status.is_success(), status.as_u16(), parsed.is_some())
        {
            self.metrics_off.store(off, Ordering::Relaxed);
        }
        let raw = parsed?;
        let warmup_out = {
            let mut tracker = self.warmup.lock().await;
            tracker.observe(&raw)
        };
        if warmup_out.just_transitioned {
            *self.prev_gen_tokens.lock().await = None;
            *self.prev_prompt_tokens.lock().await = None;
            *self.avg_accum.lock().await = (0.0, 0);
            *self.avg_prompt_accum.lock().await = (0.0, 0);
            tracing::info!(
                endpoint = %self.endpoint,
                "warmup complete — baseline captured, steady-state metrics now reported"
            );
        }
        let parsed = &warmup_out.adjusted;
        let warming_up = warmup_out.warming_up;

        let counter = |name: &str| parsed.counters.get(name).copied();
        let current_gen = counter("sglang_generation_tokens_total");
        let current_prompt = counter("sglang_prompt_tokens_total");
        let now = Instant::now();
        let tokens_per_sec = {
            let mut prev_lock = self.prev_gen_tokens.lock().await;
            let tps = match (current_gen, prev_lock.as_ref()) {
                (Some(current), Some(&(prev_val, prev_time))) => {
                    let elapsed = now.duration_since(prev_time).as_secs_f64();
                    if elapsed > 0.0 {
                        Some((current - prev_val) / elapsed)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(val) = current_gen {
                *prev_lock = Some((val, now));
            }
            tps
        };

        let prompt_tokens_per_sec = {
            let mut prev_lock = self.prev_prompt_tokens.lock().await;
            let tps = match (current_prompt, prev_lock.as_ref()) {
                (Some(current), Some(&(prev_val, prev_time))) => {
                    let elapsed = now.duration_since(prev_time).as_secs_f64();
                    if elapsed > 0.0 {
                        Some((current - prev_val) / elapsed)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(val) = current_prompt {
                *prev_lock = Some((val, now));
            }
            tps
        };

        let avg_tokens_per_sec = {
            let mut accum = self.avg_accum.lock().await;
            if let Some(tps) = tokens_per_sec {
                if tps > 0.0 {
                    accum.0 += tps;
                    accum.1 += 1;
                }
            }
            if accum.1 > 0 {
                Some(accum.0 / accum.1 as f64)
            } else {
                None
            }
        };
        let avg_prompt_tokens_per_sec = {
            let mut accum = self.avg_prompt_accum.lock().await;
            if let Some(tps) = prompt_tokens_per_sec {
                if tps > 0.0 {
                    accum.0 += tps;
                    accum.1 += 1;
                }
            }
            if accum.1 > 0 {
                Some(accum.0 / accum.1 as f64)
            } else {
                None
            }
        };

        Some(engine_metrics(
            parsed,
            &raw,
            RateInputs {
                tokens_per_sec,
                prompt_tokens_per_sec,
                avg_tokens_per_sec,
                avg_prompt_tokens_per_sec,
            },
            warming_up,
        ))
    }
}

/// The rate fields are the adapter's only stateful derivation (deltas and
/// running means across polls). They arrive as inputs so the rest of the
/// mapping is pure over one scrape and fully testable against a fixture.
#[derive(Debug, Default, Clone, Copy)]
struct RateInputs {
    tokens_per_sec: Option<f64>,
    prompt_tokens_per_sec: Option<f64>,
    avg_tokens_per_sec: Option<f64>,
    avg_prompt_tokens_per_sec: Option<f64>,
}

/// The metrics-off latch decision for one *answered* scrape.
///
/// - answered OK with SGLang series: metrics are on (`Some(false)`);
/// - answered OK without any: nothing to scrape — off (`Some(true)`);
/// - 404: the unflagged launch does not mount the route at all (verified
///   against the v0.5.21 source in review) — off (`Some(true)`);
/// - any other answered failure: transient, and a transient is not "off" —
///   `None` leaves the previous reading alone.
fn metrics_off_signal(success: bool, code: u16, has_series: bool) -> Option<bool> {
    match (success, code, has_series) {
        (true, _, true) => Some(false),
        (true, _, false) => Some(true),
        (false, 404, _) => Some(true),
        (false, _, _) => None,
    }
}

/// Map one parsed SGLang scrape onto the engine snapshot.
/// Pure: the fetch, the metrics-off latch, the warmup tracker and the rate
/// state stay in `get_metrics`; everything here reads the scrape and nothing
/// else. `parsed` is the scrape after the warmup baseline is subtracted,
/// `raw` the same scrape as read, for the lifetime figures that must not
/// restart at the baseline. `warming_up` blanks the delta-derived fields
/// exactly as the vLLM adapter blanks them.
fn engine_metrics(
    parsed: &ParsedMetrics,
    raw: &ParsedMetrics,
    rates: RateInputs,
    warming_up: bool,
) -> EngineMetrics {
    let counter = |name: &str| parsed.counters.get(name).copied();

    // Each data-parallel rank logs its own series of the scheduler gauges
    // (one per rank, from its attention-TP rank 0). The request counts
    // partition the engine's work, so they sum.
    let active_requests = parsed
        .gauge_sum("sglang_num_running_reqs")
        .map(|v| v as u64);
    let queued_requests = parsed.gauge_sum("sglang_num_queue_reqs").map(|v| v as u64);
    // `full_token_usage` is the fraction of the token pool in use (0-1),
    // SGLang's counterpart of vLLM's kv_cache_usage_perc. Verified live
    // against v0.5.2x on a GB10: 0.381 at a 454k-token pool under load.
    // Releases before the hybrid full/SWA split report only `token_usage`,
    // the same fraction for a model with one pool. Each rank's usage is a
    // ratio of its own pool, and data-parallel ranks size their pools alike,
    // so the engine-wide figure is the mean.
    let kv_cache_percent = parsed
        .gauge_mean("sglang_full_token_usage")
        .or_else(|| parsed.gauge_mean("sglang_token_usage"))
        .map(|v| v * 100.0);

    let avg_ms = |sum: Option<f64>, count: Option<f64>| match (sum, count) {
        (Some(s), Some(c)) if c > 0.0 => Some((s / c) * 1000.0),
        _ => None,
    };
    let ttft_ms = avg_ms(
        counter("sglang_time_to_first_token_seconds_sum"),
        counter("sglang_time_to_first_token_seconds_count"),
    );
    let e2e_latency_ms = avg_ms(
        counter("sglang_e2e_request_latency_seconds_sum"),
        counter("sglang_e2e_request_latency_seconds_count"),
    );
    let queue_time_ms = avg_ms(
        counter("sglang_queue_time_seconds_sum"),
        counter("sglang_queue_time_seconds_count"),
    );
    let inter_token_latency_ms = avg_ms(
        counter("sglang_inter_token_latency_seconds_sum"),
        counter("sglang_inter_token_latency_seconds_count"),
    );

    // Request count: TTFT observations == requests that reached first
    // token. The same pass-through lifetime display the vLLM adapter
    // derives from the same quantity; read from `raw` so it does not drop
    // to zero at the warmup baseline.
    let total_requests = raw
        .counters
        .get("sglang_time_to_first_token_seconds_count")
        .map(|&v| v as u64);

    // Per-request prefill throughput: prompt tokens per second of TTFT.
    // Same shape as the vLLM adapter's approximation.
    let per_request_prompt_tps = match (
        counter("sglang_prompt_tokens_total"),
        counter("sglang_time_to_first_token_seconds_sum"),
    ) {
        (Some(p), Some(t)) if t > 0.0 => Some(p / t),
        _ => None,
    };

    // Prefix cache: SGLang counts cached tokens and total prompt tokens
    // (vLLM counts hits and queries). The ratio is the same quantity —
    // fraction of prompt tokens served from the radix cache.
    let prefix_cache_hit_rate = match (
        counter("sglang_cached_tokens_total"),
        counter("sglang_prompt_tokens_total"),
    ) {
        (Some(h), Some(q)) if q > 0.0 => Some((h / q) * 100.0),
        _ => None,
    };
    let prefix_cache_queries_total = counter("sglang_prompt_tokens_total").map(|v| v as u64);

    // Speculative decoding. SGLang exposes acceptance as gauges
    // (`spec_accept_rate` is the accepted/drafted ratio of recent verify
    // rounds, `spec_accept_length` the mean accepted tokens per round)
    // and has no draft/accept token counters — so the cumulative fields
    // and the delta-derived live rate stay `None` and the acceptance
    // figures come straight from the gauges.
    //
    // A server without a drafter logs both gauges as `0` rather than
    // omitting them, so a configured drafter is judged by a nonzero
    // accept length: zero reads identically to "nothing accepted yet",
    // and a panel permanently showing 0% would be the worse of the two
    // readings. Nonzero length is the same configured signal the
    // frontend reads off vLLM's optional counters.
    let acceptance = spec_acceptance(parsed);
    let spec_decode_mean_acceptance_length = acceptance.map(|(length, _)| length);
    let spec_decode_acceptance_rate = acceptance.and_then(|(_, rate)| rate).map(|v| v * 100.0);

    // Percentiles, goodput and shipped buckets from the request-level
    // histograms. Identical math to the vLLM adapter — SGLang's buckets
    // are plain cumulative histograms in seconds.
    let percentiles_ms = |metric: &str| -> Option<LatencyPercentiles> {
        let buckets = parsed.histograms.get(metric)?;
        let to_ms = |q: f64| percentile(buckets, q).map(|s| s * 1000.0);
        let p = LatencyPercentiles {
            p50_ms: to_ms(0.50),
            p95_ms: to_ms(0.95),
            p99_ms: to_ms(0.99),
        };
        if p.p50_ms.is_none() && p.p95_ms.is_none() && p.p99_ms.is_none() {
            None
        } else {
            Some(p)
        }
    };
    let goodput_pct = |metric: &str, slo_ms: f64| -> Option<f64> {
        let buckets = parsed.histograms.get(metric)?;
        fraction_le(buckets, slo_ms / 1000.0).map(|f| f * 100.0)
    };
    let buckets_for = |metric: &str| -> Option<Vec<HistogramBucket>> {
        let raw = parsed.histograms.get(metric)?;
        if raw.is_empty() {
            return None;
        }
        Some(
            raw.iter()
                .map(|&(le, cum)| HistogramBucket {
                    le_seconds: if le.is_finite() { le } else { f64::MAX },
                    cumulative_count: cum,
                })
                .collect(),
        )
    };
    let ttft_percentiles = percentiles_ms("sglang_time_to_first_token_seconds");
    let itl_percentiles = percentiles_ms("sglang_inter_token_latency_seconds");
    let e2e_percentiles = percentiles_ms("sglang_e2e_request_latency_seconds");
    let ttft_goodput_pct = goodput_pct("sglang_time_to_first_token_seconds", TTFT_SLO_MS);
    let itl_goodput_pct = goodput_pct("sglang_inter_token_latency_seconds", ITL_SLO_MS);
    let e2e_goodput_pct = goodput_pct("sglang_e2e_request_latency_seconds", E2E_SLO_MS);
    let ttft_buckets = buckets_for("sglang_time_to_first_token_seconds");
    let itl_buckets = buckets_for("sglang_inter_token_latency_seconds");
    let e2e_buckets = buckets_for("sglang_e2e_request_latency_seconds");

    // Fields SGLang does not report at all (ADR-0004 — absent series are
    // not reported, never proxied by a stand-in): TPOT has no histogram
    // of its own in any probed version, `num_retracted_reqs` is a gauge
    // of how many requests are retracted now rather than the lifetime
    // preemption counter the Requests group names, and there is no
    // per-iteration token counter behind avg_batch_size.
    let blank = warming_up;
    EngineMetrics {
        tokens_per_sec: if blank { None } else { rates.tokens_per_sec },
        avg_tokens_per_sec: if blank {
            None
        } else {
            rates.avg_tokens_per_sec
        },
        per_request_tps: None,
        ttft_ms: if blank { None } else { ttft_ms },
        active_requests,
        queued_requests,
        kv_cache_percent,
        kv_cache_is_estimated: false,
        total_requests,
        e2e_latency_ms: if blank { None } else { e2e_latency_ms },
        prompt_tokens_per_sec: if blank {
            None
        } else {
            rates.prompt_tokens_per_sec
        },
        avg_prompt_tokens_per_sec: if blank {
            None
        } else {
            rates.avg_prompt_tokens_per_sec
        },
        per_request_prompt_tps: if blank { None } else { per_request_prompt_tps },
        swapped_requests: None,
        prefix_cache_hit_rate,
        queue_time_ms: if blank { None } else { queue_time_ms },
        inter_token_latency_ms: if blank { None } else { inter_token_latency_ms },
        preemptions_total: None,
        total_prompt_tokens: counter("sglang_prompt_tokens_total").map(|v| v as u64),
        total_generation_tokens: counter("sglang_generation_tokens_total").map(|v| v as u64),
        prefix_cache_queries_total,
        avg_batch_size: None,
        ttft_percentiles: if blank { None } else { ttft_percentiles },
        itl_percentiles: if blank { None } else { itl_percentiles },
        e2e_percentiles: if blank { None } else { e2e_percentiles },
        ttft_goodput_pct: if blank { None } else { ttft_goodput_pct },
        itl_goodput_pct: if blank { None } else { itl_goodput_pct },
        e2e_goodput_pct: if blank { None } else { e2e_goodput_pct },
        ttft_buckets: if blank { None } else { ttft_buckets },
        itl_buckets: if blank { None } else { itl_buckets },
        e2e_buckets: if blank { None } else { e2e_buckets },
        tpot_ms: None,
        tpot_percentiles: None,
        tpot_goodput_pct: None,
        tpot_buckets: None,
        spec_decode_draft_tokens_total: None,
        spec_decode_accepted_tokens_total: None,
        spec_decode_drafts_total: None,
        spec_decode_acceptance_rate,
        spec_decode_acceptance_rate_live: None,
        spec_decode_mean_acceptance_length,
        warming_up,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scrape of a real SGLang v0.5.2x (qwen4-main nightly) on a GB10,
    /// captured live with `--enable-metrics` and EAGLE/NEXTN speculative
    /// decoding configured. Trimming kept whole metric blocks; every label
    /// series is verbatim, including the `is_streaming` splits the parser
    /// sums.
    const REAL: &str = include_str!("fixtures/sglang_metrics.txt");

    fn parsed_real() -> ParsedMetrics {
        parse_sglang(REAL).expect("fixture parses")
    }

    /// `engine_metrics` for a scrape no baseline has been subtracted from —
    /// one that is its own raw reading.
    fn map_unbaselined(parsed: &ParsedMetrics, rates: RateInputs, warming: bool) -> EngineMetrics {
        engine_metrics(parsed, parsed, rates, warming)
    }

    #[test]
    fn real_scrape_series_arrive_under_underscored_names() {
        let p = parsed_real();
        assert_eq!(p.gauge_sum("sglang_num_running_reqs"), Some(1.0));
        assert_eq!(
            p.gauge_sum("sglang_full_token_usage"),
            Some(0.3813953488372093)
        );
        assert_eq!(p.gauge_sum("sglang_spec_accept_length"), Some(2.8));
    }

    /// A counter split across streaming and non-streaming label series is
    /// the sum of both, never whichever line happened to come last.
    #[test]
    fn label_split_counters_are_summed_not_picked() {
        let p = parsed_real();
        // Fixture: generation 28264.0 across its two label series. The exact
        // split is whatever the capture says; the sum must equal both lines.
        let mut want = 0.0;
        for line in REAL.lines() {
            if let Some(rest) = line.strip_prefix("sglang:generation_tokens_total{") {
                if let Some(space) = line.rfind(' ') {
                    want += line[space + 1..].parse::<f64>().expect("fixture value");
                    let _ = rest;
                }
            }
        }
        assert!(want > 0.0, "fixture must contain the series");
        let got = p.counters.get("sglang_generation_tokens_total").copied();
        assert_eq!(got, Some(want));
    }

    /// Histogram buckets split by label must merge per `le` with summed
    /// counts, or percentiles interpolate against half the observations.
    #[test]
    fn label_split_histogram_buckets_merge_per_le() {
        let p = parsed_real();
        let ttft = p
            .histograms
            .get("sglang_time_to_first_token_seconds")
            .expect("ttft histogram");
        // +Inf bucket == sum of both label series' counts (104 + 104 here;
        // read from the fixture, not hardcoded — this asserts the merge did
        // the arithmetic, and the fixture's shape).
        let mut want = 0.0;
        for line in REAL.lines() {
            if line.starts_with("sglang:time_to_first_token_seconds_count{") {
                let space = line.rfind(' ').expect("sample line");
                want += line[space + 1..].parse::<f64>().expect("fixture value");
            }
        }
        let last = ttft.last().expect("buckets");
        assert!(last.0.is_infinite());
        assert_eq!(last.1, want);
        // Percentiles come out of it: p50 of the merged histogram is a real
        // positive number, not None (the fixture has >100 observations).
        let p50 = percentile(ttft, 0.50).expect("p50 interpolates");
        assert!(p50 > 0.0 && p50 < 10.0, "got {p50}");
    }

    #[test]
    fn warmup_gate_blanks_delta_metrics_only() {
        // The mapping's blanking contract, through the real fixture: while
        // warming, everything a delta or a baseline needs reads None, and
        // every instantaneous reading passes through — the same contract the
        // vLLM adapter holds, checked on identical field names.
        let rates = RateInputs {
            tokens_per_sec: Some(30.0),
            prompt_tokens_per_sec: Some(100.0),
            avg_tokens_per_sec: Some(25.0),
            avg_prompt_tokens_per_sec: Some(90.0),
        };
        let blank = map_unbaselined(&parsed_real(), rates, true);
        assert_eq!(blank.tokens_per_sec, None);
        assert_eq!(blank.avg_tokens_per_sec, None);
        assert_eq!(blank.prompt_tokens_per_sec, None);
        assert_eq!(blank.avg_prompt_tokens_per_sec, None);
        assert_eq!(blank.ttft_ms, None);
        assert_eq!(blank.e2e_latency_ms, None);
        assert_eq!(blank.queue_time_ms, None);
        assert_eq!(blank.inter_token_latency_ms, None);
        assert_eq!(blank.per_request_prompt_tps, None);
        assert!(blank.ttft_percentiles.is_none());
        assert_eq!(blank.ttft_goodput_pct, None);
        assert!(blank.ttft_buckets.is_none());
        assert!(blank.warming_up);

        let live = map_unbaselined(&parsed_real(), rates, false);
        assert_eq!(live.tokens_per_sec, Some(30.0));
        assert_eq!(live.avg_prompt_tokens_per_sec, Some(90.0));
        assert!(live.ttft_ms.unwrap() > 0.0);
        assert!(live.ttft_percentiles.unwrap().p50_ms.unwrap() > 0.0);
        // Instantaneous fields survive warming in both passes.
        assert_eq!(live.active_requests, Some(1));
        assert_eq!(blank.active_requests, Some(1));
        assert_eq!(blank.spec_decode_acceptance_rate, Some(60.0));
        assert_eq!(blank.spec_decode_mean_acceptance_length, Some(2.8));
    }

    #[test]
    fn the_real_fixture_maps_onto_the_fields_sglang_reports() {
        // The whole ADR-0004 contract in one test: every field SGLang has a
        // series for carries the captured value, every field it has none for
        // stays None — no zero, no estimate, no stand-in.
        let m = map_unbaselined(&parsed_real(), RateInputs::default(), false);
        assert_eq!(m.active_requests, Some(1));
        assert_eq!(m.queued_requests, Some(0));
        assert!((m.kv_cache_percent.unwrap() - 38.13953488372093).abs() < 1e-9);
        assert_eq!(m.spec_decode_acceptance_rate, Some(60.0));
        assert_eq!(m.spec_decode_mean_acceptance_length, Some(2.8));
        assert!(m.total_requests.unwrap() > 0);
        assert!(m.ttft_ms.unwrap() > 0.0);
        assert!(m.itl_percentiles.unwrap().p95_ms.unwrap() > 0.0);

        // The prefix ratio and totals against the parsed counters directly,
        // including the label-series sums (streaming + non-streaming).
        let parsed = parsed_real();
        let cached = parsed.counters["sglang_cached_tokens_total"];
        let prompt = parsed.counters["sglang_prompt_tokens_total"];
        assert!((prompt - 2.3651_203e7).abs() < 1.0, "label series summed");
        assert_eq!(m.total_prompt_tokens, Some(prompt as u64));
        assert_eq!(m.total_generation_tokens, Some(166_402));
        assert!(
            (m.prefix_cache_hit_rate.unwrap() - cached / prompt * 100.0).abs() < 1e-9,
            "hit rate is cached-over-prompt of the same summed series"
        );

        // Absent series, enumerated (ADR-0004): each of these renders as a
        // dash, and the moment one gains a value from a stand-in this test
        // is where the reviewer sees it.
        assert_eq!(m.per_request_tps, None);
        assert_eq!(m.swapped_requests, None);
        assert_eq!(m.preemptions_total, None);
        assert_eq!(m.avg_batch_size, None);
        assert_eq!(m.tpot_ms, None);
        assert!(m.tpot_percentiles.is_none());
        assert_eq!(m.tpot_goodput_pct, None);
        assert!(m.tpot_buckets.is_none());
        assert_eq!(m.spec_decode_draft_tokens_total, None);
        assert_eq!(m.spec_decode_accepted_tokens_total, None);
        assert_eq!(m.spec_decode_drafts_total, None);
        assert_eq!(m.spec_decode_acceptance_rate_live, None);
        // Rates are the adapter's stateful derivation; the pure mapping was
        // given none and invents none.
        assert_eq!(m.tokens_per_sec, None);
        assert_eq!(m.prompt_tokens_per_sec, None);
    }

    #[test]
    fn zero_acceptance_gauges_mean_no_drafter_not_zero_percent() {
        // v0.5.21 logs both spec gauges as `0` when no drafter is configured
        // (metrics_reporter sets them explicitly), so `0` must not read as a
        // 0% acceptance rate on a plain engine — it reads as absent.
        let body = "# TYPE sglang:num_running_reqs gauge\n\
                    sglang:num_running_reqs{engine_type=\"unified\"} 4.0\n\
                    # TYPE sglang:spec_accept_rate gauge\n\
                    sglang:spec_accept_rate{engine_type=\"unified\"} 0.0\n\
                    # TYPE sglang:spec_accept_length gauge\n\
                    sglang:spec_accept_length{engine_type=\"unified\"} 0.0\n";
        let parsed = parse_sglang(body).expect("parses");
        let m = map_unbaselined(&parsed, RateInputs::default(), false);
        assert_eq!(m.active_requests, Some(4));
        assert_eq!(m.spec_decode_acceptance_rate, None);
        assert_eq!(m.spec_decode_mean_acceptance_length, None);
    }

    #[test]
    fn metrics_off_latches_from_the_answer_and_only_the_answer() {
        // The three ways a scrape answers, plus the one that must not latch:
        // a transient failure keeps the previous reading.
        assert_eq!(metrics_off_signal(true, 200, true), Some(false));
        assert_eq!(metrics_off_signal(true, 200, false), Some(true));
        assert_eq!(metrics_off_signal(false, 404, false), Some(true));
        assert_eq!(metrics_off_signal(false, 404, true), Some(true));
        assert_eq!(metrics_off_signal(false, 429, false), None);
        assert_eq!(metrics_off_signal(false, 500, false), None);
    }

    #[test]
    fn an_exposition_without_samples_has_none() {
        // The metrics-off decision rests on this: a body that carries no
        // sample at all (or a 404 body) holds nothing to report.
        let comment_only = parse_sglang("# only a comment\n").expect("parses");
        assert!(!has_samples(&comment_only));
        assert!(parse_sglang("").is_none_or(|p| !has_samples(&p)));
    }

    /// With `--dp-size 2` each rank logs its own scheduler gauges. The engine
    /// runs both ranks' requests, so counts sum; token usage is each rank's
    /// own pool, so it averages; the acceptance figures average over the
    /// ranks that have verified drafts, each rate paired with its rank.
    #[test]
    fn data_parallel_ranks_combine_into_one_engine() {
        let body = "\
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{dp_rank=\"0\",model_name=\"m\",tp_rank=\"0\"} 8.0
sglang:num_running_reqs{dp_rank=\"1\",model_name=\"m\",tp_rank=\"1\"} 7.0
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{dp_rank=\"0\",model_name=\"m\",tp_rank=\"0\"} 2.0
sglang:num_queue_reqs{dp_rank=\"1\",model_name=\"m\",tp_rank=\"1\"} 1.0
# TYPE sglang:full_token_usage gauge
sglang:full_token_usage{dp_rank=\"0\",model_name=\"m\",tp_rank=\"0\"} 0.5
sglang:full_token_usage{dp_rank=\"1\",model_name=\"m\",tp_rank=\"1\"} 0.3
# TYPE sglang:spec_accept_length gauge
sglang:spec_accept_length{dp_rank=\"0\",model_name=\"m\",tp_rank=\"0\"} 3.0
sglang:spec_accept_length{dp_rank=\"1\",model_name=\"m\",tp_rank=\"1\"} 2.0
# TYPE sglang:spec_accept_rate gauge
sglang:spec_accept_rate{dp_rank=\"1\",model_name=\"m\",tp_rank=\"1\"} 0.4
sglang:spec_accept_rate{dp_rank=\"0\",model_name=\"m\",tp_rank=\"0\"} 0.8
";
        let parsed = parse_sglang(body).expect("parses");
        let m = map_unbaselined(&parsed, RateInputs::default(), false);
        assert_eq!(m.active_requests, Some(15));
        assert_eq!(m.queued_requests, Some(3));
        let kv = m.kv_cache_percent.expect("kv");
        assert!((kv - 40.0).abs() < 1e-9, "mean of 50% and 30%, got {kv}");
        assert_eq!(m.spec_decode_mean_acceptance_length, Some(2.5));
        let rate = m.spec_decode_acceptance_rate.expect("rate");
        assert!(
            (rate - 60.0).abs() < 1e-9,
            "mean of 80% and 40%, got {rate}"
        );
    }

    /// A rank that has not verified a draft yet logs `0` and stays out of
    /// the acceptance mean, rather than halving the other rank's figures.
    #[test]
    fn a_rank_without_verified_drafts_stays_out_of_the_acceptance_mean() {
        let body = "\
# TYPE sglang:spec_accept_length gauge
sglang:spec_accept_length{dp_rank=\"0\"} 3.0
sglang:spec_accept_length{dp_rank=\"1\"} 0.0
# TYPE sglang:spec_accept_rate gauge
sglang:spec_accept_rate{dp_rank=\"0\"} 0.8
sglang:spec_accept_rate{dp_rank=\"1\"} 0.0
";
        let m = map_unbaselined(
            &parse_sglang(body).expect("parses"),
            RateInputs::default(),
            false,
        );
        assert_eq!(m.spec_decode_mean_acceptance_length, Some(3.0));
        assert_eq!(m.spec_decode_acceptance_rate, Some(80.0));
    }

    /// Under priority scheduling a queue gauge carries its total under
    /// `priority=""` and a per-priority breakdown beside it; summing all of
    /// them would count every request twice.
    #[test]
    fn priority_breakdowns_do_not_count_twice() {
        let body = "\
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name=\"m\",priority=\"\"} 5.0
sglang:num_running_reqs{model_name=\"m\",priority=\"0\"} 3.0
sglang:num_running_reqs{model_name=\"m\",priority=\"1\"} 2.0
";
        let m = map_unbaselined(
            &parse_sglang(body).expect("parses"),
            RateInputs::default(),
            false,
        );
        assert_eq!(m.active_requests, Some(5));
    }

    /// Releases before the hybrid full/SWA split report the pool's usage as
    /// `token_usage` only; the KV tile reads it rather than going blank.
    #[test]
    fn kv_usage_falls_back_to_token_usage() {
        let body = "\
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name=\"m\"} 0.25
";
        let m = map_unbaselined(
            &parse_sglang(body).expect("parses"),
            RateInputs::default(),
            false,
        );
        assert_eq!(m.kv_cache_percent, Some(25.0));
    }

    /// The bucket bound is the `le` label, read off the parsed label set —
    /// not the last `le="` in the line, which a label like `role="decode"`
    /// after it would turn into a bound of "decode".
    #[test]
    fn bucket_bounds_come_from_the_le_label_wherever_it_sits() {
        let body = "\
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{le=\"0.5\",role=\"decode\"} 3
sglang:time_to_first_token_seconds_bucket{le=\"+Inf\",role=\"decode\"} 4
";
        let parsed = parse_sglang(body).expect("parses");
        let ttft = parsed
            .histograms
            .get("sglang_time_to_first_token_seconds")
            .expect("ttft histogram");
        assert_eq!(ttft.len(), 2);
        assert_eq!(ttft[0], (0.5, 3.0));
        assert!(ttft[1].0.is_infinite());
        assert_eq!(ttft[1].1, 4.0);
    }

    /// SGLang says "metrics off" with a flag name, not with an empty group
    /// list — the two states render differently in the UI (ADR-0004).
    #[test]
    fn metrics_off_names_its_flag_beyond_the_group_declaration() {
        let adapter = SglangAdapter::new(
            reqwest::Client::new(),
            "http://127.0.0.1:30000".into(),
            None,
            None,
        );
        assert_eq!(adapter.metric_groups(), MetricGroup::all());
        assert!(!adapter.metrics_off());
        assert_eq!(adapter.metrics_launch_flag(), Some("--enable-metrics"));
    }

    /// A body whose request and token counters stand at `requests` and
    /// `tokens` — the shape a live engine's scrape grows through.
    fn growing_scrape(requests: u64, tokens: u64) -> String {
        format!(
            "# TYPE sglang:time_to_first_token_seconds histogram\n\
             sglang:time_to_first_token_seconds_bucket{{le=\"0.5\",is_streaming=\"true\"}} {requests}\n\
             sglang:time_to_first_token_seconds_bucket{{le=\"+Inf\",is_streaming=\"true\"}} {requests}\n\
             sglang:time_to_first_token_seconds_sum{{is_streaming=\"true\"}} {sum}\n\
             sglang:time_to_first_token_seconds_count{{is_streaming=\"true\"}} {requests}\n\
             # TYPE sglang:generation_tokens_total counter\n\
             sglang:generation_tokens_total{{is_streaming=\"true\"}} {tokens}\n\
             # TYPE sglang:num_running_reqs gauge\n\
             sglang:num_running_reqs{{tp_rank=\"0\"}} 1.0\n",
            sum = requests as f64 * 0.1,
        )
    }

    /// Warmup ends on SGLang's own request counter. It was measured on
    /// vLLM's, which an SGLang scrape never carries, so at the default
    /// `SPARK_WARMUP_SKIP_REQUESTS=1` the engine stayed warming forever and
    /// every rate, mean, percentile and goodput figure stayed blank. And the
    /// lifetime request count stays absolute across the baseline.
    #[tokio::test]
    async fn warmup_completes_as_sglang_serves_requests() {
        use super::super::test_server::TestServer;
        use std::sync::atomic::AtomicU64;
        use std::sync::Arc;

        let polls = Arc::new(AtomicU64::new(0));
        let served = polls.clone();
        let server = TestServer::start(move |_| {
            let n = served.fetch_add(1, Ordering::SeqCst);
            (200, growing_scrape(5 + n, 1000 + 100 * n))
        })
        .await;
        let adapter = SglangAdapter::new(reqwest::Client::new(), server.url.clone(), None, None);

        // First scrape: 5 requests already served — the attach cursor.
        let first = adapter.get_metrics().await.expect("metrics");
        assert!(first.warming_up);
        assert_eq!(first.total_requests, Some(5));
        assert_eq!(first.tokens_per_sec, None);

        // One more request: warmup is over, and the lifetime count does not
        // restart at the baseline.
        let second = adapter.get_metrics().await.expect("metrics");
        assert!(!second.warming_up, "warmup ends after one request");
        assert_eq!(second.total_requests, Some(6));

        // Steady state: the delta-derived figures fill in.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let third = adapter.get_metrics().await.expect("metrics");
        assert!(!third.warming_up);
        assert_eq!(third.total_requests, Some(7));
        assert!(third.tokens_per_sec.is_some_and(|tps| tps > 0.0));
        assert!(third.ttft_ms.is_some_and(|ms| ms > 0.0));
    }
}
