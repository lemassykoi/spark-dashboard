use super::histogram::{fraction_le, percentile};
use super::prometheus::{parse_prometheus_text, ParsedMetrics};
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
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// The launch flag that turns SGLang's `/metrics` endpoint on. Without it the
/// server answers everything else but serves no metrics — the state the
/// dashboard names on the wire (`metrics_off`) instead of reporting every
/// group as "not reported", which would describe the wrong fact.
pub const METRICS_LAUNCH_FLAG: &str = "--enable-metrics";

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
            warmup: Mutex::new(WarmupTracker::new(warmup_skip_from_env())),
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

/// Aggregate a SGLang metrics body into the single-sample-per-name shape the
/// shared Prometheus parser stores.
///
/// Two properties of SGLang's exposition the vLLM-shaped parser cannot carry:
///
/// * Colon-prefixed metric names (`sglang:num_running_reqs`). The parser
///   normalizes only the `vllm:` prefix — and its replacement here does the
///   same rename so lookups below use underscored names.
/// * The same logical series split across label sets — counters and
///   histograms carry `is_streaming="true"/"false"` variants (streamed and
///   non-streamed requests), and multi-rank launches add per-rank labels.
///   The parser stores one value per name (last line wins), which silently
///   drops the other variants' contributions and lets the reading depend on
///   exposition order. Counters, histogram sums/counts and buckets are summed
///   across their label series; gauges take the first occurrence (they are
///   point-in-state values, and summing e.g. a usage ratio across ranks
///   would be wrong as surely as picking one at random is right only by
///   accident).
fn aggregate_sglang_body(body: &str) -> Option<String> {
    let mut counters: Vec<(String, f64)> = Vec::new(); // ordered, summed
    let mut buckets: HashMap<(String, String), f64> = HashMap::new();
    let mut first_wins: HashMap<String, f64> = HashMap::new();
    let mut histogram_order: Vec<String> = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(space) = line.rfind(' ') else {
            continue;
        };
        let (key, value) = (&line[..space], &line[space + 1..]);
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        let name = match key.find('{') {
            Some(brace) => &key[..brace],
            None => key,
        };
        let name = name.replace("sglang:", "sglang_");

        if let Some(stem) = name.strip_suffix("_bucket").map(str::to_string) {
            let le = match key.rfind("le=\"") {
                Some(at) => match key[at + 4..].find('"') {
                    Some(end) => key[at + 4..at + 4 + end].to_string(),
                    None => continue,
                },
                None => continue,
            };
            if !buckets.contains_key(&(stem.clone(), le.clone()))
                && !histogram_order.iter().any(|h| h == &stem)
            {
                histogram_order.push(stem.clone());
            }
            *buckets.entry((stem, le)).or_insert(0.0) += value;
        } else if name.ends_with("_sum") || name.ends_with("_count") || name.ends_with("_total") {
            match counters.iter_mut().find(|(n, _)| *n == name) {
                Some(slot) => slot.1 += value,
                None => counters.push((name, value)),
            }
        } else {
            first_wins.entry(name).or_insert(value);
        }
    }

    if counters.is_empty() && first_wins.is_empty() {
        return None;
    }

    let mut out = String::new();
    for (name, value) in counters {
        out.push_str(&format!("{name} {value}\n"));
    }
    for (name, value) in first_wins {
        out.push_str(&format!("{name} {value}\n"));
    }
    for stem in histogram_order {
        let mut entries: Vec<(String, f64)> = buckets
            .iter()
            .filter(|((s, _), _)| *s == stem)
            .map(|((_, le), v)| (le.clone(), *v))
            .collect();
        // Numeric le ascending, "+Inf" last — the parser's convention.
        entries.sort_by(|a, b| {
            let ka = a.0.parse::<f64>().unwrap_or(f64::MAX);
            let kb = b.0.parse::<f64>().unwrap_or(f64::MAX);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
        out.push_str(&format!("# TYPE {stem} histogram\n"));
        for (le, count) in entries {
            out.push_str(&format!("{stem}_bucket{{le=\"{le}\"}} {count}\n"));
        }
    }
    Some(out)
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
        let aggregated = status
            .is_success()
            .then(|| aggregate_sglang_body(&body))
            .flatten();
        if let Some(off) =
            metrics_off_signal(status.is_success(), status.as_u16(), aggregated.is_some())
        {
            self.metrics_off.store(off, Ordering::Relaxed);
        }
        let aggregated = aggregated?;

        let raw = parse_prometheus_text(&aggregated)?;
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

/// Map one aggregated, parsed SGLang scrape onto the engine snapshot.
/// Pure: the fetch, the metrics-off latch, the warmup tracker and the rate
/// state stay in `get_metrics`; everything here reads the scrape and nothing
/// else. `warming_up` blanks the delta-derived fields exactly as the vLLM
/// adapter blanks them.
fn engine_metrics(parsed: &ParsedMetrics, rates: RateInputs, warming_up: bool) -> EngineMetrics {
    let gauge = |name: &str| parsed.gauge_sum(name);
    let counter = |name: &str| parsed.counters.get(name).copied();

    let active_requests = gauge("sglang_num_running_reqs").map(|v| v as u64);
    let queued_requests = gauge("sglang_num_queue_reqs").map(|v| v as u64);
    // `full_token_usage` is the fraction of the token pool in use (0-1),
    // SGLang's counterpart of vLLM's kv_cache_usage_perc. Verified live
    // against v0.5.2x on a GB10: 0.381 at a 454k-token pool under load.
    let kv_cache_percent = gauge("sglang_full_token_usage").map(|v| v * 100.0);

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
    // derives from the same quantity; read absolute so it does not dip
    // across the warmup baseline.
    let total_requests = counter("sglang_time_to_first_token_seconds_count").map(|v| v as u64);

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
    let spec_decode_mean_acceptance_length =
        gauge("sglang_spec_accept_length").filter(|v| *v > 0.0);
    let spec_decode_acceptance_rate = spec_decode_mean_acceptance_length
        .and(gauge("sglang_spec_accept_rate"))
        .map(|v| v * 100.0);

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
    /// series is verbatim, including the `is_streaming` splits that the
    /// aggregation exists for.
    const REAL: &str = include_str!("fixtures/sglang_metrics.txt");

    fn parsed_real() -> ParsedMetricsGuard {
        let aggregated = aggregate_sglang_body(REAL).expect("fixture aggregates");
        ParsedMetricsGuard(parse_prometheus_text(&aggregated).expect("fixture parses"))
    }

    struct ParsedMetricsGuard(super::super::prometheus::ParsedMetrics);
    impl std::ops::Deref for ParsedMetricsGuard {
        type Target = super::super::prometheus::ParsedMetrics;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
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

    /// The whole point of aggregating: a counter split across streaming and
    /// non-streaming label series is the sum of both, never whichever line
    /// happened to come last.
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
        let blank = engine_metrics(&parsed_real(), rates, true);
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

        let live = engine_metrics(&parsed_real(), rates, false);
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
        let m = engine_metrics(&parsed_real(), RateInputs::default(), false);
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
        let aggregated = aggregate_sglang_body(REAL).unwrap();
        let parsed = parse_prometheus_text(&aggregated).unwrap();
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
        let aggregated = aggregate_sglang_body(body).expect("aggregates");
        let parsed = parse_prometheus_text(&aggregated).expect("parses");
        let m = engine_metrics(&parsed, RateInputs::default(), false);
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
    fn an_exposition_without_samples_aggregates_to_nothing() {
        // The metrics-off decision rests on this: a body that carries no
        // sample at all (or a 404 body) yields None.
        assert!(aggregate_sglang_body("# only a comment\n").is_none());
        assert!(aggregate_sglang_body("").is_none());
    }

    /// A gauge is never summed — first occurrence wins (a usage ratio summed
    /// across ranks would be nonsense).
    #[test]
    fn gauges_are_first_wins() {
        let body = "sglang:num_running_reqs 2.0\nsglang:num_running_reqs 5.0\n";
        let aggregated = aggregate_sglang_body(body).expect("aggregates");
        let parsed = parse_prometheus_text(&aggregated).expect("parses");
        assert_eq!(parsed.gauge_sum("sglang_num_running_reqs"), Some(2.0));
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
}
