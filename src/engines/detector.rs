use super::{DeploymentMode, EngineType};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::time::Duration;

/// A candidate engine discovered by the detection layers.
#[derive(Clone, Debug)]
pub struct DetectedEngine {
    pub engine_type: EngineType,
    pub endpoint: String,
    pub deployment_mode: DeploymentMode,
    /// Model identity recovered from the launch command line (e.g. `vllm serve
    /// unsloth/Llama-3.2-1B-Instruct` or `--model Qwen/Qwen2.5-3B`). Used as a
    /// fallback when `/v1/models` returns a bare slug without the HF-style
    /// `Provider/` prefix that the operator actually started the server with.
    pub served_model: Option<String>,
    /// Host-namespace PIDs belonging to this engine: the detected process and
    /// its descendants (native scan), or every process `docker top` reports
    /// for the container. Matched against NVML's per-device compute-process
    /// lists to attribute the engine to the GPU(s) it runs on.
    pub pids: Vec<u32>,
    /// Docker container id (full) when discovered via the Docker scan layer;
    /// `None` for natively-detected engines. Used by the log viewer to stream
    /// the same container the dashboard is showing metrics for, rather than
    /// re-scanning and potentially picking a different one.
    pub container_id: Option<String>,
}

/// The port vLLM serves on when it is not told otherwise. Used wherever a
/// candidate is found but its port cannot be read off the command line.
const VLLM_DEFAULT_PORT: u16 = 8000;

/// SGLang's default HTTP port (`--port` unset), same role as above.
const SGLANG_DEFAULT_PORT: u16 = 30000;

/// The port an engine type serves on when it is not told otherwise.
fn default_port(engine_type: &EngineType) -> u16 {
    match engine_type {
        EngineType::Vllm => VLLM_DEFAULT_PORT,
        EngineType::Sglang => SGLANG_DEFAULT_PORT,
    }
}

// ---------------------------------------------------------------------------
// Public detection entry point
// ---------------------------------------------------------------------------

/// Run three-layer detection: process scan, Docker query, API health probe.
///
/// Returns only engines whose API health probe succeeded (ready to serve).
pub async fn detect_engines(
    sys: &sysinfo::System,
    client: &reqwest::Client,
) -> Vec<DetectedEngine> {
    // Layer 1: process scan
    let mut candidates = detect_by_process(sys);

    // Layer 2: Docker scan (Linux-only)
    let docker_candidates = detect_docker_engines().await;
    merge_docker_candidates(&mut candidates, docker_candidates);

    // Layer 3: API health probe -- verify each candidate is actually reachable
    let mut verified = Vec::new();
    for candidate in candidates {
        if probe_engine(client, &candidate).await {
            verified.push(candidate);
        }
    }

    verified
}

/// Merge Docker results into the process-scan candidates. Key by
/// (engine_type, endpoint) so multiple instances of the same engine on
/// different ports are preserved as distinct entries. When a Docker candidate
/// matches an existing process-scan entry by exact endpoint, upgrade its
/// deployment_mode to Docker (Docker detection is authoritative for container
/// metadata) and union the PID sets — `docker top` and the host process scan
/// may each see processes the other missed.
fn merge_docker_candidates(
    candidates: &mut Vec<DetectedEngine>,
    docker_candidates: Vec<DetectedEngine>,
) {
    let mut seen: HashSet<(EngineType, String)> = candidates
        .iter()
        .map(|c| (c.engine_type.clone(), c.endpoint.clone()))
        .collect();

    for dc in docker_candidates {
        let key = (dc.engine_type.clone(), dc.endpoint.clone());
        if seen.contains(&key) {
            if let Some(existing) = candidates
                .iter_mut()
                .find(|c| c.engine_type == dc.engine_type && c.endpoint == dc.endpoint)
            {
                existing.deployment_mode = dc.deployment_mode;
                // Prefer a non-None served_model hint: Docker discovery often
                // sees the `--model` arg the operator launched with, while
                // native process scanning may not (e.g. when vLLM is launched
                // inside a container with host networking).
                if existing.served_model.is_none() && dc.served_model.is_some() {
                    existing.served_model = dc.served_model.clone();
                }
                for pid in &dc.pids {
                    if !existing.pids.contains(pid) {
                        existing.pids.push(*pid);
                    }
                }
                existing.pids.sort_unstable();
                // Docker discovery is authoritative for container identity.
                if existing.container_id.is_none() && dc.container_id.is_some() {
                    existing.container_id = dc.container_id.clone();
                }
            }
        } else {
            seen.insert(key);
            candidates.push(dc);
        }
    }
}

/// Which engine's server an argument names, if any. The launch forms:
///   * vLLM: the `vllm` CLI (`vllm serve ...`, also by path), and
///     `python -m vllm.entrypoints.openai.api_server ...`;
///   * SGLang: the `sglang` CLI (`sglang serve ...`, also by path), and
///     `python -m sglang.launch_server ...` (the documented one) or a
///     `sglang.srt` module.
///
/// Deliberately token-precise: a proxy named "sglang-sidecar" runs nothing
/// until a command line proves otherwise, and an argument that merely
/// contains an engine's name (a log file, a config file) is not the server
/// itself.
fn server_named_by(arg: &str) -> Option<EngineType> {
    if arg == "vllm" || arg.ends_with("/vllm") || arg.contains("vllm.entrypoints") {
        Some(EngineType::Vllm)
    } else if arg == "sglang"
        || arg.ends_with("/sglang")
        || arg.contains("sglang.launch_server")
        || arg.contains("sglang.srt")
    {
        Some(EngineType::Sglang)
    } else {
        None
    }
}

/// Which engine a command line runs, if any: the first argument that names
/// an engine's server decides. The program comes before its arguments, so a
/// path or flag value that happens to name the other engine —
/// `python3 -m sglang.launch_server --download-dir /data/vllm` — cannot turn
/// the process into that engine too. A process is at most one engine.
fn engine_of_command<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<EngineType> {
    args.into_iter().find_map(server_named_by)
}

/// The argv form of [`engine_of_command`], for the host process scan: each
/// argument is intact, so a model path with spaces cannot split into a false
/// match.
fn engine_of_process(cmd: &[OsString]) -> Option<EngineType> {
    engine_of_command(cmd.iter().filter_map(|arg| arg.to_str()))
}

/// Whether a whole command line (a `docker top` row) runs the given engine's
/// server rather than something that merely mentions it — the same evidence
/// a host process is judged by: the binary being run, not a name somebody
/// chose.
///
/// Like everything the container classification uses, only reachable from the
/// Linux Docker path in normal builds, but the unit tests exercise it on every
/// platform — hence the `test` cfg.
#[cfg(any(target_os = "linux", test))]
fn runs_engine(command: &str, engine_type: &EngineType) -> bool {
    engine_of_command(command.split_whitespace()).as_ref() == Some(engine_type)
}

/// The served model named on a pre-joined command line, read the way the
/// engine type's own launcher spells it.
#[cfg(any(target_os = "linux", test))]
fn parse_model_for(engine_type: &EngineType, cmd: &str) -> Option<String> {
    match engine_type {
        EngineType::Vllm => parse_model_from_command_str(cmd),
        EngineType::Sglang => parse_sglang_model_from_command_str(cmd),
    }
}

/// What a container's declared name, image and command say about which
/// engine (if any) it runs, and how far to trust that.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, PartialEq)]
struct ContainerClass {
    engine_type: EngineType,
    /// The image or command named this engine type — trusted as-is. A
    /// name-only match is not: an operator can name anything, so such a
    /// container must additionally be seen *running* the engine process.
    strong_signal: bool,
}

/// Rank the signals by how much they say:
///   1. the **image** is what was pulled to run here — when it names exactly
///      one engine, that engine. An `lmsysorg/sglang` image whose command
///      mentions vLLM (an env var, a model path) is SGLang.
///   2. the **command** is what the entrypoint runs — its launch form, read
///      like a host process's (the first argument naming a server decides),
///      so `python3 -m sglang.launch_server --model-path /models/x-vllm-awq`
///      is SGLang. Failing a launch form, a command that mentions exactly
///      one engine anywhere (a shell wrapper's quoted `"vllm serve ..."`)
///      still names it.
///   3. the **name** is what somebody typed — a weak signal the container's
///      processes must confirm.
///
/// What stays tied — both engines named and no launch form to decide — goes
/// to vLLM: no one builds a "vllm-sglang" image on purpose, and the
/// mismatched container is more plausibly a vLLM deployment with sglang in a
/// sidecar name than the reverse.
///
/// Only reachable from the Linux Docker path in normal builds, but the unit
/// tests exercise it on every platform, hence the `test` cfg.
#[cfg(any(target_os = "linux", test))]
fn classify_container(names: &[String], image: &str, command: &str) -> Option<ContainerClass> {
    let mentions = |text: &str| (text.contains("vllm"), text.contains("sglang"));
    let class = |engine_type, strong_signal| {
        Some(ContainerClass {
            engine_type,
            strong_signal,
        })
    };

    match mentions(image) {
        (true, false) => return class(EngineType::Vllm, true),
        (false, true) => return class(EngineType::Sglang, true),
        _ => {}
    }
    if let Some(engine_type) = engine_of_command(command.split_whitespace()) {
        return class(engine_type, true);
    }
    match mentions(command) {
        (false, true) => return class(EngineType::Sglang, true),
        (true, _) => return class(EngineType::Vllm, true),
        (false, false) => {}
    }
    // An image naming both engines, with a command naming neither: the tie.
    if image.contains("vllm") {
        return class(EngineType::Vllm, true);
    }

    let named = |needle: &str| names.iter().any(|n| n.to_lowercase().contains(needle));
    if named("vllm") {
        class(EngineType::Vllm, false)
    } else if named("sglang") {
        class(EngineType::Sglang, false)
    } else {
        None
    }
}

/// The endpoint to probe a container's vLLM on, and whether the port had to be
/// assumed.
///
/// No port anywhere — no published binding, no `--port` in the command, none in
/// the container's process list — means vLLM is on its own default. That is the
/// normal shape of a host-networked container: it publishes nothing, because it
/// is already on the host's ports.
///
/// Assuming is safe and dropping the container is not. The health probe rejects
/// a candidate whose `/health` does not answer, so a wrong assumption costs one
/// request and corrects itself — while a container dropped for having no
/// readable port is invisible to the dashboard for good, engine, metrics and
/// logs alike.
fn docker_endpoint(port: Option<&str>, default_port: u16) -> (String, bool) {
    match port {
        Some(p) => (format!("http://localhost:{}", p), false),
        None => (format!("http://localhost:{}", default_port), true),
    }
}

// ---------------------------------------------------------------------------
// Layer 1: Process scan
// ---------------------------------------------------------------------------

fn detect_by_process(sys: &sysinfo::System) -> Vec<DetectedEngine> {
    // Every process is judged by its command line alone — not by its name:
    // SGLang's child processes rename themselves (`setproctitle` →
    // `sglang::scheduler`, `sglang::detokenizer`) and carry neither the
    // launch flags nor the serving `--port`, while the server itself is often
    // named `python3`. Sorted by PID so the candidates come out in a stable
    // order.
    let mut processes: Vec<(u32, &[OsString])> = sys
        .processes()
        .iter()
        .map(|(pid, p)| (pid.as_u32(), p.cmd()))
        .collect();
    processes.sort_unstable_by_key(|&(pid, _)| pid);
    let mut detected = engines_from_processes(processes);

    // vLLM's API-server process rarely holds the CUDA context itself — the
    // engine-core / tensor-parallel worker child processes do — so NVML's
    // compute-process list must be matched against the whole process tree,
    // not just the detected root PID.
    if !detected.is_empty() {
        let processes: Vec<(u32, Option<u32>)> = sys
            .processes()
            .iter()
            .map(|(pid, proc_)| (pid.as_u32(), proc_.parent().map(|pp| pp.as_u32())))
            .collect();
        for d in &mut detected {
            d.pids = expand_pid_tree(&d.pids, &processes);
        }
    }

    detected
}

/// One candidate per engine endpoint from a host's `(pid, argv)` list.
///
/// Each process is at most one engine — the one its command line launches
/// (`engine_of_process`) — so a process is never polled as both: an SGLang
/// server with `/data/vllm` among its arguments is not also a vLLM candidate
/// whose `/health` probe would run a generation on SGLang every second. One
/// candidate per distinct endpoint, too: multi-instance native setups (three
/// `vllm serve --port 8000/8001/8002` processes) surface each port as its
/// own engine rather than collapsing to the first.
fn engines_from_processes<'a>(
    processes: impl IntoIterator<Item = (u32, &'a [OsString])>,
) -> Vec<DetectedEngine> {
    let mut detected: Vec<DetectedEngine> = Vec::new();
    let mut seen_endpoints: HashMap<(EngineType, String), usize> = HashMap::new();
    for (pid, cmd) in processes {
        let Some(engine_type) = engine_of_process(cmd) else {
            continue;
        };
        let default_endpoint = format!("http://localhost:{}", default_port(&engine_type));
        let endpoint = parse_endpoint_from_args(cmd, &default_endpoint).unwrap_or(default_endpoint);
        let served_model = match engine_type {
            EngineType::Vllm => parse_model_from_args(cmd),
            EngineType::Sglang => parse_sglang_model_from_args(cmd),
        };
        match seen_endpoints.get(&(engine_type.clone(), endpoint.clone())) {
            Some(&slot) => detected[slot].pids.push(pid),
            None => {
                seen_endpoints.insert((engine_type.clone(), endpoint.clone()), detected.len());
                detected.push(DetectedEngine {
                    engine_type,
                    endpoint,
                    deployment_mode: DeploymentMode::Native,
                    served_model,
                    pids: vec![pid],
                    container_id: None,
                });
            }
        }
    }
    detected
}

/// Expand root PIDs to the full set including every descendant process,
/// given `(pid, parent_pid)` pairs for all processes on the host. Returned
/// sorted and deduplicated; cycles in the parent data are tolerated.
fn expand_pid_tree(roots: &[u32], processes: &[(u32, Option<u32>)]) -> Vec<u32> {
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(pid, parent) in processes {
        if let Some(parent) = parent {
            children.entry(parent).or_default().push(pid);
        }
    }

    let mut seen: HashSet<u32> = HashSet::new();
    let mut queue: Vec<u32> = roots.iter().copied().filter(|&p| seen.insert(p)).collect();
    let mut result = queue.clone();
    while let Some(pid) = queue.pop() {
        for &child in children.get(&pid).map(Vec::as_slice).unwrap_or_default() {
            if seen.insert(child) {
                result.push(child);
                queue.push(child);
            }
        }
    }
    result.sort_unstable();
    result
}

/// Parse `--port` and `--host` from a process's command-line arguments.
/// Returns a constructed endpoint if at least `--port` is found, otherwise
/// falls back to the default.
fn parse_endpoint_from_args(args: &[OsString], default_endpoint: &str) -> Option<String> {
    let args: Vec<String> = args
        .iter()
        .filter_map(|a| a.to_str().map(String::from))
        .collect();

    let mut host: Option<&str> = None;
    let mut port: Option<&str> = None;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--port" {
            if let Some(val) = args.get(i + 1) {
                port = Some(val.as_str());
                i += 2;
                continue;
            }
        } else if let Some(val) = args[i].strip_prefix("--port=") {
            port = Some(val);
        } else if args[i] == "--host" {
            if let Some(val) = args.get(i + 1) {
                host = Some(val.as_str());
                i += 2;
                continue;
            }
        } else if let Some(val) = args[i].strip_prefix("--host=") {
            host = Some(val);
        }
        i += 1;
    }

    // Only build a custom endpoint if we found at least one flag
    if port.is_some() || host.is_some() {
        let h = host.unwrap_or("localhost");
        // Treat 0.0.0.0 as localhost for probing purposes
        let h = if h == "0.0.0.0" { "localhost" } else { h };
        // A `--host` without a `--port` still means the engine's own default
        // port — taken from the per-type default endpoint, not hardcoded, so
        // SGLang does not get probed on vLLM's port.
        let p = port.unwrap_or_else(|| {
            default_endpoint
                .rsplit_once(':')
                .map(|(_, port)| port)
                .unwrap_or("8000")
        });
        Some(format!("http://{}:{}", h, p))
    } else {
        Some(default_endpoint.to_string())
    }
}

/// Parse the served model id from a vLLM process's command-line arguments.
///
/// Recognizes:
///   * `--model <id>` / `--model=<id>` (canonical vLLM flag)
///   * Positional `serve <id>` (matches `vllm serve unsloth/Llama-3.2-1B-Instruct`
///     and `python -m vllm.entrypoints.openai.api_server serve <id>`)
///
/// The `--model` flag is preferred when both forms appear — it is what vLLM
/// itself treats as authoritative.
fn parse_model_from_args(args: &[OsString]) -> Option<String> {
    let args: Vec<String> = args
        .iter()
        .filter_map(|a| a.to_str().map(String::from))
        .collect();

    // First pass: explicit --model flag.
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" {
            if let Some(val) = args.get(i + 1) {
                if !val.is_empty() && !val.starts_with('-') {
                    return Some(val.clone());
                }
            }
        } else if let Some(val) = args[i].strip_prefix("--model=") {
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
        i += 1;
    }

    // Second pass: positional `serve <id>` after a vllm entrypoint token.
    for (idx, arg) in args.iter().enumerate() {
        let is_vllm_entry =
            arg == "vllm" || arg.ends_with("/vllm") || arg.contains("vllm.entrypoints");
        if !is_vllm_entry {
            continue;
        }
        // Find the `serve` subcommand after the entrypoint token.
        if let Some(serve_idx) = args.iter().enumerate().skip(idx + 1).find_map(|(j, a)| {
            if a == "serve" {
                Some(j)
            } else {
                None
            }
        }) {
            if let Some(val) = args.get(serve_idx + 1) {
                if !val.is_empty() && !val.starts_with('-') {
                    return Some(val.clone());
                }
            }
        }
    }

    None
}

/// Parse the served model id from a pre-joined command string (used by Docker
/// `container.command` and `docker top` output rows). Accepts the same shapes
/// as [`parse_model_from_args`].
///
/// Only reachable from the Linux Docker path in normal builds, but the unit
/// tests exercise it on every platform — hence the `test` cfg.
#[cfg(any(target_os = "linux", test))]
fn parse_model_from_command_str(cmd: &str) -> Option<String> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();

    // First pass: explicit --model flag.
    for (i, part) in parts.iter().enumerate() {
        if *part == "--model" {
            if let Some(val) = parts.get(i + 1) {
                if !val.is_empty() && !val.starts_with('-') {
                    return Some((*val).to_string());
                }
            }
        } else if let Some(val) = part.strip_prefix("--model=") {
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
    }

    // Second pass: positional `serve <id>` after a vllm entrypoint token.
    for (idx, part) in parts.iter().enumerate() {
        let is_vllm_entry =
            *part == "vllm" || part.ends_with("/vllm") || part.contains("vllm.entrypoints");
        if !is_vllm_entry {
            continue;
        }
        if let Some(serve_idx) = parts.iter().enumerate().skip(idx + 1).find_map(|(j, a)| {
            if *a == "serve" {
                Some(j)
            } else {
                None
            }
        }) {
            if let Some(val) = parts.get(serve_idx + 1) {
                if !val.is_empty() && !val.starts_with('-') {
                    return Some((*val).to_string());
                }
            }
        }
    }

    None
}

/// Parse the served model from an SGLang launch command line.
///
/// SGLang names the model with `--model-path` (`--model` is its alias) and
/// may rename what `/v1/models` reports with `--served-model-name`. The hint
/// wants the HuggingFace id — what enrichment can resolve — so `--model-path`
/// wins over the alias, and both win over the serve-name (which is often a
/// bare alias exactly when the HF id would help most).
fn parse_sglang_model_from_args(args: &[OsString]) -> Option<String> {
    let args: Vec<String> = args
        .iter()
        .filter_map(|a| a.to_str().map(String::from))
        .collect();
    parse_sglang_model_parts(&args)
}

/// Shared by the OsString and pre-joined-string forms.
fn parse_sglang_model_parts<S: AsRef<str>>(args: &[S]) -> Option<String> {
    let flags = ["--model-path", "--model", "--served-model-name"];
    for flag in flags {
        for (i, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg == flag {
                if let Some(val) = args.get(i + 1) {
                    let val = val.as_ref();
                    if !val.is_empty() && !val.starts_with('-') {
                        return Some(val.to_string());
                    }
                }
            } else if let Some(val) = arg.strip_prefix(&format!("{flag}=")) {
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    // `sglang serve <model>` takes the model as a positional argument, like
    // `vllm serve`; the CLI has no other positional, so the token after the
    // subcommand can only be it.
    if let Some(pos) = args.iter().position(|a| a.as_ref() == "serve") {
        if let Some(val) = args.get(pos + 1) {
            let val = val.as_ref();
            if !val.is_empty() && !val.starts_with('-') {
                return Some(val.to_string());
            }
        }
    }
    None
}

/// Pre-joined-command-string form of [`parse_sglang_model_from_args`] (Docker
/// `container.command` and `docker top` rows).
#[cfg(any(target_os = "linux", test))]
fn parse_sglang_model_from_command_str(cmd: &str) -> Option<String> {
    parse_sglang_model_parts(&cmd.split_whitespace().collect::<Vec<_>>())
}

// ---------------------------------------------------------------------------
// Layer 2: Docker scan
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
pub async fn detect_docker_engines() -> Vec<DetectedEngine> {
    use bollard::query_parameters::{ListContainersOptions, TopOptionsBuilder};
    use bollard::Docker;

    let docker = match Docker::connect_with_local_defaults() {
        Ok(d) => d,
        Err(e) => {
            // Elevated to info: if multi-instance containerized vLLM isn't being
            // picked up, the operator needs to know Docker discovery failed.
            // Common cause: spark-dashboard user not in `docker` group, or the
            // service was started before that group change took effect
            // (`sudo systemctl restart spark-dashboard` usually fixes it).
            tracing::info!(
                "Docker detection unavailable (is spark-dashboard in the `docker` group \
                 and is the socket at /var/run/docker.sock?): {}",
                e
            );
            return vec![];
        }
    };

    let opts = ListContainersOptions {
        all: false, // only running containers
        ..Default::default()
    };

    let containers = match docker.list_containers(Some(opts)).await {
        Ok(c) => c,
        Err(e) => {
            tracing::info!(
                "Failed to list Docker containers (permission/socket issue?): {}",
                e
            );
            return vec![];
        }
    };

    let mut detected = Vec::new();

    for container in &containers {
        let image = container
            .image
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();
        let command = container
            .command
            .as_deref()
            .unwrap_or_default()
            .to_lowercase();

        // The image and the container's entrypoint are trustworthy signals on
        // their own. A container *name* is not: names are operator-chosen and
        // commonly include "vllm" for unrelated sidecars (an OpenResty reverse
        // proxy called "vllm-proxy"), so a name match is only a candidate —
        // confirmed below by finding an actual vLLM process inside it.
        //
        // The name has to count for something, though: a container built from a
        // private image, started with `sleep infinity` and given vLLM by hand is
        // otherwise invisible to the dashboard entirely, however plainly it is
        // named.
        let Some(class) = classify_container(
            container.names.as_deref().unwrap_or_default(),
            &image,
            &command,
        ) else {
            continue;
        };
        let ContainerClass {
            engine_type,
            strong_signal,
        } = class;

        // 1. Try port from Docker port mappings (works for -p / port-forwarding)
        let mapped_port = container
            .ports
            .as_ref()
            .and_then(|ports| ports.iter().find_map(|p| p.public_port));

        // 2. Try port + model from the container's own command string
        let container_cmd = container.command.as_deref().unwrap_or_default();
        let cmd_port = parse_port_from_command_str(container_cmd);
        let cmd_model = parse_model_for(&engine_type, container_cmd);

        let port = mapped_port.map(|p| p.to_string()).or(cmd_port);

        // 3. Inspect the actual processes running inside the container via
        //    `docker top`. Always queried when a container id is known: its rows
        //    carry the *host-namespace* PIDs (the namespace NVML reports) used
        //    for GPU association, and double as the fallback source for the port
        //    (host networking + `sleep infinity` containers) and the model
        //    argument, since `container.command` is just the container's
        //    *entrypoint* and often omits the child vllm-serve args.
        let mut pids: Vec<u32> = Vec::new();
        let mut saw_engine_process = false;
        let (port, served_model) = {
            let container_id = container.id.as_deref().unwrap_or_default();
            if container_id.is_empty() {
                (port, cmd_model)
            } else {
                let top_opts = TopOptionsBuilder::default().ps_args("-eo pid,args").build();
                match docker.top_processes(container_id, Some(top_opts)).await {
                    Ok(top) => {
                        let mut found_port = port;
                        let mut found_model = cmd_model;
                        if let Some(procs) = top.processes.as_ref() {
                            for row in procs {
                                if let Some(pid) = parse_pid_from_top_row(row) {
                                    pids.push(pid);
                                }
                                let line = row.join(" ");
                                if runs_engine(&line, &engine_type) {
                                    saw_engine_process = true;
                                }
                                if found_port.is_none() {
                                    if let Some(p) = parse_port_from_command_str(&line) {
                                        tracing::debug!(
                                            "Docker top: found port {} in: {}",
                                            p,
                                            line
                                        );
                                        found_port = Some(p);
                                    }
                                }
                                if found_model.is_none() {
                                    if let Some(m) = parse_model_for(&engine_type, &line) {
                                        tracing::debug!(
                                            "Docker top: found model {} in: {}",
                                            m,
                                            line
                                        );
                                        found_model = Some(m);
                                    }
                                }
                            }
                        }
                        (found_port, found_model)
                    }
                    Err(e) => {
                        tracing::debug!("docker top failed for {}: {}", container_id, e);
                        (port, cmd_model)
                    }
                }
            }
        };
        pids.sort_unstable();
        pids.dedup();

        // A container that only matched by name has to prove itself. Nothing
        // that merely calls itself an engine becomes an engine.
        if !strong_signal && !saw_engine_process {
            tracing::debug!(
                "Container named like an engine (image={}) runs no engine process; skipping",
                container.image.as_deref().unwrap_or("?"),
            );
            continue;
        }

        // No port anywhere — no published binding, no `--port` in the command,
        // none in the container's process list — means vLLM is on its own
        // default. That is the normal shape of a host-networked container: it
        // publishes nothing, because it is already on the host's ports.
        //
        // Guessing is safe here and dropping the container is not. The health
        // probe below rejects a candidate whose `/health` does not answer, so a
        // wrong guess costs one request and corrects itself, while a container
        // dropped for having no port is invisible to the dashboard for good —
        // engine, metrics and logs alike.
        let (endpoint, guessed) = docker_endpoint(port.as_deref(), default_port(&engine_type));
        tracing::debug!(
            "Docker engine candidate: image={}, endpoint={}{}, model={:?}",
            container.image.as_deref().unwrap_or("?"),
            endpoint,
            if guessed { " (default port)" } else { "" },
            served_model,
        );
        detected.push(DetectedEngine {
            engine_type,
            endpoint,
            deployment_mode: DeploymentMode::Docker,
            served_model,
            pids,
            container_id: container.id.clone(),
        });
    }

    detected
}

/// Parse the host-namespace PID from a `docker top` row produced with
/// `ps_args("-eo pid,args")` — the PID is the first column. Returns `None`
/// for the header row ("PID") or malformed rows.
///
/// Only reachable from the Linux Docker path in normal builds, but the unit
/// tests exercise it on every platform — hence the `test` cfg.
#[cfg(any(target_os = "linux", test))]
fn parse_pid_from_top_row(row: &[String]) -> Option<u32> {
    row.first().and_then(|cell| cell.trim().parse::<u32>().ok())
}

/// Parse `--port` value from a command string (space-separated).
#[cfg(target_os = "linux")]
fn parse_port_from_command_str(cmd: &str) -> Option<String> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "--port" {
            if let Some(val) = parts.get(i + 1) {
                if val.parse::<u16>().is_ok() {
                    return Some(val.to_string());
                }
            }
        } else if let Some(val) = part.strip_prefix("--port=") {
            if val.parse::<u16>().is_ok() {
                return Some(val.to_string());
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub async fn detect_docker_engines() -> Vec<DetectedEngine> {
    tracing::debug!("Docker engine detection stubbed on non-Linux");
    vec![]
}

// ---------------------------------------------------------------------------
// Layer 3: API health probe
// ---------------------------------------------------------------------------

/// Verify that a candidate engine's API is actually responding.
async fn probe_engine(client: &reqwest::Client, candidate: &DetectedEngine) -> bool {
    let timeout = Duration::from_secs(2);

    match candidate.engine_type {
        EngineType::Vllm => {
            // GET /health -- 200 = healthy
            client
                .get(format!("{}/health", candidate.endpoint))
                .timeout(timeout)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        }
        EngineType::Sglang => {
            // SGLang answers Ollama-compatible routes too, so "a route exists"
            // proves little — but `/ready` is SGLang's own cheap readiness
            // probe (vLLM has no such route, and with an API key it answers
            // 401 there rather than 200), and the server-info route is
            // SGLang's own: a 200 (open) or 401 (gated — the route exists)
            // says SGLang, a 404 on both spellings says this is not SGLang.
            // `/server_info` is the canonical route; `/get_server_info` is its
            // deprecated alias, checked so old servers keep working and the
            // probe survives the alias's eventual removal.
            let ready = client
                .get(format!("{}/ready", candidate.endpoint))
                .timeout(timeout)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if !ready {
                return false;
            }
            let info_route_exists = |route: &str| {
                let client = client.clone();
                let endpoint = candidate.endpoint.clone();
                let route = route.to_string();
                async move {
                    client
                        .get(format!("{}{}", endpoint, route))
                        .timeout(timeout)
                        .send()
                        .await
                        .map(|r| r.status().as_u16() != 404)
                        .unwrap_or(false)
                }
            };
            info_route_exists("/server_info").await || info_route_exists("/get_server_info").await
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn to_args(parts: &[&str]) -> Vec<OsString> {
        parts.iter().map(OsString::from).collect()
    }

    #[tokio::test]
    async fn the_sglang_probe_asks_for_its_routes_with_exactly_one_slash() {
        // Regression: the route strings carry their own leading slash, and a
        // format that added another asked SGLang for `//server_info` — which
        // it answers 404, so the probe rejected every real SGLang server.
        // Live-verified against a running server; pinned here so the URL
        // shape cannot regress silently again.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let paths: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let accept_paths = paths.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let paths = accept_paths.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let path = String::from_utf8_lossy(&buf[..n])
                        .lines()
                        .next()
                        .unwrap_or("")
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("")
                        .to_string();
                    paths.lock().unwrap().push(path);
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        )
                        .await;
                });
            }
        });
        let candidate = DetectedEngine {
            engine_type: EngineType::Sglang,
            endpoint: format!("http://127.0.0.1:{}", port),
            deployment_mode: DeploymentMode::Native,
            served_model: None,
            pids: vec![],
            container_id: None,
        };
        assert!(probe_engine(&reqwest::Client::new(), &candidate).await);
        assert_eq!(
            paths.lock().unwrap().clone(),
            vec!["/ready".to_string(), "/server_info".to_string()],
        );
    }

    #[test]
    fn parses_positional_serve_arg() {
        let args = to_args(&["vllm", "serve", "unsloth/Llama-3.2-1B-Instruct"]);
        assert_eq!(
            parse_model_from_args(&args).as_deref(),
            Some("unsloth/Llama-3.2-1B-Instruct"),
        );
    }

    #[test]
    fn parses_explicit_model_flag() {
        let args = to_args(&[
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "Qwen/Qwen2.5-3B-Instruct",
        ]);
        assert_eq!(
            parse_model_from_args(&args).as_deref(),
            Some("Qwen/Qwen2.5-3B-Instruct"),
        );
    }

    #[test]
    fn prefers_model_flag_over_positional_and_handles_equals_form() {
        // `--model=` wins and the trailing `--port` is not misread as a value.
        let args = to_args(&[
            "vllm",
            "serve",
            "--model=mistralai/Mistral-7B",
            "--port",
            "8001",
        ]);
        assert_eq!(
            parse_model_from_args(&args).as_deref(),
            Some("mistralai/Mistral-7B"),
        );
    }

    #[test]
    fn returns_none_when_serve_has_no_positional() {
        let args = to_args(&["vllm", "serve"]);
        assert_eq!(parse_model_from_args(&args), None);
    }

    #[test]
    fn command_str_parses_positional_serve() {
        assert_eq!(
            parse_model_from_command_str("vllm serve google/gemma-2-2b-it --port 8002").as_deref(),
            Some("google/gemma-2-2b-it"),
        );
    }

    #[test]
    fn command_str_parses_explicit_model_equals_flag() {
        assert_eq!(
            parse_model_from_command_str(
                "python -m vllm.entrypoints.openai.api_server --model=meta-llama/Llama-3.1-8B-Instruct",
            )
            .as_deref(),
            Some("meta-llama/Llama-3.1-8B-Instruct"),
        );
    }

    #[test]
    fn command_str_returns_none_for_unrelated_command() {
        assert_eq!(parse_model_from_command_str("sleep infinity"), None);
    }

    fn to_row(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn top_row_pid_is_parsed_from_first_column() {
        assert_eq!(
            parse_pid_from_top_row(&to_row(&["4242", "vllm serve --port 8000"])),
            Some(4242),
        );
    }

    #[test]
    fn top_row_header_and_malformed_rows_yield_none() {
        assert_eq!(parse_pid_from_top_row(&to_row(&["PID", "COMMAND"])), None);
        assert_eq!(parse_pid_from_top_row(&[]), None);
    }

    #[test]
    fn pid_tree_includes_children_and_grandchildren() {
        // 100 (root) -> 101 -> 102, plus unrelated 200 -> 201.
        let processes = vec![
            (100, Some(1)),
            (101, Some(100)),
            (102, Some(101)),
            (200, Some(1)),
            (201, Some(200)),
        ];
        assert_eq!(expand_pid_tree(&[100], &processes), vec![100, 101, 102]);
    }

    #[test]
    fn pid_tree_without_children_returns_only_roots() {
        let processes = vec![(100, Some(1)), (200, Some(1))];
        assert_eq!(expand_pid_tree(&[100], &processes), vec![100]);
    }

    #[test]
    fn pid_tree_tolerates_parent_cycles() {
        // Degenerate parent data forming a cycle must not loop forever.
        let processes = vec![(100, Some(101)), (101, Some(100))];
        assert_eq!(expand_pid_tree(&[100], &processes), vec![100, 101]);
    }

    fn candidate(endpoint: &str, mode: DeploymentMode, pids: &[u32]) -> DetectedEngine {
        DetectedEngine {
            engine_type: EngineType::Vllm,
            endpoint: endpoint.to_string(),
            deployment_mode: mode,
            served_model: None,
            pids: pids.to_vec(),
            container_id: None,
        }
    }

    #[test]
    fn a_vllm_server_process_is_recognized_in_its_several_launch_forms() {
        assert!(runs_engine(
            "/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen3-8B",
            &EngineType::Vllm
        ));
        assert!(runs_engine(
            "vllm serve mistralai/Mistral-7B",
            &EngineType::Vllm
        ));
        assert!(runs_engine(
            "python -m vllm.entrypoints.openai.api_server --port 8001",
            &EngineType::Vllm
        ));
    }

    #[test]
    fn a_process_that_merely_mentions_vllm_is_not_a_server() {
        // What the container-name signal has to be protected against: a proxy
        // or a shell that has the word in its arguments but serves nothing.
        assert!(!runs_engine(
            "nginx: master process /usr/sbin/nginx",
            &EngineType::Vllm
        ));
        assert!(!runs_engine("sleep infinity", &EngineType::Vllm));
        assert!(!runs_engine(
            "/bin/sh -c echo starting vllm-proxy",
            &EngineType::Vllm
        ));
        assert!(!runs_engine("tail -f /var/log/vllm.log", &EngineType::Vllm));
    }

    #[test]
    fn docker_endpoint_uses_the_port_detection_found() {
        assert_eq!(
            docker_endpoint(Some("8001"), VLLM_DEFAULT_PORT),
            ("http://localhost:8001".to_string(), false)
        );
    }

    #[test]
    fn docker_endpoint_assumes_vllms_default_when_no_port_is_readable() {
        // The host-networked container: it publishes no ports because it is
        // already on the host's, and it was started without an explicit
        // --port. Before this it was dropped from detection entirely, which
        // took its metrics and its logs with it.
        assert_eq!(
            docker_endpoint(None, VLLM_DEFAULT_PORT),
            ("http://localhost:8000".to_string(), true)
        );
    }

    fn argv(line: &str) -> Vec<std::ffi::OsString> {
        line.split(' ').map(std::ffi::OsString::from).collect()
    }

    #[test]
    fn container_classification_follows_image_then_command_then_name() {
        // The image is the strongest claim about what runs in a container:
        // an `lmsysorg/sglang` image whose command string mentions vLLM (an
        // env var, a mounted path) is SGLang, not vLLM.
        let class = classify_container(
            &["app".to_string()],
            "lmsysorg/sglang:latest",
            "python3 -m sglang.launch_server --load-prefix /vllm-cache",
        )
        .expect("classified");
        assert!(matches!(class.engine_type, EngineType::Sglang));
        assert!(class.strong_signal);

        // Without image evidence, vLLM still wins a mixed command.
        let class = classify_container(&[], "", "vllm serve --env SGLANG_MODE=1").expect("vllm");
        assert!(matches!(class.engine_type, EngineType::Vllm));

        // A launch form with no command string at all (`sleep infinity`) is
        // only a name — not a strong signal; `docker top` decides.
        let named = classify_container(&["sglang-lane".to_string()], "", "sleep infinity")
            .expect("name still classifies");
        assert!(matches!(named.engine_type, EngineType::Sglang));
        assert!(!named.strong_signal);

        assert!(classify_container(&["sidecar".to_string()], "", "nginx").is_none());
    }

    /// Without image evidence, an SGLang launch command wins: a vLLM-ish
    /// container name or a model path that merely contains "vllm" does not
    /// turn it into vLLM — which would make a weak vLLM candidate dropped by
    /// `docker top`, or a strong one polled on the wrong engine's routes.
    #[test]
    fn an_sglang_launch_command_beats_a_vllm_name_or_path() {
        let sglang = Some(ContainerClass {
            engine_type: EngineType::Sglang,
            strong_signal: true,
        });
        assert_eq!(
            classify_container(
                &["qwen-vllm-compat".to_string()],
                "my/inference",
                "python3 -m sglang.launch_server --port 30000",
            ),
            sglang
        );
        assert_eq!(
            classify_container(
                &[],
                "my/inference",
                "python3 -m sglang.launch_server --model-path /models/x-vllm-awq",
            ),
            sglang
        );
        // And the reverse: a vLLM launch with an SGLang path stays vLLM.
        assert_eq!(
            classify_container(&[], "", "vllm serve /models/sglang-distill"),
            Some(ContainerClass {
                engine_type: EngineType::Vllm,
                strong_signal: true,
            })
        );
    }

    #[test]
    fn a_renamed_sglang_child_is_not_mistaken_for_the_server() {
        // `setproctitle` children name themselves `sglang::scheduler` and
        // `sglang::detokenizer`; their argv proves no server and carries no
        // `--port`. Only the launching process may set the endpoint.
        let sglang = Some(EngineType::Sglang);
        assert_eq!(
            engine_of_process(&argv("python3 -m sglang.launch_server --port 30001")),
            sglang
        );
        assert_eq!(
            engine_of_process(&argv("/usr/local/bin/sglang serve Qwen/Qwen2.5-0.5B")),
            sglang
        );
        assert_eq!(engine_of_process(&argv("sglang::scheduler")), None);
        assert_eq!(engine_of_process(&argv("sglang::detokenizer")), None);
        // The same for vLLM children, and a model path containing the word
        // must not match (single-argument precision).
        assert_eq!(
            engine_of_process(&argv("vllm serve google/gemma-3-4b --port 8001")),
            Some(EngineType::Vllm)
        );
        assert_eq!(engine_of_process(&argv("tail -f /var/log/vllm.log")), None);
        // One argument holding the string whole: per-argument precision
        // means a path containing the word is not the binary being run.
        let whole_arg = ["python3", "serve", "/models/my vllm model"].map(std::ffi::OsString::from);
        assert_eq!(engine_of_process(&whole_arg), None);
    }

    /// A process is at most one engine. An SGLang server with a vLLM path
    /// among its arguments used to match both scans and become two
    /// candidates on its port — and the vLLM one polled SGLang's `/health`,
    /// a one-token generation, every second.
    #[test]
    fn a_process_is_detected_as_one_engine_only() {
        let sglang = argv("python3 -m sglang.launch_server --download-dir /data/vllm --port 30000");
        let vllm = argv("vllm serve /models/sglang-distill --port 8001");
        let detected = engines_from_processes([(10, sglang.as_slice()), (20, vllm.as_slice())]);
        let found: Vec<_> = detected
            .iter()
            .map(|d| (d.engine_type.clone(), d.endpoint.as_str(), d.pids.clone()))
            .collect();
        assert_eq!(
            found,
            vec![
                (EngineType::Sglang, "http://localhost:30000", vec![10]),
                (EngineType::Vllm, "http://localhost:8001", vec![20]),
            ]
        );
    }

    /// `--host` without `--port` still serves on the engine's own default
    /// port: SGLang's 30000, never vLLM's 8000.
    #[test]
    fn a_host_without_a_port_keeps_the_engines_default_port() {
        let sglang = argv("python3 -m sglang.launch_server --host 0.0.0.0");
        let vllm = argv("vllm serve Qwen/Qwen3-8B --host 0.0.0.0");
        let endpoints: Vec<_> =
            engines_from_processes([(1, sglang.as_slice()), (2, vllm.as_slice())])
                .into_iter()
                .map(|d| d.endpoint)
                .collect();
        assert_eq!(
            endpoints,
            vec!["http://localhost:30000", "http://localhost:8000"]
        );
    }

    /// Several processes of one server on one endpoint (workers forked with
    /// the launch argv) are one engine, holding every PID.
    #[test]
    fn processes_on_one_endpoint_are_one_engine() {
        let a = argv("vllm serve m --port 8000");
        let b = argv("vllm serve m --port 8000");
        let c = argv("vllm serve m --port 8001");
        let detected =
            engines_from_processes([(1, a.as_slice()), (2, b.as_slice()), (3, c.as_slice())]);
        assert_eq!(detected.len(), 2);
        assert_eq!(detected[0].pids, vec![1, 2]);
        assert_eq!(detected[1].pids, vec![3]);
    }

    #[test]
    fn a_sglang_server_process_is_recognized_in_its_launch_forms() {
        assert!(runs_engine(
            "python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --port 30001",
            &EngineType::Sglang
        ));
        assert!(runs_engine(
            "sglang serve --model Qwen/Qwen2.5-3B-Instruct",
            &EngineType::Sglang
        ));
        assert!(runs_engine(
            "/usr/local/bin/sglang serve --model-path=google/gemma-3-4b",
            &EngineType::Sglang
        ));
        assert!(runs_engine(
            "python -m sglang.srt.server --host 127.0.0.1 --port 30000",
            &EngineType::Sglang
        ));
    }

    #[test]
    fn a_process_that_merely_mentions_sglang_is_not_a_server() {
        assert!(!runs_engine(
            "nginx: master process /usr/sbin/nginx",
            &EngineType::Sglang
        ));
        assert!(!runs_engine(
            "tail -f /var/log/sglang.log",
            &EngineType::Sglang
        ));
        assert!(!runs_engine(
            "/bin/sh -c echo starting sglang-proxy",
            &EngineType::Sglang
        ));
        assert!(!runs_engine("sleep infinity", &EngineType::Sglang));
    }

    /// A container's command is read with its engine's own model flags:
    /// `--model-path` for SGLang, `--model` or `serve <id>` for vLLM.
    #[test]
    fn a_container_command_is_read_with_its_engines_model_flags() {
        let cmd = "python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000";
        assert_eq!(
            parse_model_for(&EngineType::Sglang, cmd).as_deref(),
            Some("Qwen/Qwen3-8B")
        );
        assert_eq!(
            parse_model_for(&EngineType::Vllm, "vllm serve Qwen/Qwen3-8B").as_deref(),
            Some("Qwen/Qwen3-8B")
        );
    }

    #[test]
    fn sglang_model_parses_its_flags_with_the_right_precedence() {
        // --model-path is the HF id enrichment can resolve, so it wins over
        // the alias flag and over the serve-name flag.
        assert_eq!(
            parse_sglang_model_from_command_str(
                "sglang serve --model-path meta-llama/Llama-3.1-8B --served-model-name llama8b --port 30000",
            )
            .as_deref(),
            Some("meta-llama/Llama-3.1-8B"),
        );
        assert_eq!(
            parse_sglang_model_from_command_str(
                "python -m sglang.launch_server --model=Qwen/Qwen2.5-3B-Instruct"
            )
            .as_deref(),
            Some("Qwen/Qwen2.5-3B-Instruct"),
        );
        // Only an alias present: still better than nothing.
        assert_eq!(
            parse_sglang_model_from_command_str("sglang serve --served-model-name my-model")
                .as_deref(),
            Some("my-model"),
        );
        // `sglang serve <model>` — positional, no flag at all.
        assert_eq!(
            parse_sglang_model_from_command_str(
                "sglang serve Qwen/Qwen2.5-0.5B-Instruct --port 30000",
            )
            .as_deref(),
            Some("Qwen/Qwen2.5-0.5B-Instruct"),
        );
        assert_eq!(parse_sglang_model_from_command_str("sleep infinity"), None);
    }

    #[test]
    fn docker_merge_unions_pids_and_upgrades_mode() {
        let mut candidates = vec![candidate(
            "http://localhost:8000",
            DeploymentMode::Native,
            &[100, 101],
        )];
        let docker = vec![candidate(
            "http://localhost:8000",
            DeploymentMode::Docker,
            &[101, 102],
        )];
        merge_docker_candidates(&mut candidates, docker);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].deployment_mode, DeploymentMode::Docker);
        assert_eq!(candidates[0].pids, vec![100, 101, 102]);
    }

    #[test]
    fn docker_merge_keeps_distinct_endpoints_separate() {
        let mut candidates = vec![candidate(
            "http://localhost:8000",
            DeploymentMode::Native,
            &[100],
        )];
        let docker = vec![candidate(
            "http://localhost:8001",
            DeploymentMode::Docker,
            &[200],
        )];
        merge_docker_candidates(&mut candidates, docker);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].pids, vec![100]);
        assert_eq!(candidates[1].pids, vec![200]);
    }
}
