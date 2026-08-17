//! Exit classification, per PLAN.md §7.4's table. Pure — takes the signals
//! the supervisor already has (exit code/signal, whether `/health` was ever
//! seen, and the captured stderr tail) and turns them into one of a fixed
//! set of causes. Quantitative *advice* for the OOM cases lives in
//! `studio_core::estimator::oom_advice`, not here — this module only knows
//! about process signals, not VRAM math.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExitClassification {
    /// stderr shows an OOM signature and `/health` never returned 200 —
    /// the weights alone don't fit.
    OomAtLoad,
    /// Same OOM signature, but after `/health` was already OK once — hit
    /// during a real request's prefill.
    OomAtPrefill,
    /// Exit code 137 / killed by `SIGKILL` from outside us — the OS OOM
    /// killer, not a VRAM problem.
    HostOomKilled,
    /// A `CRANE_ISQ`/config panic — should have been caught by
    /// `LaunchSpec::validate()` before ever spawning; this is a `CraneStudio`
    /// bug if it happens.
    BadConfig,
    /// The chosen port was already in use.
    PortInUse,
    /// Exited 0 before `/health` ever came up — usually a bad
    /// `--model-type` or a missing tokenizer file.
    CleanEarlyExit,
    /// We asked it to stop — not a failure at all.
    Stopped,
    /// None of the above matched; show the stderr tail verbatim.
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub struct ExitContext<'a> {
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub health_ok_observed: bool,
    pub stderr_tail: &'a str,
    /// Set when the supervisor itself sent the stop signal — distinguishes
    /// an intentional shutdown from every other exit reason.
    pub requested_stop: bool,
}

#[must_use]
pub fn classify(ctx: &ExitContext) -> ExitClassification {
    if ctx.requested_stop {
        return ExitClassification::Stopped;
    }
    if ctx.signal == Some(9) || ctx.exit_code == Some(137) {
        return ExitClassification::HostOomKilled;
    }
    if ctx.stderr_tail.contains("invalid CRANE_ISQ") {
        return ExitClassification::BadConfig;
    }
    if is_port_in_use(ctx.stderr_tail) {
        return ExitClassification::PortInUse;
    }
    if is_oom_signature(ctx.stderr_tail) {
        return if ctx.health_ok_observed {
            ExitClassification::OomAtPrefill
        } else {
            ExitClassification::OomAtLoad
        };
    }
    if ctx.exit_code == Some(0) && !ctx.health_ok_observed {
        return ExitClassification::CleanEarlyExit;
    }
    ExitClassification::Unknown
}

fn is_oom_signature(stderr: &str) -> bool {
    stderr.contains("CUDA_ERROR_OUT_OF_MEMORY") || stderr.to_lowercase().contains("out of memory")
}

fn is_port_in_use(stderr: &str) -> bool {
    let lower = stderr.to_lowercase();
    lower.contains("address already in use") || (lower.contains("bind") && lower.contains("in use"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(
        exit_code: Option<i32>,
        signal: Option<i32>,
        health_ok: bool,
        stderr: &str,
    ) -> ExitContext<'_> {
        ExitContext {
            exit_code,
            signal,
            health_ok_observed: health_ok,
            stderr_tail: stderr,
            requested_stop: false,
        }
    }

    #[test]
    fn oom_before_health_is_oom_at_load() {
        let c = ctx(
            Some(1),
            None,
            false,
            "thread panicked: CUDA_ERROR_OUT_OF_MEMORY",
        );
        assert_eq!(classify(&c), ExitClassification::OomAtLoad);
    }

    #[test]
    fn oom_after_health_is_oom_at_prefill() {
        let c = ctx(Some(1), None, true, "CUDA error: out of memory");
        assert_eq!(classify(&c), ExitClassification::OomAtPrefill);
    }

    #[test]
    fn sigkill_is_host_oom_regardless_of_stderr() {
        let c = ctx(None, Some(9), true, "");
        assert_eq!(classify(&c), ExitClassification::HostOomKilled);
        let c2 = ctx(Some(137), None, false, "");
        assert_eq!(classify(&c2), ExitClassification::HostOomKilled);
    }

    #[test]
    fn isq_panic_is_bad_config() {
        let c = ctx(
            Some(101),
            None,
            false,
            "thread 'main' panicked at src/models/qwen3_5/model.rs:624:\ninvalid CRANE_ISQ: unknown quantization level 'bogus'",
        );
        assert_eq!(classify(&c), ExitClassification::BadConfig);
    }

    #[test]
    fn port_conflict_is_recognized() {
        let c = ctx(
            Some(1),
            None,
            false,
            "Error: Address already in use (os error 98)",
        );
        assert_eq!(classify(&c), ExitClassification::PortInUse);
    }

    #[test]
    fn clean_zero_exit_before_health_is_early_exit() {
        let c = ctx(Some(0), None, false, "missing tokenizer.json");
        assert_eq!(classify(&c), ExitClassification::CleanEarlyExit);
    }

    #[test]
    fn requested_stop_wins_over_everything_else() {
        let mut c = ctx(Some(137), Some(9), false, "CUDA_ERROR_OUT_OF_MEMORY");
        c.requested_stop = true;
        assert_eq!(classify(&c), ExitClassification::Stopped);
    }

    #[test]
    fn unrecognized_signal_is_unknown() {
        let c = ctx(Some(1), None, true, "some unrelated crash");
        assert_eq!(classify(&c), ExitClassification::Unknown);
    }
}
