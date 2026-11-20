use std::collections::{BTreeSet, HashMap};

/// Tracks independent pause reasons both globally (`*`) and per Wayland output.
///
/// Removing one reason (e.g. `fullscreen`) never clears unrelated reasons
/// such as `manual` or `battery`.
#[derive(Debug, Default, Clone)]
pub struct PauseState {
    global: BTreeSet<String>,
    per_output: HashMap<String, BTreeSet<String>>,
}

impl PauseState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pause reason globally (`output == None` or `"*"`) or for a specific output.
    /// Returns `true` if the reason was newly inserted.
    pub fn add(&mut self, output: Option<&str>, reason: &str) -> bool {
        let reason = normalize_reason(reason);
        match normalize_output(output) {
            None => self.global.insert(reason),
            Some(out) => self
                .per_output
                .entry(out.to_string())
                .or_default()
                .insert(reason),
        }
    }

    /// Remove a pause reason globally or for a specific output.
    /// Returns `true` if the reason was present and removed.
    pub fn remove(&mut self, output: Option<&str>, reason: &str) -> bool {
        let reason = normalize_reason(reason);
        match normalize_output(output) {
            None => {
                let removed_global = self.global.remove(&reason);
                // Explicit global resume of a tag also clears per-output tags only when targeting all;
                // Wait: keep global & per-output strict? For `manual` toggle/resume on `*`, clearing
                // only `global` keeps per-output auto-pause (`fullscreen`) intact, and for explicit
                // per-output `manual`, we also clear matching manual tags if global was empty.
                let mut removed_any = removed_global;
                if !removed_global {
                    for set in self.per_output.values_mut() {
                        if set.remove(&reason) {
                            removed_any = true;
                        }
                    }
                    self.per_output.retain(|_, set| !set.is_empty());
                }
                removed_any
            }
            Some(out) => {
                if let Some(set) = self.per_output.get_mut(out) {
                    let r = set.remove(&reason);
                    if set.is_empty() {
                        self.per_output.remove(out);
                    }
                    r
                } else {
                    false
                }
            }
        }
    }

    /// Remove an output entry completely (e.g. on Wayland output hot-unplug).
    pub fn drop_output(&mut self, output: &str) {
        self.per_output.remove(output);
    }

    /// Toggle `"manual"` pause state globally or per output.
    /// Returns the new manual-paused boolean for that target.
    pub fn toggle_manual(&mut self, output: Option<&str>) -> bool {
        const MANUAL: &str = "manual";
        match normalize_output(output) {
            None => {
                if self.global.contains(MANUAL) {
                    self.global.remove(MANUAL);
                    false
                } else {
                    self.global.insert(MANUAL.to_string());
                    true
                }
            }
            Some(out) => {
                let is_set = self.per_output.get(out).is_some_and(|s| s.contains(MANUAL));
                if is_set {
                    self.remove(Some(out), MANUAL);
                    false
                } else {
                    self.add(Some(out), MANUAL);
                    true
                }
            }
        }
    }

    /// Returns `true` if the given output has any active global or per-output pause reason.
    pub fn is_paused(&self, output: &str) -> bool {
        !self.global.is_empty() || self.per_output.get(output).is_some_and(|s| !s.is_empty())
    }

    /// Returns deduplicated sorted active reasons affecting `output`.
    pub fn reasons_for(&self, output: &str) -> Vec<String> {
        let mut combined = self.global.clone();
        if let Some(local) = self.per_output.get(output) {
            combined.extend(local.iter().cloned());
        }
        combined.into_iter().collect()
    }

    pub fn global_reasons(&self) -> Vec<String> {
        self.global.iter().cloned().collect()
    }
}

fn normalize_output(output: Option<&str>) -> Option<&str> {
    match output.map(str::trim) {
        None | Some("") | Some("*") => None,
        Some(other) => Some(other),
    }
}

fn normalize_reason(reason: &str) -> String {
    let trimmed = reason.trim();
    if trimmed.is_empty() {
        "manual".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::PauseState;

    #[test]
    fn independent_reasons_do_not_clobber_each_other() {
        let mut pause = PauseState::new();
        assert!(!pause.is_paused("DP-1"));

        pause.add(None, "manual");
        pause.add(Some("DP-1"), "fullscreen");
        pause.add(Some("HDMI-A-1"), "battery");

        assert!(pause.is_paused("DP-1"));
        assert!(pause.is_paused("HDMI-A-1"));
        assert_eq!(pause.reasons_for("DP-1"), vec!["fullscreen", "manual"]);

        // Exiting fullscreen on DP-1 must keep global manual pause active.
        assert!(pause.remove(Some("DP-1"), "fullscreen"));
        assert!(pause.is_paused("DP-1"));
        assert_eq!(pause.reasons_for("DP-1"), vec!["manual"]);

        // Clearing global manual pause resumes DP-1, while HDMI-A-1 still keeps battery pause.
        assert!(pause.remove(None, "manual"));
        assert!(!pause.is_paused("DP-1"));
        assert!(pause.is_paused("HDMI-A-1"));
        assert_eq!(pause.reasons_for("HDMI-A-1"), vec!["battery"]);
    }

    #[test]
    fn toggle_manual_flips_cleanly() {
        let mut pause = PauseState::new();
        assert!(pause.toggle_manual(None));
        assert!(pause.is_paused("DP-1"));
        assert!(!pause.toggle_manual(None));
        assert!(!pause.is_paused("DP-1"));

        assert!(pause.toggle_manual(Some("DP-2")));
        assert!(pause.is_paused("DP-2"));
        assert!(!pause.is_paused("DP-1"));
        assert!(!pause.toggle_manual(Some("DP-2")));
        assert!(!pause.is_paused("DP-2"));
    }
}
