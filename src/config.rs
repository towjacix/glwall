use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::{FitArg, HwAccelArg};
use crate::gpu::uniforms::{TextureSpec, UniformValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FitMode {
    #[default]
    Cover,
    Contain,
    Stretch,
    Center,
}

impl FitMode {
    pub fn as_gl_int(self) -> i32 {
        match self {
            FitMode::Cover => 0,
            FitMode::Contain => 1,
            FitMode::Stretch => 2,
            FitMode::Center => 3,
        }
    }
}

impl From<FitArg> for FitMode {
    fn from(arg: FitArg) -> Self {
        match arg {
            FitArg::Cover => FitMode::Cover,
            FitArg::Contain => FitMode::Contain,
            FitArg::Stretch => FitMode::Stretch,
            FitArg::Center => FitMode::Center,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum HwAccel {
    #[default]
    Auto,
    Cuda,
    Vaapi,
    None,
}

impl HwAccel {
    pub fn as_ffmpeg_arg(self) -> Option<&'static str> {
        match self {
            HwAccel::Auto => Some("auto"),
            HwAccel::Cuda => Some("cuda"),
            HwAccel::Vaapi => Some("vaapi"),
            HwAccel::None => None,
        }
    }
}

impl From<HwAccelArg> for HwAccel {
    fn from(arg: HwAccelArg) -> Self {
        match arg {
            HwAccelArg::Auto => HwAccel::Auto,
            HwAccelArg::Cuda => HwAccel::Cuda,
            HwAccelArg::Vaapi => HwAccel::Vaapi,
            HwAccelArg::None => HwAccel::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    #[default]
    Bt709Tv,
    Bt709Pc,
    Bt601Tv,
    Bt601Pc,
}

impl ColorSpace {
    pub fn as_gl_int(self) -> i32 {
        match self {
            ColorSpace::Bt709Tv => 0,
            ColorSpace::Bt709Pc => 1,
            ColorSpace::Bt601Tv => 2,
            ColorSpace::Bt601Pc => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputSpec {
    pub target: Option<String>,
    pub fit: FitMode,
    pub shader: Option<String>,
    pub transition: String,
    pub duration_ms: u32,
    pub params: [f32; 4],
    pub fps: Option<f32>,
    pub hwaccel: HwAccel,
    pub textures: BTreeMap<String, TextureSpec>,
    pub uniforms: BTreeMap<String, UniformValue>,
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            target: None,
            fit: FitMode::Cover,
            shader: None,
            transition: "fade".to_string(),
            duration_ms: 450,
            params: [0.0; 4],
            fps: None,
            hwaccel: HwAccel::Auto,
            textures: BTreeMap::new(),
            uniforms: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub target: Option<String>,
    pub fit: Option<FitMode>,
    pub shader: Option<String>,
    pub transition: Option<String>,
    pub duration_ms: Option<u32>,
    pub params: Option<[f32; 4]>,
    pub fps: Option<f32>,
    pub hwaccel: Option<HwAccel>,
    pub textures: BTreeMap<String, TextureSpec>,
    pub uniforms: BTreeMap<String, UniformValue>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub default_fit: FitMode,
    pub default_transition: String,
    pub default_transition_ms: u32,
    pub auto_pause_fullscreen: bool,
    #[serde(default = "default_true")]
    pub persist_state: bool,
    pub fps_cap: Option<f32>,
    pub hwaccel: HwAccel,
    pub outputs: HashMap<String, OutputConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_fit: FitMode::Cover,
            default_transition: "fade".to_string(),
            default_transition_ms: 450,
            auto_pause_fullscreen: true,
            persist_state: true,
            fps_cap: None,
            hwaccel: HwAccel::Auto,
            outputs: HashMap::new(),
        }
    }
}

impl Config {
    pub fn load_or_default(explicit: Option<&Path>) -> Result<Self> {
        let path = explicit.map(Path::to_path_buf).or_else(default_config_path);
        let Some(path) = path else {
            return Ok(Self::default());
        };
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("failed to parse config {}", path.display()))
    }

    pub fn base_spec_for(&self, output: &str) -> OutputSpec {
        let mut spec = OutputSpec {
            target: None,
            fit: self.default_fit,
            shader: None,
            transition: self.default_transition.clone(),
            duration_ms: self.default_transition_ms,
            params: [0.0; 4],
            fps: self.fps_cap,
            hwaccel: self.hwaccel,
            textures: BTreeMap::new(),
            uniforms: BTreeMap::new(),
        };
        if let Some(cfg) = self.outputs.get("*").or_else(|| self.outputs.get(output)) {
            apply_output_config(&mut spec, cfg);
        }
        if self.outputs.contains_key("*")
            && let Some(cfg) = self.outputs.get(output)
        {
            apply_output_config(&mut spec, cfg);
        }
        spec
    }
}

fn apply_output_config(spec: &mut OutputSpec, cfg: &OutputConfig) {
    if cfg.target.is_some() {
        spec.target = cfg.target.clone();
    }
    if let Some(fit) = cfg.fit {
        spec.fit = fit;
    }
    if cfg.shader.is_some() {
        spec.shader = cfg.shader.clone();
    }
    if let Some(ref tr) = cfg.transition {
        spec.transition = tr.clone();
    }
    if let Some(ms) = cfg.duration_ms {
        spec.duration_ms = ms;
    }
    if let Some(params) = cfg.params {
        spec.params = params;
    }
    if cfg.fps.is_some() {
        spec.fps = cfg.fps;
    }
    if let Some(hw) = cfg.hwaccel {
        spec.hwaccel = hw;
    }
    for (name, tex) in &cfg.textures {
        spec.textures.insert(name.clone(), tex.clone());
    }
    for (name, val) in &cfg.uniforms {
        spec.uniforms.insert(name.clone(), val.clone());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersistedState {
    pub version: u32,
    pub outputs: HashMap<String, OutputSpec>,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            version: 1,
            outputs: HashMap::new(),
        }
    }
}

impl PersistedState {
    pub fn load_or_default(path: &Path) -> Self {
        let Ok(raw) = fs::read_to_string(path) else {
            return Self::default();
        };
        serde_json::from_str::<Self>(&raw)
            .ok()
            .filter(|s| s.version == 1)
            .unwrap_or_default()
    }

    pub fn save_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create state directory {}", parent.display())
            })?;
        }
        let encoded = serde_json::to_vec_pretty(self)?;
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("state.json");
        let tmp_path = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
        let mut file = fs::File::create(&tmp_path)
            .with_context(|| format!("failed to create temp state {}", tmp_path.display()))?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp_path, path)
            .with_context(|| format!("failed to rename state into {}", path.display()))?;
        Ok(())
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.trim().is_empty()
    {
        return Some(PathBuf::from(xdg).join("glwall/config.toml"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config/glwall/config.toml"))
}

pub fn default_state_path() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_STATE_HOME")
        && !xdg.trim().is_empty()
    {
        return Some(PathBuf::from(xdg).join("glwall/state.json"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state/glwall/state.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::uniforms::{TextureFilter, TextureWrap};

    #[test]
    fn config_parses_defaults_and_overrides() {
        let toml_src = r#"
default_fit = "contain"
default_transition = "dissolve"
default_transition_ms = 600
auto_pause_fullscreen = false
persist_state = false
fps_cap = 60.0
hwaccel = "cuda"

[outputs."*"]
params = [0.2, 0.4, 0.6, 0.8]

[outputs."*".uniforms]
u_parallaxStrength = 0.04
u_pointer = [0.5, 0.5]

[outputs."DP-1"]
fit = "stretch"
transition = "wipe"

[outputs."DP-1".textures.u_depth]
path = "/tmp/depth.png"
filter = "linear"
wrap = "clamp_to_edge"
flip_y = false

[outputs."DP-1".uniforms]
u_pointer = [0.8, 0.2]
"#;
        let cfg: Config = toml::from_str(toml_src).unwrap();
        assert!(!cfg.auto_pause_fullscreen);
        assert!(!cfg.persist_state);
        let dp1 = cfg.base_spec_for("DP-1");
        assert_eq!(dp1.fit, FitMode::Stretch);
        assert_eq!(dp1.transition, "wipe");
        assert_eq!(dp1.params, [0.2, 0.4, 0.6, 0.8]);
        assert_eq!(dp1.hwaccel, HwAccel::Cuda);
        assert_eq!(
            dp1.uniforms.get("u_parallaxStrength"),
            Some(&UniformValue::Float(0.04))
        );
        assert_eq!(
            dp1.uniforms.get("u_pointer"),
            Some(&UniformValue::Vec2([0.8, 0.2]))
        );
        assert_eq!(
            dp1.textures.get("u_depth"),
            Some(&TextureSpec {
                path: PathBuf::from("/tmp/depth.png"),
                filter: TextureFilter::Linear,
                wrap: TextureWrap::ClampToEdge,
                flip_y: false,
            })
        );

        let edp1 = cfg.base_spec_for("eDP-1");
        assert_eq!(edp1.fit, FitMode::Contain);
        assert_eq!(edp1.transition, "dissolve");
        assert_eq!(edp1.duration_ms, 600);
        assert_eq!(
            edp1.uniforms.get("u_pointer"),
            Some(&UniformValue::Vec2([0.5, 0.5]))
        );
    }

    #[test]
    fn persisted_state_roundtrips_atomically_and_accepts_v1_minimal() {
        let dir = std::env::temp_dir().join(format!("glwall-cfg-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let state_path = dir.join("state.json");

        let mut state = PersistedState::default();
        let mut textures = BTreeMap::new();
        textures.insert(
            "u_depth".to_string(),
            TextureSpec {
                path: PathBuf::from("/tmp/depth.png"),
                filter: TextureFilter::Nearest,
                wrap: TextureWrap::Repeat,
                flip_y: true,
            },
        );
        let mut uniforms = BTreeMap::new();
        uniforms.insert("u_pointer".to_string(), UniformValue::Vec2([0.25, 0.75]));

        state.outputs.insert(
            "DP-1".to_string(),
            OutputSpec {
                target: Some("/tmp/wall.png".to_string()),
                fit: FitMode::Cover,
                shader: Some("/tmp/grain.frag".to_string()),
                transition: "fade".to_string(),
                duration_ms: 450,
                params: [1.0, 2.0, 3.0, 4.0],
                fps: Some(60.0),
                hwaccel: HwAccel::Auto,
                textures,
                uniforms,
            },
        );

        state.save_atomic(&state_path).unwrap();
        let loaded = PersistedState::load_or_default(&state_path);
        assert_eq!(loaded, state);

        // v0.1.x state.json without `textures`/`uniforms` fields still loads cleanly.
        let legacy_json = r#"{
            "version": 1,
            "outputs": {
                "eDP-1": {
                    "target": "/tmp/bg.png",
                    "fit": "cover",
                    "shader": null,
                    "transition": "fade",
                    "duration_ms": 450,
                    "params": [0.0, 0.0, 0.0, 0.0],
                    "fps": null,
                    "hwaccel": "auto"
                }
            }
        }"#;
        let legacy: PersistedState = serde_json::from_str(legacy_json).unwrap();
        assert!(legacy.outputs["eDP-1"].textures.is_empty());
        assert!(legacy.outputs["eDP-1"].uniforms.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }
}
