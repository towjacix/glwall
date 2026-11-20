use std::fmt;
use std::path::PathBuf;

use anyhow::{Result, bail};
use glow::HasContext;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UniformType {
    Float,
    Int,
    Bool,
    Vec2,
    Vec3,
    Vec4,
}

impl UniformType {
    pub fn from_gl_type(gl_type: u32) -> Option<Self> {
        match gl_type {
            glow::FLOAT => Some(Self::Float),
            glow::INT => Some(Self::Int),
            glow::BOOL => Some(Self::Bool),
            glow::FLOAT_VEC2 => Some(Self::Vec2),
            glow::FLOAT_VEC3 => Some(Self::Vec3),
            glow::FLOAT_VEC4 => Some(Self::Vec4),
            _ => None,
        }
    }

    pub fn gl_type_name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Int => "int",
            Self::Bool => "bool",
            Self::Vec2 => "vec2",
            Self::Vec3 => "vec3",
            Self::Vec4 => "vec4",
        }
    }
}

impl fmt::Display for UniformType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.gl_type_name())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UniformValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

impl UniformValue {
    pub fn value_type(&self) -> UniformType {
        match self {
            Self::Bool(_) => UniformType::Bool,
            Self::Int(_) => UniformType::Int,
            Self::Float(_) => UniformType::Float,
            Self::Vec2(_) => UniformType::Vec2,
            Self::Vec3(_) => UniformType::Vec3,
            Self::Vec4(_) => UniformType::Vec4,
        }
    }

    /// Parses a single `NAME=VALUE` uniform assignment from the CLI.
    pub fn parse_cli(spec: &str) -> Result<(String, Self)> {
        let Some((raw_name, raw_val)) = spec.split_once('=') else {
            bail!("invalid uniform '{spec}': expected NAME=VALUE");
        };
        let name = raw_name.trim();
        if name.is_empty() {
            bail!("invalid uniform '{spec}': empty uniform name");
        }
        let val = Self::parse_value_str(raw_val)?;
        Ok((name.to_string(), val))
    }

    pub fn parse_value_str(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            bail!("empty uniform value");
        }
        if trimmed.eq_ignore_ascii_case("true") {
            return Ok(Self::Bool(true));
        }
        if trimmed.eq_ignore_ascii_case("false") {
            return Ok(Self::Bool(false));
        }

        let inner = trimmed
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(trimmed)
            .trim();

        if inner.contains(',') {
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            let parse_f32 = |s: &str| -> Result<f32> {
                let v: f32 = s
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid float component '{s}' in '{trimmed}'"))?;
                if !v.is_finite() {
                    bail!("non-finite float component '{s}' in '{trimmed}'");
                }
                Ok(v)
            };
            return match parts.as_slice() {
                [a, b] => Ok(Self::Vec2([parse_f32(a)?, parse_f32(b)?])),
                [a, b, c] => Ok(Self::Vec3([parse_f32(a)?, parse_f32(b)?, parse_f32(c)?])),
                [a, b, c, d] => Ok(Self::Vec4([
                    parse_f32(a)?,
                    parse_f32(b)?,
                    parse_f32(c)?,
                    parse_f32(d)?,
                ])),
                _ => bail!(
                    "vector uniform '{trimmed}' has {} components (expected 2, 3, or 4)",
                    parts.len()
                ),
            };
        }

        // Preserve exact integers (e.g. `2`) for GL `int` uniforms; `upload_uniform`
        // widens `Int -> Float` automatically when the reflected GL type is `float`.
        if !inner.contains(['.', 'e', 'E'])
            && let Ok(i) = inner.parse::<i32>()
        {
            return Ok(Self::Int(i));
        }

        let v: f32 = inner
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid scalar uniform value '{trimmed}'"))?;
        if !v.is_finite() {
            bail!("non-finite float uniform value '{trimmed}'");
        }
        Ok(Self::Float(v))
    }

    pub fn is_finite(&self) -> bool {
        match self {
            Self::Bool(_) | Self::Int(_) => true,
            Self::Float(v) => v.is_finite(),
            Self::Vec2(v) => v.iter().all(|x| x.is_finite()),
            Self::Vec3(v) => v.iter().all(|x| x.is_finite()),
            Self::Vec4(v) => v.iter().all(|x| x.is_finite()),
        }
    }
}

pub fn upload_uniform(
    gl: &glow::Context,
    loc: &glow::UniformLocation,
    expected: UniformType,
    value: &UniformValue,
) -> Result<()> {
    if !value.is_finite() {
        bail!("refusing to upload non-finite uniform value");
    }
    unsafe {
        match (expected, value) {
            (UniformType::Float, UniformValue::Float(v)) => gl.uniform_1_f32(Some(loc), *v),
            (UniformType::Float, UniformValue::Int(i)) => gl.uniform_1_f32(Some(loc), *i as f32),
            (UniformType::Int, UniformValue::Int(i)) => gl.uniform_1_i32(Some(loc), *i),
            (UniformType::Int, UniformValue::Bool(b)) => {
                gl.uniform_1_i32(Some(loc), if *b { 1 } else { 0 })
            }
            (UniformType::Bool, UniformValue::Bool(b)) => {
                gl.uniform_1_i32(Some(loc), if *b { 1 } else { 0 })
            }
            (UniformType::Bool, UniformValue::Int(i)) => {
                gl.uniform_1_i32(Some(loc), if *i != 0 { 1 } else { 0 })
            }
            (UniformType::Vec2, UniformValue::Vec2(v)) => gl.uniform_2_f32(Some(loc), v[0], v[1]),
            (UniformType::Vec3, UniformValue::Vec3(v)) => {
                gl.uniform_3_f32(Some(loc), v[0], v[1], v[2])
            }
            (UniformType::Vec4, UniformValue::Vec4(v)) => {
                gl.uniform_4_f32(Some(loc), v[0], v[1], v[2], v[3])
            }
            _ => bail!(
                "uniform type mismatch: shader expects {}, got {}",
                expected,
                value.value_type()
            ),
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureFilter {
    #[default]
    Linear,
    Nearest,
}

impl TextureFilter {
    pub fn to_gl(self) -> u32 {
        match self {
            Self::Linear => glow::LINEAR,
            Self::Nearest => glow::NEAREST,
        }
    }

    pub fn parse_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "nearest" => Ok(Self::Nearest),
            other => bail!("invalid texture filter '{other}' (expected linear or nearest)"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureWrap {
    #[default]
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

impl TextureWrap {
    pub fn to_gl(self) -> u32 {
        match self {
            Self::ClampToEdge => glow::CLAMP_TO_EDGE,
            Self::Repeat => glow::REPEAT,
            Self::MirroredRepeat => glow::MIRRORED_REPEAT,
        }
    }

    pub fn parse_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "clamp" | "clamp_to_edge" | "clamp-to-edge" => Ok(Self::ClampToEdge),
            "repeat" => Ok(Self::Repeat),
            "mirror" | "mirrored_repeat" | "mirrored-repeat" => Ok(Self::MirroredRepeat),
            other => bail!(
                "invalid texture wrap '{other}' (expected clamp_to_edge, repeat, or mirrored_repeat)"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextureSpec {
    pub path: PathBuf,
    #[serde(default)]
    pub filter: TextureFilter,
    #[serde(default)]
    pub wrap: TextureWrap,
    #[serde(default)]
    pub flip_y: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectedUniform {
    pub name: String,
    #[serde(rename = "type")]
    pub utype: UniformType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflectedSampler {
    pub name: String,
    pub has_size_uniform: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderSchema {
    pub mode: String,
    pub uses_time: bool,
    pub uses_progress: bool,
    pub uses_ichannel0: bool,
    pub uniforms: Vec<ReflectedUniform>,
    pub samplers: Vec<ReflectedSampler>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cli_scalar_and_vector_uniforms() {
        assert_eq!(
            UniformValue::parse_cli("u_flag=true").unwrap(),
            ("u_flag".into(), UniformValue::Bool(true))
        );
        assert_eq!(
            UniformValue::parse_cli("u_mode=3").unwrap(),
            ("u_mode".into(), UniformValue::Int(3))
        );
        assert_eq!(
            UniformValue::parse_cli("u_strength=0.125").unwrap(),
            ("u_strength".into(), UniformValue::Float(0.125))
        );
        assert_eq!(
            UniformValue::parse_cli("u_pointer=0.25, 0.75").unwrap(),
            ("u_pointer".into(), UniformValue::Vec2([0.25, 0.75]))
        );
        assert_eq!(
            UniformValue::parse_cli("u_rgb=[1, 0.5, 0.0]").unwrap(),
            ("u_rgb".into(), UniformValue::Vec3([1.0, 0.5, 0.0]))
        );
        assert_eq!(
            UniformValue::parse_cli("u_tint=0.1,0.2,0.3,0.4").unwrap(),
            ("u_tint".into(), UniformValue::Vec4([0.1, 0.2, 0.3, 0.4]))
        );
    }

    #[test]
    fn parse_cli_rejects_bad_inputs() {
        assert!(UniformValue::parse_cli("no_equals").is_err());
        assert!(UniformValue::parse_cli("=1.0").is_err());
        assert!(UniformValue::parse_cli("u_nan=NaN").is_err());
        assert!(UniformValue::parse_cli("u_bad=1,2,3,4,5").is_err());
    }

    #[test]
    fn serde_roundtrip_untagged_uniforms_and_texture_spec() {
        let map: std::collections::BTreeMap<String, UniformValue> =
            serde_json::from_str(r#"{"u_flag":true,"u_int":2,"u_float":0.5,"u_vec2":[0.1,0.9]}"#)
                .unwrap();
        assert_eq!(map["u_flag"], UniformValue::Bool(true));
        assert_eq!(map["u_int"], UniformValue::Int(2));
        assert_eq!(map["u_float"], UniformValue::Float(0.5));
        assert_eq!(map["u_vec2"], UniformValue::Vec2([0.1, 0.9]));

        let spec: TextureSpec = serde_json::from_str(
            r#"{"path":"/tmp/depth.png","filter":"nearest","wrap":"repeat","flip_y":true}"#,
        )
        .unwrap();
        assert_eq!(spec.filter, TextureFilter::Nearest);
        assert_eq!(spec.wrap, TextureWrap::Repeat);
        assert!(spec.flip_y);
    }
}
