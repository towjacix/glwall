use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub const BUILTIN_PASSTHROUGH: &str = include_str!("../../shaders/passthrough.frag");
pub const BUILTIN_FADE: &str = include_str!("../../shaders/fade.frag");
pub const BUILTIN_DISSOLVE: &str = include_str!("../../shaders/dissolve.frag");
pub const BUILTIN_PIXELATE: &str = include_str!("../../shaders/pixelate.frag");
pub const BUILTIN_WIPE: &str = include_str!("../../shaders/wipe.frag");

pub const VERTEX_SHADER_SRC: &str = r#"#version 300 es
out vec2 v_uv;

void main() {
    float x = float((gl_VertexID & 1) << 2);
    float y = float((gl_VertexID & 2) << 1);
    v_uv = vec2(x * 0.5, y * 0.5);
    gl_Position = vec4(x - 1.0, y - 1.0, 0.0, 1.0);
}
"#;

const FRAGMENT_PREAMBLE: &str = r#"#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 glwall_frag_color;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_delta;
uniform float u_progress;
uniform int u_frame;
uniform vec4 u_params;
uniform int u_output_index;

uniform sampler2D u_tex0_rgba;
uniform sampler2D u_tex0_y;
uniform sampler2D u_tex0_uv;
uniform int u_slot0_kind;
uniform vec2 u_slot0_size;
uniform int u_slot0_fit;
uniform int u_slot0_cs;

uniform sampler2D u_tex1_rgba;
uniform sampler2D u_tex1_y;
uniform sampler2D u_tex1_uv;
uniform int u_slot1_kind;
uniform vec2 u_slot1_size;
uniform int u_slot1_fit;
uniform int u_slot1_cs;

uniform sampler2D iChannel0;

bool glwall_fit_uv(vec2 uv, vec2 srcSize, vec2 dstSize, int fitMode, out vec2 outUv) {
    vec2 safeSrc = max(srcSize, vec2(1.0));
    vec2 safeDst = max(dstSize, vec2(1.0));
    vec2 rel = uv;

    if (fitMode == 0) {
        // Cover
        float sAspect = safeSrc.x / safeSrc.y;
        float dAspect = safeDst.x / safeDst.y;
        vec2 crop = sAspect > dAspect
            ? vec2(dAspect / sAspect, 1.0)
            : vec2(1.0, sAspect / dAspect);
        rel = vec2(0.5) + (uv - vec2(0.5)) * crop;
    } else if (fitMode == 1) {
        // Contain
        float sAspect = safeSrc.x / safeSrc.y;
        float dAspect = safeDst.x / safeDst.y;
        vec2 scale = sAspect > dAspect
            ? vec2(1.0, sAspect / dAspect)
            : vec2(dAspect / sAspect, 1.0);
        rel = vec2(0.5) + (uv - vec2(0.5)) * scale;
    } else if (fitMode == 3) {
        // Center (1:1 source pixels)
        vec2 ratio = safeDst / safeSrc;
        rel = vec2(0.5) + (uv - vec2(0.5)) * ratio;
    }
    // fitMode == 2 is Stretch (rel == uv)

    if (rel.x < 0.0 || rel.x > 1.0 || rel.y < 0.0 || rel.y > 1.0) {
        outUv = vec2(0.0);
        return false;
    }

    // Textures uploaded top-row-first place row 0 at v = 0, so flip Y for screen-up orientation.
    outUv = vec2(rel.x, 1.0 - rel.y);
    return true;
}

vec3 glwall_yuv_to_rgb(float yRaw, vec2 uvRaw, int cs) {
    float y;
    vec2 c;
    if (cs == 0 || cs == 2) {
        // Limited / TV range: Y in [16..235], UV in [16..240]
        y = max(0.0, (yRaw - (16.0 / 255.0)) * (255.0 / 219.0));
        c = (uvRaw - vec2(128.0 / 255.0)) * (255.0 / 224.0);
    } else {
        // Full / PC range
        y = yRaw;
        c = uvRaw - vec2(0.5);
    }

    vec3 rgb;
    if (cs == 0 || cs == 1) {
        // BT.709
        rgb.r = y + 1.5748 * c.y;
        rgb.g = y - 0.187324 * c.x - 0.468124 * c.y;
        rgb.b = y + 1.8556 * c.x;
    } else {
        // BT.601
        rgb.r = y + 1.402 * c.y;
        rgb.g = y - 0.344136 * c.x - 0.714136 * c.y;
        rgb.b = y + 1.772 * c.x;
    }
    return clamp(rgb, 0.0, 1.0);
}

vec4 glwall_sample_slot0(vec2 uv) {
    if (u_slot0_kind == 0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec2 texUv;
    if (!glwall_fit_uv(uv, u_slot0_size, u_resolution, u_slot0_fit, texUv)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    if (u_slot0_kind == 1) {
        return texture(u_tex0_rgba, texUv);
    }
    float y = texture(u_tex0_y, texUv).r;
    vec2 chroma = texture(u_tex0_uv, texUv).rg;
    return vec4(glwall_yuv_to_rgb(y, chroma, u_slot0_cs), 1.0);
}

vec4 glwall_sample_slot1(vec2 uv) {
    if (u_slot1_kind == 0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec2 texUv;
    if (!glwall_fit_uv(uv, u_slot1_size, u_resolution, u_slot1_fit, texUv)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    if (u_slot1_kind == 1) {
        return texture(u_tex1_rgba, texUv);
    }
    float y = texture(u_tex1_y, texUv).r;
    vec2 chroma = texture(u_tex1_uv, texUv).rg;
    return vec4(glwall_yuv_to_rgb(y, chroma, u_slot1_cs), 1.0);
}

vec4 sampleWallpaper(vec2 uv) {
    return glwall_sample_slot0(uv);
}

vec4 samplePrevWallpaper(vec2 uv) {
    return glwall_sample_slot1(uv);
}
"#;

const SHADERTOY_MAIN_WRAPPER: &str = r#"
void main() {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(color, gl_FragCoord.xy);
    glwall_frag_color = vec4(color.rgb, 1.0);
}
"#;

const GLWALL_MAIN_WRAPPER: &str = r#"
void main() {
    glwall_frag_color = wallpaper(v_uv);
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderMode {
    Glwall,
    Shadertoy,
}

#[derive(Debug, Clone)]
pub struct AssembledShader {
    pub mode: ShaderMode,
    pub uses_time: bool,
    pub uses_ichannel0: bool,
    pub fragment_source: String,
}

pub fn resolve_transition_source(name_or_path: &str) -> Result<Option<String>> {
    let trimmed = name_or_path.trim();
    match trimmed {
        "" | "fade" => Ok(Some(BUILTIN_FADE.to_string())),
        "dissolve" => Ok(Some(BUILTIN_DISSOLVE.to_string())),
        "pixelate" => Ok(Some(BUILTIN_PIXELATE.to_string())),
        "wipe" => Ok(Some(BUILTIN_WIPE.to_string())),
        "none" => Ok(None),
        other => {
            let p = Path::new(other);
            let src = fs::read_to_string(p)
                .with_context(|| format!("failed to read transition shader {}", p.display()))?;
            Ok(Some(src))
        }
    }
}

pub fn detect_shader_mode(source: &str) -> ShaderMode {
    let clean = strip_glsl_comments(source);
    if contains_main_image(&clean) {
        ShaderMode::Shadertoy
    } else {
        ShaderMode::Glwall
    }
}

pub fn shader_uses_time(source: &str) -> bool {
    let clean = strip_glsl_comments(source);
    const TIME_IDENTS: &[&str] = &[
        "u_time",
        "u_delta",
        "u_frame",
        "iTime",
        "iTimeDelta",
        "iFrame",
    ];
    TIME_IDENTS
        .iter()
        .any(|ident| contains_ident(&clean, ident))
}

pub fn shader_uses_ichannel0(source: &str) -> bool {
    let clean = strip_glsl_comments(source);
    contains_ident(&clean, "iChannel0")
}

pub fn assemble_fragment_shader(user_source: &str) -> AssembledShader {
    let mode = detect_shader_mode(user_source);
    let uses_time = shader_uses_time(user_source);
    let uses_ichannel0 = mode == ShaderMode::Shadertoy && shader_uses_ichannel0(user_source);

    let sanitized_user = strip_leading_version_directive(user_source);
    let mut out = String::with_capacity(
        FRAGMENT_PREAMBLE.len() + sanitized_user.len() + SHADERTOY_MAIN_WRAPPER.len() + 128,
    );
    out.push_str(FRAGMENT_PREAMBLE);

    if mode == ShaderMode::Shadertoy {
        out.push_str(
            "#define iResolution vec3(u_resolution, 1.0)\n\
             #define iTime u_time\n\
             #define iTimeDelta u_delta\n\
             #define iFrame u_frame\n\
             #define iMouse vec4(0.0)\n",
        );
    }

    out.push_str("#line 1\n");
    out.push_str(&sanitized_user);
    if !sanitized_user.ends_with('\n') {
        out.push('\n');
    }

    match mode {
        ShaderMode::Glwall => out.push_str(GLWALL_MAIN_WRAPPER),
        ShaderMode::Shadertoy => out.push_str(SHADERTOY_MAIN_WRAPPER),
    }

    AssembledShader {
        mode,
        uses_time,
        uses_ichannel0,
        fragment_source: out,
    }
}

fn strip_leading_version_directive(src: &str) -> String {
    src.lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#version ") || trimmed.starts_with("precision ") {
                String::new()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_glsl_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn contains_ident(haystack: &str, ident: &str) -> bool {
    let mut offset = 0;
    while let Some(pos) = haystack[offset..].find(ident) {
        let start = offset + pos;
        let end = start + ident.len();
        let prev_ok = start == 0 || !is_ident_char(haystack.as_bytes()[start - 1]);
        let next_ok = end == haystack.len() || !is_ident_char(haystack.as_bytes()[end]);
        if prev_ok && next_ok {
            return true;
        }
        offset = end;
    }
    false
}

fn contains_main_image(clean: &str) -> bool {
    let mut offset = 0;
    while let Some(pos) = clean[offset..].find("mainImage") {
        let start = offset + pos;
        let end = start + "mainImage".len();
        let prev_ok = start == 0 || !is_ident_char(clean.as_bytes()[start - 1]);
        if prev_ok {
            let rest = clean[end..].trim_start();
            if rest.starts_with('(') {
                return true;
            }
        }
        offset = end;
    }
    false
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_shader_mode_ignoring_comments() {
        let native = "// void mainImage(out vec4 c, in vec2 f)\nvec4 wallpaper(vec2 uv) { return sampleWallpaper(uv); }";
        assert_eq!(detect_shader_mode(native), ShaderMode::Glwall);

        let toy =
            "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }";
        assert_eq!(detect_shader_mode(toy), ShaderMode::Shadertoy);
    }

    #[test]
    fn detects_time_usage_accurately() {
        assert!(!shader_uses_time(BUILTIN_PASSTHROUGH));
        assert!(!shader_uses_time(
            "// u_time\nvec4 wallpaper(vec2 uv) { float u_timer = 1.0; return vec4(u_timer); }"
        ));
        assert!(shader_uses_time(
            "vec4 wallpaper(vec2 uv) { return vec4(sin(u_time)); }"
        ));
        assert!(shader_uses_time(
            "void mainImage(out vec4 c, in vec2 f) { c = vec4(sin(iTime)); }"
        ));
    }

    #[test]
    fn shadertoy_wrapper_detects_ichannel0_and_line_directive() {
        let toy = "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n  fragColor = texture(iChannel0, fragCoord / iResolution.xy);\n}\n";
        let assembled = assemble_fragment_shader(toy);
        assert_eq!(assembled.mode, ShaderMode::Shadertoy);
        assert!(assembled.uses_ichannel0);
        assert!(!assembled.uses_time);
        assert!(assembled.fragment_source.contains("#line 1\n"));
        assert!(
            assembled
                .fragment_source
                .contains("mainImage(color, gl_FragCoord.xy);")
        );
    }
}
