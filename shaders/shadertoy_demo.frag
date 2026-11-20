// Shadertoy-compatible fragment shader example.
// Also supports `iChannel0` when paired with an image or video via `glwall shader`.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / max(iResolution.y, 1.0);
    float t = iTime * 0.4;

    vec3 col = 0.5 + 0.5 * cos(t + uv.xyx * 2.5 + vec3(0.0, 2.0, 4.0));
    float vignette = smoothstep(1.25, 0.15, length(uv));
    col *= mix(0.25, 1.0, vignette);

    fragColor = vec4(col, 1.0);
}
