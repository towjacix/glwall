vec4 wallpaper(vec2 uv) {
    float p = clamp(u_progress, 0.0, 1.0);
    if (p <= 0.0) {
        return samplePrevWallpaper(uv);
    }
    if (p >= 1.0) {
        return sampleWallpaper(uv);
    }
    float diag = (uv.x * 0.72 + (1.0 - uv.y) * 0.28);
    float edge = 0.08;
    float threshold = mix(-edge, 1.0 + edge, p);
    float mask = smoothstep(threshold - edge, threshold + edge, diag);
    vec4 fromColor = samplePrevWallpaper(uv);
    vec4 toColor = sampleWallpaper(uv);
    return mix(toColor, fromColor, mask);
}
