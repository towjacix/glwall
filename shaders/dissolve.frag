float glwall_cell_noise(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

vec4 wallpaper(vec2 uv) {
    float p = clamp(u_progress, 0.0, 1.0);
    if (p <= 0.0) {
        return samplePrevWallpaper(uv);
    }
    if (p >= 1.0) {
        return sampleWallpaper(uv);
    }
    vec4 fromColor = samplePrevWallpaper(uv);
    vec4 toColor = sampleWallpaper(uv);
    vec2 cell = floor(uv * max(u_resolution, vec2(1.0)) / 10.0);
    float threshold = glwall_cell_noise(cell);
    float reveal = smoothstep(threshold - 0.035, threshold + 0.035, p);
    return mix(fromColor, toColor, reveal);
}
