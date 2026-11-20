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
    return mix(fromColor, toColor, p);
}
