vec4 wallpaper(vec2 uv) {
    float p = clamp(u_progress, 0.0, 1.0);
    if (p <= 0.0) {
        return samplePrevWallpaper(uv);
    }
    if (p >= 1.0) {
        return sampleWallpaper(uv);
    }
    vec2 size = max(u_resolution, vec2(1.0));
    float pixelSize = mix(size.x / min(size.x, 24.0), 1.0, p);
    vec2 pixelUv = (floor(uv * size / pixelSize) + 0.5) * pixelSize / size;
    vec4 fromColor = samplePrevWallpaper(pixelUv);
    vec4 toColor = sampleWallpaper(uv);
    return mix(fromColor, toColor, p);
}
