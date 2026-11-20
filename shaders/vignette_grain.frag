// Post-processing shader compatible with static images, GIF/WebP, and NV12 video.
// Tune live via: glwall params <vignette> <grain> <saturation> <0>
float glwall_hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec4 wallpaper(vec2 uv) {
    vec4 base = sampleWallpaper(uv);
    float vignetteStrength = (u_params.x > 0.0) ? u_params.x : 0.32;
    float grainStrength = (u_params.y > 0.0) ? u_params.y : 0.035;
    float sat = (u_params.z > 0.0) ? u_params.z : 1.08;

    float luma = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 graded = mix(vec3(luma), base.rgb, sat);

    vec2 centered = uv - 0.5;
    float vig = 1.0 - dot(centered, centered) * vignetteStrength * 1.8;
    graded *= clamp(vig, 0.2, 1.0);

    float noise = glwall_hash12(floor(uv * u_resolution) + fract(u_time * 37.0) * 256.0) - 0.5;
    graded += noise * grainStrength;

    return vec4(clamp(graded, 0.0, 1.0), base.a);
}
