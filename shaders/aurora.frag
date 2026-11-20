// Smooth animated procedural aurora/mesh gradient shader.
// Optional u_params:
//   x = animation speed multiplier (default ~0.35 when 0.0)
//   y = contrast/saturation boost
vec4 wallpaper(vec2 uv) {
    float speed = (u_params.x > 0.001) ? u_params.x : 0.35;
    float t = u_time * speed + float(u_output_index) * 1.7;

    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / max(u_resolution.y, 1.0);

    float w1 = sin(p.x * 1.4 + t * 0.7) * cos(p.y * 1.2 - t * 0.5);
    float w2 = cos((p.x + p.y) * 1.1 - t * 0.4) * sin(p.y * 2.1 + t * 0.6);
    float w3 = sin(length(p - vec2(sin(t * 0.25) * 0.4, cos(t * 0.2) * 0.3)) * 2.3 - t * 0.8);

    vec3 deep = vec3(0.04, 0.07, 0.14);
    vec3 teal = vec3(0.07, 0.42, 0.52);
    vec3 indigo = vec3(0.24, 0.16, 0.56);
    vec3 glow = vec3(0.28, 0.78, 0.64);

    vec3 col = mix(deep, teal, smoothstep(-0.7, 0.8, w1));
    col = mix(col, indigo, smoothstep(-0.4, 0.9, w2) * 0.65);
    col += glow * pow(max(0.0, 0.5 + 0.5 * w3), 4.0) * 0.35;

    float boost = 1.0 + max(u_params.y, 0.0) * 0.4;
    return vec4(clamp(col * boost, 0.0, 1.0), 1.0);
}
