// Depth-driven 2.5D parallax shader showcasing v0.2.0 generic named samplers
// and reflected typed uniforms.
//
// Example usage:
//   glwall apply \
//     --media wallpaper.png \
//     --shader shaders/parallax_depth.frag \
//     --texture u_depth=depth.png:linear:clamp_to_edge \
//     --uniform u_pointer=0.5,0.5 \
//     --uniform u_parallaxStrength=0.04 \
//     --wait

uniform sampler2D u_depth;
uniform vec2 u_depth_size;

uniform vec2 u_pointer;
uniform float u_parallaxStrength;

vec4 wallpaper(vec2 uv) {
    float depth = sampleTextureFitted(u_depth, u_depth_size, uv).r;
    vec2 pointer = clamp(u_pointer, vec2(0.0), vec2(1.0));
    vec2 offset = (pointer - vec2(0.5)) * (depth - 0.5) * u_parallaxStrength;
    vec2 displaced = clamp(uv - offset, vec2(0.0), vec2(1.0));
    return sampleWallpaper(displaced);
}
