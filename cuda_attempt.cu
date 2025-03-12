extern "C" __global__ void ray_aabb_intersect_top16(
        const float* ray_ori, const float* ray_dir,
        const float* min_corners, const float* max_corners,
        int* out_indices, int num_rays, int num_boxes);

__global__ void get_depth_values(
        const float* ray_ori, 
        const float* ray_dir,
        const float* min_corners, const float* max_corners,
        float* out_depth, int num_rays, int num_boxes
    );


__global__ void ray_aabb_intersect_top16(
    const float* ray_ori,    // [R, 3] ray origins
    const float* ray_dir,    // [R, 3] ray directions
    const float* min_corners, // [B, 3] AABB min corners
    const float* max_corners, // [B, 3] AABB max corners
    int* out_indices,        // [R, 32] output indices
    int num_rays,            // Number of rays
    int num_boxes            // Number of boxes
) {
    // Get ray index
    const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;
    
    // Load ray origin and direction
    const float ox = ray_ori[ray_idx * 3];
    const float oy = ray_ori[ray_idx * 3 + 1];
    const float oz = ray_ori[ray_idx * 3 + 2];
    
    const float dx = ray_dir[ray_idx * 3];
    const float dy = ray_dir[ray_idx * 3 + 1];
    const float dz = ray_dir[ray_idx * 3 + 2];
    
    // Compute reciprocal of direction for fast intersection
    const float rdx = 1.0f / dx;
    const float rdy = 1.0f / dy;
    const float rdz = 1.0f / dz;
    
    // Arrays to hold top 32 intersections
    float top_t[32];
    int top_idx[32];
    
    // Initialize with large values and -1 indices
    for (int i = 0; i < 32; i++) {
        top_t[i] = 1e10f;
        top_idx[i] = -1;
    }
    
    // Check each box
    for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
        // Load box corners
        const float bmin_x = min_corners[box_idx * 3];
        const float bmin_y = min_corners[box_idx * 3 + 1];
        const float bmin_z = min_corners[box_idx * 3 + 2];
        
        const float bmax_x = max_corners[box_idx * 3];
        const float bmax_y = max_corners[box_idx * 3 + 1];
        const float bmax_z = max_corners[box_idx * 3 + 2];
        
        // Compute t values for each pair of slabs
        const float tx1 = (bmin_x - ox) * rdx;
        const float tx2 = (bmax_x - ox) * rdx;
        float tmin = fminf(tx1, tx2);
        float tmax = fmaxf(tx1, tx2);
        
        const float ty1 = (bmin_y - oy) * rdy;
        const float ty2 = (bmax_y - oy) * rdy;
        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));
        
        const float tz1 = (bmin_z - oz) * rdz;
        const float tz2 = (bmax_z - oz) * rdz;
        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));
        
        // Check if we have a valid intersection
        if (tmax >= tmin && tmax > 0) {
            // Use tmin if ray origin is outside the box, otherwise tmax
            const float t_hit = (tmin > 0) ? tmin : tmax;
            
            // Check if this hit is among the 32 closest
            if (t_hit < top_t[31]) {
                // Find insertion position
                int insert_pos = 31;
                while (insert_pos > 0 && t_hit < top_t[insert_pos - 1]) {
                    insert_pos--;
                }
                
                // Shift elements to make room
                for (int j = 31; j > insert_pos; j--) {
                    top_t[j] = top_t[j - 1];
                    top_idx[j] = top_idx[j - 1];
                }
                
                // Insert new hit
                top_t[insert_pos] = t_hit;
                top_idx[insert_pos] = box_idx;
            }
        }
    }
    
    // Write output indices
    for (int i = 0; i < 32; i++) {
        out_indices[ray_idx * 32 + i] = top_idx[i];
    }
}
