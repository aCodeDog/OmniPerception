
#define __MATH_DECLARING_DOUBLE  1
#define __MATH_DECLARING_FLOATN  0
# define __USE_GNU	1
#define __GLIBC_USE(F)	__GLIBC_USE_ ## F

# if !__MATH_DECLARING_FLOATN || defined __USE_GNU || !__GLIBC_USE (ISOC2X)
/* Return maximum numeric value from X and Y.  */
__MATHCALLX (fmax,, (_Mdouble_ __x, _Mdouble_ __y), (__const__));

/* Return minimum numeric value from X and Y.  */
__MATHCALLX (fmin,, (_Mdouble_ __x, _Mdouble_ __y), (__const__));
# endif

__MATHCALL (fma,, (_Mdouble_ __x, _Mdouble_ __y, _Mdouble_ __z));

template<typename Type>
inline CUDA_CALLABLE vec_t<3,Type> cross(vec_t<3,Type> a, vec_t<3,Type> b)
{
    return {
        Type(a[1]*b[2] - a[2]*b[1]),
        Type(a[2]*b[0] - a[0]*b[2]),
        Type(a[0]*b[1] - a[1]*b[0])
    };
}



CUDA_CALLABLE inline float xorf(float x, int y)
{
	return __int_as_float(__float_as_int(x) ^ y);
}

inline float __int_as_float(int i)
{
	return *(float*)(&i);
}

inline int __float_as_int(float f)
{
	return *(int*)(&f);
}
CUDA_CALLABLE inline float xorf(float x, int y)
{
	return __int_as_float(__float_as_int(x) ^ y);
}

CUDA_CALLABLE inline int sign_mask(float x)
{
	return __float_as_int(x) & 0x80000000;
}

CUDA_CALLABLE inline float diff_product(float a, float b, float c, float d) 
{
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}

CUDA_CALLABLE inline int sign_mask(float x)
{
	return __float_as_int(x) & 0x80000000;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> normalize(vec_t<3, Type> a)
{
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
    if (l > Type(kEps))
        return vec_t<3, Type>(a.c[0]/l,a.c[1]/l,a.c[2]/l);
    else
        return vec_t<3, Type>();
}


template<unsigned Length, typename Type>
CUDA_CALLABLE inline int longest_axis(const vec_t<Length, Type>& v)
{
    Type lmax = abs(v[0]);
    int ret(0);
    for( unsigned i=1; i < Length; ++i )
    {
        Type l = abs(v[i]);
        if( l > lmax )
        {
            ret = i;
            lmax = l;
        }
    }
    return ret;
}



CUDA_CALLABLE inline int max_dim(vec3 a)
{
	float x = abs(a[0]);
	float y = abs(a[1]);
	float z = abs(a[2]);

	return longest_axis(vec3(x, y, z));
}

// http://jcgt.org/published/0002/01/05/
CUDA_CALLABLE inline bool intersect_ray_tri_woop(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& sign, vec3* normal)
{
	// todo: precompute for ray

	int kz = max_dim(dir);
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (dir[kz] < 0.0f)
	{
		float tmp = kx;
		kx = ky;
		ky = tmp;
	}

	float Sx = dir[kx]/dir[kz];
	float Sy = dir[ky]/dir[kz];
	float Sz = 1.0f/dir[kz];

	// todo: end precompute

	const vec3 A = a-p;
	const vec3 B = b-p;
	const vec3 C = c-p;
	
	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];
		
    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

	if (U == 0.0f || V == 0.0f || W == 0.0f) 
	{
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
    {
        return false;
    }

	float det = U+V+W;

	if (det == 0.0f) 
    {
		return false;
    }

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
		return false;
    }

	const float rcpDet = 1.0f/det;
	u = U*rcpDet;
	v = V*rcpDet;
	t = T*rcpDet;
	sign = det;
	
	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
	{
		const vec3 ab = b-a;
		const vec3 ac = c-a;

		// calculate normal
		*normal = cross(ab, ac); 
	}

	return true;
}




CUDA_CALLABLE inline bool intersect_ray_aabb(const vec3& pos, const vec3& rcp_dir, const vec3& lower, const vec3& upper, float& t)
{
	float l1, l2, lmin, lmax;

    l1 = (lower[0] - pos[0]) * rcp_dir[0];
    l2 = (upper[0] - pos[0]) * rcp_dir[0];
    lmin = min(l1,l2);
    lmax = max(l1,l2);

    l1 = (lower[1] - pos[1]) * rcp_dir[1];
    l2 = (upper[1] - pos[1]) * rcp_dir[1];
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    l1 = (lower[2] - pos[2]) * rcp_dir[2];
    l2 = (upper[2] - pos[2]) * rcp_dir[2];
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
}



CUDA_CALLABLE inline bool mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, float& t, float& u, float& v, float& sign, vec3& normal, int& face)
{
    Mesh mesh = mesh_get(id);

    int stack[32];
    stack[0] = *mesh.bvh.root;
    int count = 1;

    vec3 rcp_dir = vec3(1.0f/dir[0], 1.0f/dir[1], 1.0f/dir[2]);

    float min_t = max_t;
    int min_face;
    float min_u;
    float min_v;
    float min_sign = 1.0f;
    vec3 min_normal;

    while (count)
    {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
        BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];

        // todo: switch to robust ray-aabb, or expand bounds in build stage
        float eps = 1.e-3f;
        float t = 0.0f;
        bool hit = intersect_ray_aabb(start, rcp_dir, vec3(lower.x-eps, lower.y-eps, lower.z-eps), vec3(upper.x+eps, upper.y+eps, upper.z+eps), t);

        if (hit && t < min_t)
        {
            const int left_index = lower.i;
            const int right_index = upper.i;

            if (lower.b)
            {	
                // compute closest point on tri
                int i = mesh.indices[left_index*3+0];
                int j = mesh.indices[left_index*3+1];
                int k = mesh.indices[left_index*3+2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                float t, u, v, sign;
                vec3 n;
                
                if (intersect_ray_tri_woop(start, dir, p, q, r, t, u, v, sign, &n))
                {
                    if (t < min_t && t >= 0.0f)
                    {
                        min_t = t;
                        min_face = left_index;
                        min_u = u;
                        min_v = v;
                        min_sign = sign;
                        min_normal = n;
                    }
                }
            }
            else
            {
                stack[count++] = left_index;
                stack[count++] = right_index;
            }
        }
    }

    if (min_t < max_t)
    {
        // write outputs
        u = min_u;
        v = min_v;
        sign = min_sign;
        t = min_t;
        normal = normalize(min_normal);
        face = min_face;

        return true;
    }
    else
    {
        return false;
    }
    
}


@staticmethod
@wp.kernel
def draw_optimized_kernel_pointcloud(
    mesh_ids: wp.array(dtype=wp.uint64),
    lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
    lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
    ray_vectors: wp.array2d(dtype=wp.vec3),
    # ray_noise_magnitude: wp.array(dtype=float),
    far_plane: float,
    pixels: wp.array(dtype=wp.vec3, ndim=4),
    local_dist: wp.array(dtype=wp.float32, ndim=4),
    pointcloud_in_world_frame: bool,
):

    env_id, cam_id, scan_line, point_index = wp.tid()
    mesh = mesh_ids[0]
    lidar_position = lidar_pos_array[env_id, cam_id]
    # if env_id == 1 :
    #     wp.print(lidar_position)
    lidar_quaternion = lidar_quat_array[env_id, cam_id]
    ray_origin = lidar_position
    # perturb ray_vectors with uniform noise
    ray_dir = ray_vectors[scan_line, point_index]  # + sampled_vec3_noise
    ray_dir = wp.normalize(ray_dir)
    ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    dist = NO_HIT_RAY_VAL
    query = wp.mesh_query_ray(mesh,ray_origin, ray_direction_world, far_plane)
    if query.result:
        dist = query.t
        local_dist[env_id, cam_id, scan_line, point_index] = dist
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
        else:
            pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir
        #wp.print(dist)
