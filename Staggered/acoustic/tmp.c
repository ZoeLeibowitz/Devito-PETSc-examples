/* Devito generated code for Operator `Kernel` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  int * size;
  unsigned long nbytes;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct profiler
{
  double section0;
  double section1;
} ;


int Kernel(const float h_x, const float h_z, struct dataobj *restrict p_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict v_x_vec, struct dataobj *restrict v_z_vec, const int x_M, const int x_m, const int z_M, const int z_m, const float dt, const float o_x, const float o_z, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers)
{
  float (*restrict p)[p_vec->size[1]][p_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[p_vec->size[1]][p_vec->size[2]]) p_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict v_x)[v_x_vec->size[1]][v_x_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_x_vec->size[1]][v_x_vec->size[2]]) v_x_vec->data;
  float (*restrict v_z)[v_z_vec->size[1]][v_z_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_z_vec->size[1]][v_z_vec->size[2]]) v_z_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r1 = 1.0F/h_x;
  float r2 = 1.0F/dt;
  float r3 = 1.0F/h_z;

  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    START(section0)
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(p,v_x,v_z:32)
      for (int z = z_m; z <= z_M; z += 1)
      {
        v_x[t1][x + 2][z + 2] = dt*(r2*v_x[t0][x + 2][z + 2] - (-r1*p[t0][x + 2][z + 2] + r1*p[t0][x + 3][z + 2]));
        v_z[t1][x + 2][z + 2] = dt*(r2*v_z[t0][x + 2][z + 2] - (-r3*p[t0][x + 2][z + 2] + r3*p[t0][x + 2][z + 3]));
      }
    }
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(p,v_x,v_z:32)
      for (int z = z_m; z <= z_M; z += 1)
      {
        p[t1][x + 2][z + 2] = dt*(r2*p[t0][x + 2][z + 2] + (-1.6e+1F)*(-r1*v_x[t1][x + 1][z + 2] + r1*v_x[t1][x + 2][z + 2] - r3*v_z[t1][x + 2][z + 1] + r3*v_z[t1][x + 2][z + 2]));
      }
    }
    STOP(section0,timers)

    START(section1)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      for (int rsrcx = 0; rsrcx <= 1; rsrcx += 1)
      {
        for (int rsrcz = 0; rsrcz <= 1; rsrcz += 1)
        {
          int posx = (int)(floorf((-o_x + src_coords[p_src][0])/h_x));
          int posz = (int)(floorf((-o_z + src_coords[p_src][1])/h_z));
          float px = -floorf((-o_x + src_coords[p_src][0])/h_x) + (-o_x + src_coords[p_src][0])/h_x;
          float pz = -floorf((-o_z + src_coords[p_src][1])/h_z) + (-o_z + src_coords[p_src][1])/h_z;
          if (rsrcx + posx >= x_m - 1 && rsrcz + posz >= z_m - 1 && rsrcx + posx <= x_M + 1 && rsrcz + posz <= z_M + 1)
          {
            float r0 = (rsrcx*px + (1 - rsrcx)*(1 - px))*(rsrcz*pz + (1 - rsrcz)*(1 - pz))*src[time][p_src];
            p[t1][rsrcx + posx + 2][rsrcz + posz + 2] += r0;
          }
        }
      }
    }
    STOP(section1,timers)
  }

  return 0;
}

170
