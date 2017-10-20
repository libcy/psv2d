// * -PSM2D-
// *                    P and SV WAVES
// ************************************************************************
// * Calculating P and SV wavefields in homogeneous half-space for a      *
// * point source by the Finite-Difference Method.                        *
// * **********************************************************************
// * Last modified: May 14, 2017                                          *
// * Author: Yanbin WANG                                                  *
// *         Department of Earth and Planetary Sciences                   *
// *         Faculty of Sciences, Kyushu University                       *
// *         Hakozaki 6-10-1, Fukuoka, 812-8581, Japan                    *
// * Now at: Department of Geophysics, Peking University                  *
// *         100871, Beijing, China                                       *
// * Modified to staggered-grid scheme on 16 June 2005.                   *
// * Modified to PSM/FDM hybrid method in February 2006                   *
// *                        by Xing Wei and Yanbin Wang.                  *
// * Modified for Lanzhou basin on 11 January 2011.                       *
// *                        by Yanbin Wang.                               *
// * Modified to Finite-Difference Method on March, 2016                  *
// *                        by Xueyan Li and Yanbin Wang.                 *
// * Modified to Cuda C on March, 2017                                    *
// *                        by Congyue Cui and Yanbin Wang.               *
// ************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// map block and thread index to i and j
#define devij(dimx, dimy) \
int i = blockIdx.x % dimx; \
int j = threadIdx.x + (blockIdx.x - i) / dimx * dimy / d_nbt; \
int ij = i * dimy + j

// parameter
const int nx = 2048, ny = 1024, nx2 = nx * 2, ny2 = ny * 2;
const float dx = 0.0342, dy = 0.0342, dt = 1.0e-3;
const int ntmax = 30000, nwrite = 500;
const float at = 0.1 / 4.0, t0 = at * 2;
const int na = 0;
const int nst = 512, nsskip = nx / nst;
const int nxa = 20, nya = 20;
const int nskip = 10, ntskp = ntmax / nskip + 1;
const int nbt = 8;

// plotting parameter
const int nsnap=60;
const float pamp=0.5,samp=2.2;
const float pampall=0.5,sampall=2.2;

// device parameter
__constant__ int is0 = 292, js0 = 146;
__constant__ float ax = 0.0342, ay = 0.0342;
__constant__ float fxx = 0.0, fyy = 0.0, fzz = 0.0;
__constant__ float dpxx = 0.0, dpyy = 0.0, dpzz = 0.0;
__constant__ float rmxx = 1.0, rmxy = 0.0, rmyy = -1.0, rmyx=0.0;
__constant__ float c0 = 9.0 / 8.0, c1 = 1.0 / 24.0;
__constant__ int d_nbt = 8;

// matrix related function
namespace mat{
    float *create(const int m) {
        // create floating-point device array
    	float *a;
    	cudaMalloc((void**)&a, m * sizeof(float));
    	return a;
    }
    float *create_h(const int m) {
        // create floating-point host array
    	return (float *)malloc(m * sizeof(float));
    }
    int *create_i(const int m){
        // create integer device array
        int *a;
    	cudaMalloc((void**)&a, m * sizeof(int));
    	return a;
    }

    __global__ void set_d(float *a, const float init, const int m, const int n){
        devij(m, n);
        a[ij] = init;
    }
    void set(float *a, const float init, const int m, const int n){
        // initialize the value of a device matrix
        mat::set_d<<<m * nbt, n / nbt>>>(a, init, m, n);
    }
    void copyhd(float *d_a, const float *a, const int m){
        // copy memory from host(a) to device(d_a)
        cudaMemcpy(d_a, a , m * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copydh(float *a, const float *d_a, const int m){
        // copy memory from device(d_a) to host(a)
        cudaMemcpy(a, d_a , m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void write(FILE *file, float *d_a, float *a, const int nx, const int ny){
        // write matrix data to file
        mat::copydh(a, d_a, nx * ny);
    	for(int i= 0; i < nx; i++){
    		for(int j = 0; j < ny; j++){
    			fprintf(file,"%f\n", a[i * ny + j]);
    		}
    	}
    }
    void read(FILE *file, float *a, const int nx, const int ny){
        // read matrix data from file
        for(int i = 0; i < nx; i++){
            for(int j = 0; j < ny; j++){
                int ij = i * ny + j;
                fscanf(file, "%f", a + ij);
    		}
    	}
    }
}

// forward related function
namespace psv{
    __device__ float dherrman(float a, float x, float x0){
    	float a2 = 2.0 * a;
    	float t = x - x0;
    	float td = (t + a2) / a;
    	if(t <= -a2) return 0.0;
    	if(t <= -a) return td / (a2 * a);
    	if(t <= a) return (-td + 2.0) / (a2 * a);
    	if(t <= a2) return (td - 4.0) / (a2 * a);
    	return 0.0;
    }
    __device__ float herrman(float a, float x, float x0){
    	float a2 = 2.0*a;
    	float t = x - x0;
    	float td = (t + a2)/a;
    	if(t <= -a2) return 0.0;
    	if(t <= -a) return (0.5 * td * td) / a2;
    	if(t <= a) return (-0.5 * td * td + 2.0 * td - 1.0) / a2;
    	if(t <= a2) return (0.5 * td * td - 4.0 * td + 8.0) / a2;
    	return 0.0;
    }
    __device__ float fxmxz(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at,float xs,float zs){
    	float x0 = i0*dx+xs;
    	float z0 = j0*dz+zs;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = psv::herrman(ax,x,x0);
    	float fhz = -psv::dherrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float fzmxz(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at,float xs,float zs){
    	float x0 = i0*dx+xs;
    	float z0 = j0*dz+zs;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = -psv::dherrman(ax,x,x0);
    	float fhz = psv::herrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float fzmzz(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at,float xs,float zs){
    	float x0 = i0*dx+xs;
    	float z0 = j0*dz+zs;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = psv::herrman(ax,x,x0);
    	float fhz = -psv::dherrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float fxmxx(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at,float xs,float zs){
    	float x0 = i0*dx+xs;
    	float z0 = j0*dz+zs;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = -psv::dherrman(ax,x,x0);
    	float fhz = psv::herrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float fx(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at){
    	float x0 = i0*dx;
    	float z0 = j0*dz;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = psv::herrman(ax,x,x0);
    	float fhz = psv::herrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float fz(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at){
    	float x0 = i0*dx;
    	float z0 = j0*dz;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = psv::herrman(ax,x,x0);
    	float fhz = psv::herrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float exforce(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at){
    	float x0 = i0*dx;
    	float z0 = j0*dz;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = -psv::dherrman(ax,x,x0);
    	float fhz = psv::herrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }
    __device__ float ezforce(int i, int j, int i0, int j0, float dx, float dz, float ax, float az, float t, float t0, float at){
    	float x0 = i0*dx;
    	float z0 = j0*dz;
    	float x = (i+1)*dx;
    	float z = (j+1)*dz;
    	float fhx = psv::herrman(ax,x,x0);
    	float fhz = -psv::dherrman(az,z,z0);
    	float fht = psv::herrman(at,t,t0);
    	return fhx*fhz*fht;
    }

    __global__ void istxy(int *istx, int *isty, const int na){
        int i = blockIdx.x;
        istx[i] = i * 4 + 1;
		isty[i] = na + 1;
    }
    __global__ void rdl(float *rig, float *den, float *lam,
        const float *nd, const float *q1d,
        const int nx2, const int ny2, const float dy){
        devij(nx2, ny2);

        float depth = j * dy / 2.0;
        float vpb, vsb, rob;
        float rrigb, rlanb, rdenb;

        if(depth <= -q1d[i] / 1000.0){
            vpb = 1.70;
            vsb = 0.85;
            rob = 1.8;
        }
        else if(depth <= -nd[i] / 1000.0){
            vpb = 4.0;
            vsb = 2.1;
            rob = 2.4;
        }
        else if(depth <= 15.0){
            vpb = 5.8;
            vsb = 3.3;
            rob = 2.7;
        }
        else if(depth <= 32.0){
            vpb = 6.4;
            vsb = 3.6;
            rob = 2.85;
        }
        else{
            vpb = 6.9;
            vsb = 3.9;
            rob = 3.1;
        }

        rrigb = rob * vsb * vsb;
        rlanb = rob * vpb * vpb - 2.0 * rrigb;
        rdenb = rob;

        if(j < na * 2){
            rig[ij] = 0.0;
            den[ij] = rdenb;
            lam[ij] = 0.0;
        }
        else{
            rig[ij] = rrigb;
            den[ij] = rdenb;
            lam[ij] = rlanb;
        }
    }
    __global__ void gg(float *ggg, const float apara,
        const int nx, const int ny, const int nxa, const int nya){
        devij(nx, ny);

        if(i + 1 < nxa){
			ggg[ij]=exp(-pow(apara * (nxa - i - 1), 2));
		}
		else if(i + 1 > (nx - nxa + 1)){
			ggg[ij]=exp(-pow(apara * (i - nx + nxa), 2));
		}
		else if(j + 1 > (ny - nya + 1)){
			ggg[ij]=exp(-pow(apara * (j - ny + nya), 2));
		}
		else{
			ggg[ij]=1.0;
		}
    }
    __global__ void finidyy(float *a, float *dya,
        const int nx, const int ny, const float dx, const float dy, const float dt){
        devij(nx, ny);
        float *ai = a + i * ny;

        if(j == 0){
			dya[ij] = 1.0 / dy * (c0 * ai[0] - c1 * ai[1]);
		}
		else if(j == 1){
			dya[ij] = 1.0 / dy * (c0 * (ai[1] - ai[0]) - c1 * ai[2]);
		}
		else if(j == ny - 1){
			dya[ij] = 1.0 / dy * (c0 * (ai[ny - 1] - ai[ny - 2]) + c1 * ai[ny - 3]);
		}
		else{
			dya[ij] = 1.0 / dy * (c0 * (ai[j] - ai[j - 1]) - c1 * (ai[j + 1] - ai[j - 2]));
		}
    }
    __global__ void finidyx(float *a, float *dya,
        const int nx, const int ny, const float dx, const float dy, const float dt){
        devij(nx, ny);
        float *ai = a + i * ny;

        if(j == 0){
			dya[ij] = 1.0 / dy * (c0 * (ai[1] - ai[0]) - c1 * ai[2]);
		}
		else if(j == ny - 2){
			dya[ij] = 1.0 / dy * (c0 * (ai[ny - 1] - ai[ny - 2]) + c1 * ai[ny - 3]);
		}
		else if(j == ny - 1){
			dya[ij] = 1.0 / dy * (c0 * (-ai[ny - 1]) + c1 * ai[ny - 2]);
		}
		else{
			dya[ij] = 1.0 / dy * (c0 * (ai[j + 1] - ai[j]) - c1 * (ai[j + 2] - ai[j - 1]));
		}
    }
    __global__ void finidxy(float *a, float *dya,
        const int nx, const int ny, const float dx, const float dy, const float dt){
        devij(nx, ny);
        float *aj = a + j;

        if(i == 0){
			dya[ij] = 1.0 / dx * (c0 * aj[0] - c1 * aj[ny]);
		}
		else if(i == 1){
			dya[ij] = 1.0 / dx * (c0 * (aj[ny] - aj[0]) - c1 * aj[2 * ny]);
		}
		else if(i == nx - 1){
			dya[ij] = 1.0 / dx * (c0 * (aj[(nx - 1) * ny] - aj[(nx - 2) * ny]) + c1 * aj[(nx - 3) * ny]);
		}
		else{
			dya[ij] = 1.0 / dx * (c0 * (aj[i * ny] - aj[(i - 1) * ny]) - c1 * (aj[(i + 1) * ny] - aj[(i - 2) * ny]));
		}
    }
    __global__ void finidxx(float *a, float *dya,
        const int nx, const int ny, const float dx, const float dy, const float dt){
        devij(nx, ny);
        float *aj = a + j;

        if(i == 0){
			dya[ij] = 1.0 / dx * (c0 * (aj[ny] - aj[0]) - c1 * aj[2 * ny]);
		}
		else if(i == nx - 2){
			dya[ij] = 1.0 / dx * (c0 * (aj[(nx - 1) * ny] - aj[(nx - 2) * ny]) + c1 * aj[(nx - 3) * ny]);
		}
		else if(i == nx - 1){
			dya[ij] = 1.0 / dx * (c0 * (-aj[(nx - 1) * ny ]) + c1 * aj[(nx - 2) * ny]);
		}
		else{
			dya[ij] = 1.0 / dx * (c0 * (aj[(i + 1) * ny] - aj[i * ny]) - c1 * (aj[(i + 2) * ny] - aj[(i - 1) * ny]));
		}
    }
    __global__ void sxy(float *sxx, float *syy, float *sxy,
        const float *lam, const float *rig, const float *ggg,
    	const float *dxvx, const float *dxvy, const float *dyvx, const float *dyvy,
        const int nx, const int ny, const float dt){
        devij(nx, ny);

        float ram1 = lam[(i * 2 + 1) * ny + j * 2];
    	float rig1 = rig[(i * 2 + 1) * ny + j * 2];
    	float rig2 = rig[i * 2 * ny + j * 2 + 1];
    	float gg = ggg[ij];

    	float sxxt1ij = (ram1 + 2.0 * rig1) * dxvx[ij] + ram1 * dyvy[ij];
    	float syyt1ij = (ram1 + 2.0 * rig1) * dyvy[ij] + ram1 * dxvx[ij];
    	float sxyt1ij = rig2 * (dxvy[ij] + dyvx[ij]);

    	sxx[ij] = sxx[ij] * gg + dt * sxxt1ij;
    	syy[ij] = syy[ij] * gg + dt * syyt1ij;
    	sxy[ij] = sxy[ij] * gg + dt * sxyt1ij;

    	if(j == na) syy[i * ny + na] = 0.0;
    }
    __global__ void vxyuxy(float *vx, float *vy, float *ux, float *uy,
        const float *dxsxx, const float *dxsxy, const float *dysxy, const float *dysyy,
    	const float *ggg, const float *den, const float t, const float ftmax,
        const int nx, const int ny, const float dx, const float dy, const float dt,
        const float t0, const float at){
        devij(nx, ny);

    	float gg = ggg[ij];
    	float denvx = den[i * 2 * ny + j * 2];
    	float denvy = den[(i * 2 + 1) * ny + j * 2 + 1];

    	float fx1,fy1;
    	if(t < ftmax){
    		fx1 = rmxx * psv::fxmxx(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at, 0.0, 0.0) +
    			rmxy * psv::fxmxz(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at, 0.0, 0.0) +
    			fxx * psv::fx(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at) +
    			dpxx * psv::exforce(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at);
    		fy1 = rmyx * psv::fzmxz(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at, -dx/2, -dy/2) +
    			rmyy * psv::fzmzz(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at, -dx/2, -dy/2) +
    			fzz * psv::fz(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at) +
    			dpzz * psv::ezforce(i, j ,is0, js0, dx, dy, ax, ay, t, t0, at);
    	}
    	else{
    		fx1 = 0.0;
    		fy1 = 0.0;
    	}

    	float uxt2ij = (dxsxx[ij] + dysxy[ij] + fx1) / denvx;
    	float uyt2ij = (dxsxy[ij] + dysyy[ij] + fy1) / denvy;

    	vx[ij] = vx[ij] * gg + dt * uxt2ij;
    	vy[ij] = vy[ij] * gg + dt * uyt2ij;

    	ux[ij] = ux[ij] * gg + dt * vx[ij];
    	uy[ij] = uy[ij] * gg + dt * vy[ij];
    }
    __global__ void uxyall(float *uxall, float *uyall, const float *ux, const float *uy,
        const int *istx, const int *isty, const int it1, const int ntskp, const int ny){
        int ns = blockIdx.x;
        int isx = istx[ns]-1;
        int isy = isty[ns]-1;
        if(threadIdx.x){
            uxall[ns * ntskp + it1] = ux[isx * ny + isy];
        }
        else{
    		uyall[ns * ntskp + it1] = uy[isx * ny + isy];
        }
    }
    __global__ void ups(float *up, float *us, const float *dxux, const float *dyuy,
        const float *dxuy, const float *dyux, const int nx, const int ny){
        devij(nx, ny);

        up[ij] = dxux[ij] + dyuy[ij];
        us[ij] = dxuy[ij] - dyux[ij];
    }

    void query(){
        int devCount;
        cudaGetDeviceCount(&devCount);
        printf("CUDA Device Query...\n");
        printf("There are %d CUDA devices.\n", devCount);

        for (int i = 0; i < devCount; ++i){
            printf("\nCUDA Device #%d\n", i);
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, i);
            printf("Major revision number:         %d\n",  devProp.major);
            printf("Minor revision number:         %d\n",  devProp.minor);
            printf("Name:                          %s\n",  devProp.name);
            printf("Total global memory:           %u\n",  (unsigned int)devProp.totalGlobalMem);
            printf("Total shared memory per block: %u\n",  (unsigned int)devProp.sharedMemPerBlock);
            printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
            printf("Warp size:                     %d\n",  devProp.warpSize);
            printf("Maximum memory pitch:          %u\n",  (unsigned int)devProp.memPitch);
            printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
            for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
            for (int i = 0; i < 3; ++i)
            printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
            printf("Clock rate:                    %d\n",  devProp.clockRate);
            printf("Total constant memory:         %u\n",  (unsigned int)devProp.totalConstMem);
            printf("Texture alignment:             %u\n",  (unsigned int)devProp.textureAlignment);
            printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
            printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
            printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        }
    }
    void forward(const char *oname, const char *wname, const int output){
        // dimension
        float *sxx = mat::create(nx * ny), *sxy = mat::create(nx * ny), *syy = mat::create(nx * ny);
    	float *den = mat::create(nx2 * ny2), *rig = mat::create(nx2 * ny2), *lam = mat::create(nx2 * ny2);
    	float *ux = mat::create(nx * ny), *uy = mat::create(nx * ny);
    	float *vx = mat::create(nx * ny), *vy = mat::create(nx * ny);
    	float *up = mat::create(nx * ny), *us = mat::create(nx * ny);
    	float *dxux = mat::create(nx * ny), *dxuy = mat::create(nx * ny);
    	float *dyux = mat::create(nx * ny), *dyuy = mat::create(nx * ny);
    	float *dxvx = mat::create(nx * ny), *dxvy = mat::create(nx * ny);
    	float *dyvx = mat::create(nx * ny), *dyvy = mat::create(nx * ny);
    	float *dxsxx = mat::create(nx * ny), *dxsxy = mat::create(nx * ny);
    	float *dysxy = mat::create(nx * ny), *dysyy = mat::create(nx * ny);
    	float *ggg = mat::create(nx * ny);
    	float *uxall = mat::create(nst * ntskp), *uyall = mat::create(nst * ntskp);
        float *nd = mat::create(nx2), *q1d = mat::create(nx2);
        float *h_up = mat::create_h(nx * ny), *h_us = mat::create_h(nx * ny);
        float *h_uxall = mat::create_h(nst*  ntskp), *h_uyall = mat::create_h(nst*  ntskp);
    	int *istx = mat::create_i(nst), *isty = mat::create_i(nst);

        // output file
        FILE *wfile=fopen(wname,"w");
		FILE *ofile=fopen(oname,"w");

        // observation points
        psv::istxy<<<nst, 1>>>(istx, isty, na);

        // velocity structure
        FILE *n4096 = fopen("N4096.dat", "r");
    	FILE *q14096 = fopen("Q14096.dat", "r");
        float *h_nd = mat::create_h(nx2);
        float *h_q1d = mat::create_h(nx2);
        for(int i = 0; i < nx2; i++){
    		fscanf(n4096, "%f", &h_nd[i]);
    		fscanf(q14096, "%f", &h_q1d[i]);
    	}
    	fclose(n4096);
    	fclose(q14096);
        mat::copyhd(nd, h_nd, nx2);
        mat::copyhd(q1d, h_q1d, nx2);
        free(h_nd);
        free(h_q1d);
        psv::rdl<<<nx2 * nbt, ny2 / nbt>>>(rig, den, lam, nd, q1d, nx2, ny2, dy);

        // initialize
        float ftmax = t0 + at * 2;

        mat::set(vx, 0.0, nx, ny);
        mat::set(vy, 0.0, nx, ny);
        mat::set(ux, 0.0, nx, ny);
        mat::set(uy, 0.0, nx, ny);
        mat::set(sxx, 0.0, nx, ny);
        mat::set(sxy, 0.0, nx, ny);
        mat::set(syy, 0.0, nx, ny);

        // absorbing boundary confition
        float apara = 0.015;
        psv::gg<<<nx * nbt, ny / nbt>>>(ggg, apara, nx, ny, nxa, nya);

        // time step start
        int ntw = 0;
    	int ntt = 0;
		clock_t timestart=clock();

        for(int it = 0; it < ntmax; it++){
            if(it % 500 == 0){
                printf("timestep: %d / %d\n", it, ntmax);
            }

            ntt++;
    		ntw++;
    		float t=dt*it;

            psv::finidxx<<<nx * nbt, ny / nbt>>>(vx, dxvx, nx, ny, dx, dy, dt);
            psv::finidxy<<<nx * nbt, ny / nbt>>>(vy, dxvy, nx, ny, dx, dy, dt);
            psv::finidyx<<<nx * nbt, ny / nbt>>>(vx, dyvx, nx, ny, dx, dy, dt);
            psv::finidyy<<<nx * nbt, ny / nbt>>>(vy, dyvy, nx, ny, dx, dy, dt);

            psv::sxy<<<nx * nbt, ny / nbt>>>(sxx, syy, sxy, lam, rig, ggg, dxvx, dxvy, dyvx, dyvy, nx, ny, dt);

            psv::finidxy<<<nx * nbt, ny / nbt>>>(sxx, dxsxx, nx, ny, dx, dy, dt);
            psv::finidxx<<<nx * nbt, ny / nbt>>>(sxy, dxsxy, nx, ny, dx, dy, dt);
            psv::finidyy<<<nx * nbt, ny / nbt>>>(sxy, dysxy, nx, ny, dx, dy, dt);
            psv::finidyx<<<nx * nbt, ny / nbt>>>(syy, dysyy, nx, ny, dx, dy, dt);

            psv::vxyuxy<<<nx * nbt, ny / nbt>>>(vx, vy, ux, uy, dxsxx, dxsxy, dysxy, dysyy, ggg, den, t, ftmax, nx, ny, dx, dy, dt, t0, at);

            if(ntt == nskip){
                // save waveform
                ntt = 0;
                uxyall<<<nst, 2>>>(uxall, uyall, ux, uy, istx, isty, (it+1)/nskip, ntskp, ny);
            }
            if(output && ntw == nwrite){
                // write snapshot
                ntw = 0;

                psv::finidxx<<<nx * nbt, ny / nbt>>>(ux, dxux, nx, ny, dx, dy, dt);
                psv::finidxy<<<nx * nbt, ny / nbt>>>(uy, dxuy, nx, ny, dx, dy, dt);
                psv::finidyx<<<nx * nbt, ny / nbt>>>(ux, dyux, nx, ny, dx, dy, dt);
                psv::finidyy<<<nx * nbt, ny / nbt>>>(uy, dyuy, nx, ny, dx, dy, dt);

                psv::ups<<< nx * nbt, ny / nbt>>>(up, us, dxux, dyuy, dxuy, dyux, nx, ny);
                mat::write(ofile, up, h_up, nx, ny);
                mat::write(ofile, us, h_us, nx, ny);
            }
        }

        {
            printf("\ntotal time: %.2fs\n",(float)(clock()-timestart)/CLOCKS_PER_SEC);
            size_t free_byte ;
            size_t total_byte ;
            cudaMemGetInfo( &free_byte, &total_byte ) ;
            float free_db = (float)free_byte ;
            float total_db = (float)total_byte ;
            float used_db = total_db - free_db ;

            printf("memory usage: %.1fMB / %.1fMB\n", used_db/1024.0/1024.0, total_db/1024.0/1024.0);
        }


        // write waveform
        mat::write(wfile, uxall, h_uxall, nst, ntskp);
        mat::write(wfile, uyall, h_uxall, nst, ntskp);

        fclose(ofile);
        fclose(wfile);

        cudaFree(sxx); cudaFree(sxy); cudaFree(syy);
        cudaFree(den); cudaFree(rig); cudaFree(lam);
        cudaFree(ux); cudaFree(uy);
        cudaFree(vx); cudaFree(uy);
        cudaFree(up); cudaFree(us);
        cudaFree(dxux); cudaFree(dxuy);
        cudaFree(dyux); cudaFree(dyuy);
        cudaFree(dxvx); cudaFree(dxvy);
        cudaFree(dyvx); cudaFree(dyvy);
        cudaFree(dxsxx); cudaFree(dxsxy);
        cudaFree(dysxy); cudaFree(dysyy);
        cudaFree(ggg);
        cudaFree(nd); cudaFree(q1d);
        cudaFree(istx); cudaFree(isty);

        free(h_up); free(h_us);
        free(h_uxall); free(h_uyall);
    }
    void waveform(const char *wname){
        int ndskip = 1;
    	float dt2 = dt * 10, dx2 = dx * 4;
    	float *ux = mat::create_h(nst * ntskp), *uz = mat::create_h(nst * ntskp);

    	FILE *file = fopen(wname,"r");

    	FILE *filex = fopen("ux", "w");
    	FILE *filez = fopen("uz", "w");

        mat::read(file, ux, nst, ntskp);
        mat::read(file, uz, nst, ntskp);

        fclose(file);

    	for(int i = 0; i < nst; i += nsskip){
    		fprintf(filex, ">\n");
    		fprintf(filez, ">\n");
    		for(int j = 0; j < ntskp; j += ndskip){
                int ij = i * ntskp + j;
    			float tm = j*dt2;
    			float shift = i*dx2;

    			fprintf(filex, "%f %f\n", tm, ux[ij] * 15.0 + shift);
    			fprintf(filez, "%f %f\n", tm, uz[ij] * 15.0 + shift);
    		}
    	}
    }
    void snapshot(const char *oname){
        FILE *file=fopen(oname,"r");

    	float *up = mat::create_h(nx * ny), *us = mat::create_h(nx * ny);
    	float *u = mat::create_h(nx * ny), *p = mat::create_h(nx * ny), *s = mat::create_h(nx * ny);
    	int n[5]={0,1,2,3,4};

    	FILE **snapshot = (FILE **)malloc(5*sizeof(FILE *));
    	*snapshot = fopen("snap1", "w");
    	*(snapshot + 1) = fopen("snap2", "w");
    	*(snapshot + 2) = fopen("snap3", "w");
    	*(snapshot + 3) = fopen("snap4", "w");
    	*(snapshot + 4) = fopen("snap5", "w");

    	float pmax, smax, cp, lp ,cs, ls, x, y;

    	for(int isnap = 0; isnap < nsnap; isnap++){
    		for(int i = 0; i < nx; i++){
    			for(int j = 0; j < ny; j++){
    				u[i * ny + j] = 0;
    			}
    		}
    		mat::read(file, up, nx, ny);
    		mat::read(file, us, nx, ny);

    		pmax=0.0;
    		smax=0.0;

    		for(int i = 0; i < nx; i++){
    			for(int j = 0; j < ny; j++){
                    int ij = i * ny + j;
    				if(pmax < abs(up[ij])){
    					pmax = abs(up[ij]);
    				}
    				if(smax < abs(us[ij])){
    					smax = abs(us[ij]);
    				}
    			}
    		}
    		// printf("Pmax=%f Smax=%f\n",pmax,smax);

    		for(int i = 0; i < nx; i++){
    			for(int j = 0; j < ny; j++){
                    int ij = i * ny + j;
    				cp=pamp;
    				lp=0.1*pmax;
    				if(abs(up[ij]) > cp && up[ij] < 0.0){
    					up[ij] = -cp;
    				}
    				else if(abs(up[ij]) > cp && up[ij] > 0.0){
    					up[ij] = cp;
    				}
    				if(abs(us[ij]) < lp){
    					up[ij] = 0.0;
    				}
    			}
    		}

    		for(int i = 0; i < nx; i++){
    			for(int j = 0; j < ny; j++){
                    int ij = i * ny + j;
    				cs = samp;
    				ls = 0.1 * smax;
    				if(abs(us[ij]) > cs && us[ij] < 0.0){
    					us[ij] = -cs;
    				}
    				else if(abs(us[ij]) > cs && us[ij] > 0.0){
    					us[ij] = cs;
    				}
    				if(abs(us[ij]) < ls){
    					us[ij] = 0.0;
    				}
    			}
    		}

    		if(isnap == n[0] || isnap == n[1] || isnap == n[2] || isnap == n[3] || isnap == n[4]){
    			for(int j = 0; j < ny; j++){
    				for(int i = 0; i < nx; i++){
                        int ij = i * ny + j;
    					x = i * dx;
    					y = j * dy;
    					p[ij] = up[ij] / pampall;
    					s[ij] = us[ij] / sampall;
    					// if(up[i][j]>1e-5||us[i][j]>1e-5){
    						// printf("%f %f\n", up[i][j],us[i][j]);
    					// }
    				}
    			}
    			for(int j = 0; j < ny; j++){
    				for(int i = 0; i < nx; i++){
                        int ij = i * ny + j;
    					x = i * dx;
    					y = j * dy;
    					if(abs(s[ij]) > abs(p[ij])){
    						u[ij] = -abs(s[ij]);
    					}
    					else if(abs(p[ij]) > abs(s[ij])){
    						u[ij] = abs(s[ij]);
    					}
    					fprintf(*(snapshot+isnap), "%f %f %f\n", x, y, u[ij]);
    				}
    			}
    		}
    	}

    	fclose(file);
    	fclose(*(snapshot));
    	fclose(*(snapshot+1));
    	fclose(*(snapshot+2));
    	fclose(*(snapshot+3));
    	fclose(*(snapshot+4));
    }
}

int main(int argc , char *argv[]){
    // command-line options (e.g. "psv.exe fsw". default: f)
    // q: gpu device query
    // f: forward modeling with waveform output only
    // o: forward modeling with waveform and snapshot output (with much more time consumption)
    // w: convert output waveform data to gmt readable format
    // s: convert output snapshot data to gmt readable format

    int cfg[5] = {0};
    if(argc > 1){
        for(int i = 0; i < argv[1][i] != '\0'; i++){
            switch(argv[1][i]){
                case 'q': cfg[0] = 1;break;
                case 'f': cfg[1] = 1; break;
                case 'o': cfg[1] = 1; cfg[2] = 1; break;
                case 'w': cfg[3] = 1; break;
                case 's': cfg[4] = 1; break;
            }
		}
    }
    else{
        cfg[1] = 0;
    }

    // output file name
    char *oname="opsv";
	char *wname="wpsv";

    if(cfg[0]) psv::query();
    if(cfg[1]) psv::forward(oname, wname, cfg[2]);
    if(cfg[3]) psv::waveform(wname);
    if(cfg[4]) psv::snapshot(oname);
}
