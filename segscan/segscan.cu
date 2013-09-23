/**
 * Segmented Scan CUDA sample
 *
 * Original:
 * "Efficient Parallel Scan Algorithms for GPUs",
 * Shubhabrata Sengupta,Mark Harris,  Michael Garland.
 * https://research.nvidia.com/sites/default/files/publications/nvr-2008-003.pdf
 *
 * via
 * aokomoriuta san
 * http://qiita.com/aokomoriuta/items/3c2a80181a01c7f22e7f
 *
 * Using a template kernel.cu in NVIDIA Cuda Toolkit 5.5
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

template<typename T>
class AddOp {
public:
    static __device__ inline T apply(const T a, const T b) {
        return a + b;
    }
};

template<typename T>
class MaxOp {
public:
    static __device__ inline T apply(const T a, const T b) {
        return max(a,b);
    }
};

cudaError_t segmentedScanTest(int *c, const int *a, const int *b, unsigned int size);

template<typename T, class OP>
__device__ void compute_segscan(volatile T *p, volatile int *hd,
        const unsigned int tid, const unsigned int offset) {
    const unsigned int lane = tid & 31;
    if (lane >= offset) {
        p[tid] = hd[tid] ? p[tid] : OP::apply(p[tid - offset],p[tid]);
        hd[tid] = hd[tid - offset] | hd[tid];
    }
}

/**
 * Figure 6
 */
template<typename T, int Kind, class OP>
__device__ T segscan_warp(volatile T *p, volatile int *hd,
        const unsigned int tid = threadIdx.x) {
    const unsigned int lane = tid & 31;
    compute_segscan<T,OP>(p, hd, tid, 1);
    compute_segscan<T,OP>(p, hd, tid, 2);
    compute_segscan<T,OP>(p, hd, tid, 4);
    compute_segscan<T,OP>(p, hd, tid, 8);
    compute_segscan<T,OP>(p, hd, tid, 16);

    if (Kind == 0)
        return p[tid];
    else
        return (lane > 0) ? p[tid - 1] : 0;
}

template <typename T, class OP>
__global__ void segscan_warp_kernel(const T *src, T *dst, int *flag) {
	const unsigned int tid = threadIdx.x;
	dst[tid] = src[tid];
	segscan_warp<int,0,OP>(dst, flag, threadIdx.x);
}

/**
 * Figure 3
 */
template<typename T, int Kind, class OP>
__device__ T scan_warp(volatile T *p,
        const unsigned int tid = threadIdx.x) {
    const int lane = tid & 31;
    if (lane >= 1)
        p[tid] = OP::apply(p[tid - 1], p[tid]);
    if (lane >= 2)
        p[tid] = OP::apply(p[tid - 2], p[tid]);
    if (lane >= 4)
        p[tid] = OP::apply(p[tid - 4], p[tid]);
    if (lane >= 8)
        p[tid] = OP::apply(p[tid - 8], p[tid]);
    if (lane >= 16)
        p[tid] = OP::apply(p[tid - 16], p[tid]);

    return p[tid];
}

/**
 * Figure 7
 */
template<typename T, int Kind, class OP>
__device__ T segscan_warp2(volatile T *p, volatile int *hd,
        const unsigned int tid = threadIdx.x) {
    const unsigned int lane = tid & 31;

    if (hd[tid])
        hd[tid] = lane;
    int mindex = scan_warp<T,Kind, MaxOp<T> >(hd,tid);

    if (lane >= mindex + 1)
        p[tid] = OP::apply(p[tid - 1],p[tid]);
    if (lane >= mindex + 2)
        p[tid] = OP::apply(p[tid - 2], p[tid]);
    if (lane >= mindex + 4)
        p[tid] = OP::apply(p[tid - 4], p[tid]);
    if (lane >= mindex + 8)
        p[tid] = OP::apply(p[tid - 8], p[tid]);
    if (lane >= mindex + 16)
        p[tid] = OP::apply(p[tid - 16], p[tid]);

    if (Kind == 0)
        return p[tid];
    else
        return (lane > 0 && mindex != lane) ? p[tid - 1] : 0;
}

template <typename T, class OP>
__global__ void segscan_warp2_kernel(const T *src, T *dst, int *flag)  {
	const unsigned int tid = threadIdx.x;
	dst[tid] = src[tid];
	dst[tid] = segscan_warp2<int,0, OP>(dst, flag, tid);
}

/**
 * Figure 10
 */
template<typename T, int Kind, class OP>
__device__ T segscan_block(volatile T *p, volatile int *hd,
        const unsigned int tid = threadIdx.x) {
    const unsigned int warpid = tid >> 5;
    const unsigned int warp_first = warpid << 5;
    const unsigned int warp_last = warp_first + 31;

// step 1a
    bool warp_is_open = (hd[warp_first] == 0);
    __syncthreads();

// step 1b
    T val = segscan_warp2<T,Kind, OP>(p, hd, tid);

// step 2a
    T warp_total = p[warp_last];

// step 2b
    int warp_flag = hd[warp_last] != 0 || !warp_is_open;
    bool will_accumulate = warp_is_open && hd[tid] == 0;

    __syncthreads();

// step 2c
    if (tid == warp_last) {
        p[warpid] = warp_total;
        hd[warpid] = warp_flag;
    }
    __syncthreads();

// step 3
    if (warpid == 0)
        segscan_warp2<T,0, OP>(p, hd, tid);

    __syncthreads();

// step 4
    if (warpid != 0 && will_accumulate)
        val = OP::apply( p[tid - 1] , val);

    p[tid] = val;
    __syncthreads();

    return val;
}

template <typename T, class OP>
__global__ void segscan_block_kernel(const T *src, T *dst, int *flag)  {
	const unsigned int tid = threadIdx.x;
	dst[tid] = src[tid];
	dst[tid] = segscan_block<int,0, OP>(dst, flag, tid);
}

template <typename T, class OP, unsigned int SIZE>
__global__ void segscan_block_kernel_smem(const T *src, T *dst, int *flag)  {
	const unsigned int tid = threadIdx.x;
	__shared__ T smem[SIZE];
	smem[tid] = src[tid];
	dst[tid] = segscan_block<int,0, OP>(smem, flag, tid);
}


template<typename T>
void segmentedScanCpu( const T *src, T *dst, int *flag, const unsigned int size ) {
	
	dst[0] = src[0];
	for ( int i=1; i<size; i++ ) {
		dst[i] = flag[i] ? src[i] : dst[i-1] + src[i];
	}
}

int main() {
    const int arraySize = 1024;
    int src[arraySize];
    int hd[arraySize]={0};
    int dst[arraySize] = { 0 };
    int dstCpu[arraySize] = { 0 };

    for ( int i=0; i<arraySize; i++) {
        src[i] = i;
        hd[i] = (i % 4)==0 ? 1 : 0;
    }

    cudaError_t cudaStatus = segmentedScanTest(dst, src, hd, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	const char *fmt = "%4d";
	for ( int i=0; i<32; i++ ) {
		printf(fmt, src[i]);
	}
	puts("");

	for ( int i=0; i<32; i++ ) {
		printf(fmt, hd[i]);
	}
	puts("");

	for ( int i=0; i<32; i++ ) {
		printf(fmt, dst[i]);
	}
	puts("");

	segmentedScanCpu(src, dstCpu, hd, arraySize);
	for ( int i=0; i<arraySize; i++ ) {
		if ( dstCpu[i] != dst[i] ) {
			puts("compared... not ok");
			break;
		}
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t segmentedScanTest(int *c, const int *a, const int *b, unsigned int size) {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
                "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**) &dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**) &dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**) &dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int),
            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int),
            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	// segscan_warp_kernel<int, AddOp<int> ><<<1, 32>>>(dev_a, dev_c, dev_b);
	//scan_warp_max_kernel<<<1, 32>>>(dev_b, dev_c);
	//segscan_warp2_kernel<<<1, 32>>>(dev_a, dev_c, dev_b);
	segscan_block_kernel<int,  AddOp<int> ><<<1, size>>>(dev_a, dev_c, dev_b);
	//segscan_block_kernel_smem<int, 2048><<<1, size>>>(dev_a, dev_c, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, " launch failed: %s\n",
                cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
                "cudaDeviceSynchronize returned error code %d after launching addKernel!\n",
                cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int),
            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error: cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
