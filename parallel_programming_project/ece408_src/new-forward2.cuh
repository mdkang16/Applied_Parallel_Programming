#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 8


#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int xTileWidth = TILE_WIDTH + K-1;
    extern __shared__ float shmem[];
    float* tileX = &shmem[0];
    float* tileK = &shmem[xTileWidth*xTileWidth];
    #define xTile(i1, i0) tileX[(i1)* (xTileWidth) + i0]
    #define kTile(i1, i0) tileK[(i1)* K + i0]
    /*
    X =
    TILE_WIDTH = blockDim;
    B = number of input batches
    M = number of output images (feature map)
    C = number of input images (feature map)
    H = height of input image
    W = width of input image
    K = Mask size
    */

    int H_out = H-K+1;
    int W_out = W-K+1;
    int W_grid = ceil(W_out/1.0f/TILE_WIDTH);

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.z%W_grid)*TILE_WIDTH;
    int w_base = (blockIdx.z/W_grid)*TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;


    // Performing convolution for all input channels
    float res = 0.0f;
    for (int c = 0; c < C; c++)
    {
      // Load kernel from W
      if (threadIdx.x < K && threadIdx.y < K) {
        kTile(threadIdx.y, threadIdx.x) = k4d(m, c, threadIdx.y, threadIdx.x);
      }

      // Load input tile from X
      for (int i = h; i < h_base + xTileWidth; i+= TILE_WIDTH) {
        for (int j = w; j < w_base + xTileWidth; j+= TILE_WIDTH) {
    			if (b < B && i < H && j < W)
    				xTile(i-h_base, j-w_base) = x4d(b,c,i,j);
    			else
    				xTile(i-h_base, j-w_base) = 0.0f;
            }
      }

      // Perform convolution
      __syncthreads();
      for (int y_mask = 0; y_mask < K; y_mask++){
        for (int x_mask = 0; x_mask < K; x_mask++){
          // Read from shared memroy
          res += kTile(y_mask, x_mask) * xTile(threadIdx.y+y_mask, threadIdx.x+x_mask);
        }
      }
      __syncthreads();
    }

    // Write for all output within bounds
    if (b < B && m < M && h < H_out && w < W_out)
    {
      y4d(b, m, h, w) = res;
    }

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a

#undef y4d
#undef x4d
#undef k4d
#undef xTile
#undef kTile
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    /*
    Extract the tensor dimensions into B,M,C,H,W,K
    X =
    TILE_WIDTH = blockDim;
    B = number of input batches
    M = number of output images (feature map)
    C = number of input images (feature map)
    H = height of input image
    W = width of input image
    K = Mask size
    */
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H-K+1;
    const int W_out = W-K+1;
	  const int H_grid = ceil(H_out/1.0/TILE_WIDTH);
	  const int W_grid = ceil(W_out/1.0/TILE_WIDTH);
    const int Z = W_grid*H_grid;

    // Set the kernel dimensions
    int xTileWidth = TILE_WIDTH + K-1;
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // Allocate shared memory space in runtime
    size_t shmem_size = (xTileWidth*xTileWidth + K*K)*sizeof(float);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    // CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif