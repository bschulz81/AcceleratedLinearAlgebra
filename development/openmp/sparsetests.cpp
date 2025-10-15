#include <vector>
#include "datablockcontainer.h"
#include "inkernel_mathfunctions.h"
#include "gpu_mathfunctions.h"
#include "mdspan_omp.h"
#include "mdspan_data.h"
int main()
{


    size_t M = 8, K = 8, N = 8;
    std::vector<double> A(M*K,0), B(K*N,0), C1(M*N,0),C2(M*N,0);

    // fill A and B with simple values
    for (size_t i =0; i < M*K; ++i) A[i] = i;
    for (size_t i = 0; i < K*N; ++i) B[i] = i + 1;


    // wrap dense arrays in DataBlock
    std::vector<size_t> extA{M,K}, extB{K,N},extC{M,N};
    std::vector<size_t> stridesA{K,1}, stridesB{N,1}, stridesC{N,1};

    DataBlock<double> Ad(A.data(), M*K, true, 2, extA.data(), stridesA.data(), false,false,false);

    size_t sub_ext[2],sub_strides[2];
    DataBlock<double> A1 = Ad.subspanmatrix(0, 0, 4, 4, sub_ext, sub_strides);
    DataBlock<double> A2 = Ad.subspanmatrix(4, 0, 4, 4, sub_ext, sub_strides);
    DataBlock<double> A3 = Ad.subspanmatrix(0, 4, 4, 4, sub_ext, sub_strides);
    DataBlock<double> A4 = Ad.subspanmatrix(4, 4, 4, 4, sub_ext, sub_strides);

// fill some blocks (e.g., leave A3 zero)
    for(size_t i = 0; i < 4; ++i)
        for(size_t j = 0; j < 4; ++j)
        {
            A1(i,j) = i+i;
            A2(i,j) = i;
            A3(i,j)=0;
            A4(i,j) =0;
        }


    size_t sub_ext2[2],sub_strides2[2];
    DataBlock<double> Bd(B.data(), K*N, true, 2, extB.data(), stridesB.data(), false,false,false);

    DataBlock<double> B1 = Bd.subspanmatrix(0, 0, 4, 4, sub_ext2, sub_strides2);
    DataBlock<double> B2 = Bd.subspanmatrix(4, 0, 4, 4, sub_ext2, sub_strides2);
    DataBlock<double> B3 = Bd.subspanmatrix(0, 4, 4, 4, sub_ext2, sub_strides2);
    DataBlock<double> B4 = Bd.subspanmatrix(4, 4, 4, 4, sub_ext2, sub_strides2);

// fill some blocks (e.g., leave A3 zero)
    for(size_t i = 0; i < 4; ++i)
        for(size_t j = 0; j < 4; ++j)
        {
            B1(i,j) = i;
            B2(i,j) = 0;
            B3(i,j)=0;
            B4(i,j) =0;
        }



    Bd.printtensor();
cout <<"sparsity "<<Bd.sparsity()<<endl;

    DataBlock<double> C1d(C1.data(), M*N, true, 2, extC.data(), stridesC.data(), false,false,false);
    DataBlock<double> C2d(C2.data(), M*N, true, 2, extC.data(), stridesC.data(), false,false,false);
cout<<"naive matrix multiplication"<<endl;
    In_Kernel_Mathfunctions<double>::matrix_multiply_dot_s(Ad, Bd, C1d);
    C1d.printtensor();

    size_t block_shape[2]={2,2};
    BlockedDataView<double> Ablocks(Ad, block_shape,true);
    size_t block_shape2[2]={2,2};
    BlockedDataView<double> Bblocks(Bd, block_shape2,true);


cout<<"We now do a sparse multiplication"<<endl;
   In_Kernel_Mathfunctions<double>::matrix_multiply_dot_sparse_s(Ablocks, Bblocks, C2d);
   //would also work on device
    //GPU_Math_Functions<double>::matrix_multiply_dot_sparse_g(Ablocks,Bblocks,C2d,omp_get_default_device(),true,true);

    C2d.printtensor();

cout<<"now an example with sparse matrx multiplication and the mdspan class"<<endl;

mdspan<double, std::vector<size_t>> Aspan(A.data(),  {M,K},true);
mdspan<double, std::vector<size_t>> Bspan(B.data(),  {K,N},true);
mdspan_data<double, std::vector<size_t>> Cspan({M,N},true);

cout<<"of course we offload the data first to device"<<endl;
Aspan.device_data_upload(true);
Bspan.device_data_upload(true);
Cspan.device_data_alloc(true);

cout <<"sparsity "<<Bspan.sparsity()<<endl;

BlockedDataView<double> Ablocks1(Aspan, block_shape,true);
BlockedDataView<double> Bblocks2(Bspan, block_shape2,true);



GPU_Math_Functions<double>::matrix_multiply_dot_sparse_g(Ablocks1,Bspan,Cspan,omp_get_default_device(),true,true);

Cspan.printtensor();

    return 0;
}
