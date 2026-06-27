#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <cstddef>

#include "mdspan_data.h"
#include "mathfunctions.h"
#include "datablock_mpifunctions.h"
#include "mathfunctions_mpi.h"


using namespace std;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
//    {
//
//
//
//
//        int process_Rank, size_Of_Cluster;
//
//        MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
//        MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);
//
//        {
//
//            size_t rows = 4, cols = 4;
//            if(process_Rank == 0)
//            {
//
//                cout<<"this tests recursive algorithms of the library that use hybrid gpu and cpu mode with Message passing interface and OpenMP on device"<<endl;
//
//                vector<double>A2_data(16,4);
//                mdspan<double, std::vector<size_t>> A2(A2_data.data(),  {rows, cols},true);
//                DataBlock_MPI_Functions<double>::MPI_Send_DataBlock(A2,1,1,MPI_COMM_WORLD);
//                cout<<"Message Sent:\n";
//                A2.printtensor();
//            }
//            else if(process_Rank == 1)
//            {
//
//                cout<<"As a recieve buffer, mdspan_data is very useful, which allocates its own memory in the suitable size." <<endl;
//                cout<<" It can do so on a memory map, on host working memory, or on device memory, which is then accesible only with a device kernel"<<endl;
//                mdspan_data<double, std::vector<size_t>> B( {rows, cols},true);
//
//                DataBlock_MPI_Functions<double>::MPI_Recv_DataBlock(B,0,1,MPI_COMM_WORLD);
//                cout<<"Message recieved"<<endl;
//                B.printtensor();
//            }
//
//        }
//
//        {
//
//
//            size_t rows = 8, cols = 8;
//            Math_MPI_RecursiveMultiplication_Policy p(Math_Functions_Policy::GPU_ONLY,true,true);
//            p.update_host=true;
//            if(process_Rank == 0)
//            {
//                vector<double>A3_data(rows*cols,0);
//                vector<double>B3_data(rows*cols,0);
//                for (size_t i = 0; i < rows * cols; ++i)
//                {
//                    A3_data[i] = i + 1;
//                    B3_data[i] = i ;
//                }
//
//                mdspan<double, std::vector<size_t>> A3(A3_data.data(),  {rows, cols},true);
//                mdspan<double, std::vector<size_t>> B3(B3_data.data(), {rows, cols},true);
//
//                cout<<"We define two matrices A and B:" <<endl;
//                A3.printtensor();
//                B3.printtensor();
//
//                {
//
//                    cout<<"ordinary matrix multiplication on a single node with openmp. It will decide automatically whether to compute on gpu or not"<<endl;
//
//                    Math_Functions_Policy p1(Math_Functions_Policy::AUTO);
//                    cout<<"supplying nullptr instead of a pointer to Math_Functions_Policy lets the library use a global default that can be configured."<<endl;
//                    mdspan_data<double, std::vector<size_t>> C({rows, cols},true);
//                    Math_Functions<double>::matrix_multiply_dot(A3, B3, C,&p1);
//                    C.printtensor();
//
//                }
//
//                {
//
//                    mdspan_data<double, std::vector<size_t>> C3({rows, cols},true);
//
//                    cout<<"matrix multiplication with the Strassen algorithm over message passing interface"<<std::endl;
//                    cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
//                    cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
//                    cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
//                    cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
//                    cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl<<endl;
//
//                    Math_Functions_MPI<double>::strassen_multiply(A3, B3, C3,&p);
//                    C3.printtensor();
//                    Math_Functions_MPI<double>::MPI_recursion_helper_end(p.comm);
//                }
//            }
//            else
//            {
//                Math_Functions_MPI<double>::MPI_recursive_multiplication_helper(&p);
//            }
//
//        }
//
//    }
//
    {

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rootrank=0;
        if(rank==rootrank)
            cout<<"\n\n\n this multiplies two matrices with the summa matrix multiplication algorithm. the data is handled on GPU\n\n";



        std::vector<double> A_data,B_data,C_data;
        size_t extentsA[2],extentsB[2],extentsC[2];
        size_t stridesA[2],stridesB[2],stridesC[2];

        DataBlock<double> A1,B1,C1;







        if (rank == rootrank)
        {
            constexpr size_t M = 11;
            constexpr size_t K = 17;
            constexpr size_t N = 13;

            A_data.resize(M*K);
            std::iota(A_data.begin(), A_data.end(), 0);
            extentsA[0] = M;
            extentsA[1] = K;

            A1=DataBlock<double> (A_data.data(),0, true,2, extentsA,stridesA,true, true, false, -1);
            cout<<"Matrix A\n";
            A1.printtensor();

            B_data.resize(K*N);
            std::iota(B_data.rbegin(), B_data.rend(), 0);

            extentsB[0] = K;
            extentsB[1] = N;

            B1=DataBlock<double>(B_data.data(),0, true,2, extentsB,stridesB,true, true, false, -1);
            cout<<"Matrix B\n";
            B1.printtensor();

            C_data.resize(M*N, 0);

            extentsC[0] = M;
            extentsC[1] = N;

            C1=DataBlock<double>(C_data.data(),0, true,2, extentsC,stridesC,true, true, false, -1);


        }

        MPI_Comm cart_comm =  Math_Functions_MPI<double>::create_summa_communicator(6,6, rank == rootrank ? &A1 : nullptr,
                                                        rank == rootrank ? &B1 : nullptr,
                                                          rank == rootrank ? &C1 : nullptr,
                                                          rootrank);
        if(cart_comm == MPI_COMM_NULL)
        {
            goto endofblock;
        }
        DistributedDataBlock<double> block1,block2,block3;

        MPI_CartesianContext ctx=MPI_CartesianContext(cart_comm);
        BlockMappingPolicy policy=BlockMappingPolicy(ctx.gridrank);


        DataBlock_MPI_Functions<double>::MPI_Scatter_matrix_to_submatrices_alloc(6,6,block1,false,true,  omp_get_default_device(),&ctx,&policy, rootrank,rank==rootrank? &A1:nullptr);
        DataBlock_MPI_Functions<double>::MPI_Scatter_matrix_to_submatrices_alloc(6,6,block2,false,true,  omp_get_default_device(),&ctx,&policy, rootrank,rank==rootrank? &B1:nullptr);
        DataBlock_MPI_Functions<double>::MPI_Scatter_matrix_to_submatrices_alloc(6,6,block3,false,true, omp_get_default_device(),&ctx,&policy, rootrank,rank==rootrank? &C1:nullptr);
        block1.printtensors();
        block2.printtensors();
        block3.printtensors();


        Math_Functions_MPI<double>::matrix_multiply_dot_Distributed(block1,block2,block3);

        DataBlock<double> A1copy;

        DataBlock_MPI_Functions<double>::MPI_Gather_matrix_from_submatrices_alloc(block3,rootrank,rank==rootrank? &A1copy:nullptr,  false, false, -1);

        if(rank==rootrank)
        {
            cout<<"Matrix C\n";
            A1copy.printtensor();
            DataBlock_MPI_Functions<double>::MPI_Free_DataBlock(A1copy,false);
        }

        if(block1.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block1);
        }

        if(block2.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block2);
        }
        if(block3.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block3);
        }

        MPI_Comm_free(&cart_comm);



    }

endofblock:
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rootrank=0;

        if(rank==rootrank)
            cout<<"\n\n\n This example scatters two vectors on gpu, computes the scalar product, and makes a vector addition\n\n";


        std::vector<double> A_data,B_data,C_data;
        size_t extentsA[1],extentsB[1],extentsC[1];
        size_t stridesA[1],stridesB[1],stridesC[1];

        DataBlock<double> A1,B1,C1;

        MPI_Comm cart_comm;
//
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);



        int dims[1] = {0};
        MPI_Dims_create(size, 1, dims);

        int periods[1] = {0};

        MPI_Cart_create(
            MPI_COMM_WORLD,
            1,
            dims,
            periods,
            0,
            &cart_comm
        );

        if (rank == rootrank)
        {
            A_data.resize(12*12);
            std::iota (std::begin(A_data), std::end(A_data), 0);

            extentsA[0]=12*12;

            A1=DataBlock<double> (A_data.data(),0, true,1, extentsA,stridesA,true, true, false, -1);
            cout<<"Vector A\n";
            A1.printtensor();

            B_data.resize(12*12);
            std::iota (std::rbegin(B_data), std::rend(B_data), 0);

            extentsB[0]=12*12;


            B1=DataBlock<double>(B_data.data(),0, true,1, extentsB,stridesB,true, true, false, -1);
            cout<<"Vector B\n";
            B1.printtensor();

            C_data.resize(12*12,0);

            extentsC[0]= 12*12;
            C1=DataBlock<double>(C_data.data(),0, true,1, extentsC,stridesC,true, true, false, -1);


        }

        DistributedDataBlock<double> block1,block2,block3;

        MPI_CartesianContext ctx=MPI_CartesianContext(cart_comm);
        BlockMappingPolicy policy=BlockMappingPolicy(ctx.gridrank);
        DataBlock_MPI_Functions<double>::MPI_Scatter_vector_to_subvectors_alloc(6,block1,false,true, omp_get_default_device(),&ctx,&policy, rootrank,rank==rootrank? &A1:nullptr);
        DataBlock_MPI_Functions<double>::MPI_Scatter_vector_to_subvectors_alloc(6,block2,false,true, omp_get_default_device(),&ctx,&policy, rootrank,rank==rootrank? &B1:nullptr);
        DataBlock_MPI_Functions<double>::MPI_Scatter_vector_to_subvectors_alloc(6,block3,false,true, omp_get_default_device(),&ctx,&policy, rootrank,rank==rootrank? &C1:nullptr);
        block1.printtensors();
        block2.printtensors();


        double result=0;
        Math_Functions_MPI<double>::dot_product_Distributed(block1,block2,0,&result);


        Math_Functions_MPI<double>::vector_add_Distributed(block1,block2,block3);

        block3.printtensors();

        DataBlock<double> A1copy;

        DataBlock_MPI_Functions<double>::MPI_Gather_vector_from_subvectors_alloc(block3,rootrank,rank==rootrank? &A1copy:nullptr,  false, false, -1);

        if(rank==rootrank)
        {
            cout<<"scalarproduct result: "<< result<<"\n";
            cout<<"result of vector operation\n";
            A1copy.printtensor();
            DataBlock_MPI_Functions<double>::MPI_Free_DataBlock(A1copy,false);
        }

        if(block1.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block1);
        }

        if(block2.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block2);
        }
        if(block3.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block3);
        }

        MPI_Comm_free(&cart_comm);

   }


    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rootrank=0;
        MPI_Comm cart_comm;

        if(rank==rootrank)
            cout<<"\n\n\n This example scatters a matrix and a vector and multiplies them into a vector on gpu\n\n";



        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);

        int periods[2] = {0, 0};

        MPI_Cart_create(
            MPI_COMM_WORLD,
            2,
            dims,
            periods,
            0,
            &cart_comm
        );

        std::vector<double> A_data,B_data,C_data;
        size_t extentsA[2],extentsB[1],extentsC[1];
        size_t stridesA[2],stridesB[1],stridesC[1];

        DataBlock<double> A1,B1,C1;


        if (rank == rootrank)
        {
            A_data.resize(11*6);
            std::iota (std::begin(A_data), std::end(A_data), 0);

            extentsA[0]=6;
            extentsA[1]=11;


            A1=DataBlock<double> (A_data.data(),0, false,2, extentsA,stridesA,true, true, false, -1);

            A1.printtensor();
            B_data.resize(11);
            std::iota (std::begin(B_data), std::end(B_data),1);

            extentsB[0]= 11;

            B1=DataBlock<double>(B_data.data(),0, true,1, extentsB,stridesB,true, true, false, -1);

            B1.printtensor();
            C_data.resize(8,0);

            extentsC[0]= 6;

            C1=DataBlock<double>(C_data.data(),0, true,1, extentsC,stridesC,true, true, false, -1);


        }

        DistributedDataBlock<double> block1,block2,block3;

        MPI_CartesianContext ctx=MPI_CartesianContext(cart_comm);
        BlockMappingPolicy policy=BlockMappingPolicy(ctx.gridrank);
        DataBlock_MPI_Functions<double>::MPI_Scatter_matrix_to_submatrices_alloc(3,3,block1,false,true, omp_get_default_device(), &ctx,&policy,rootrank,rank==rootrank? &A1:nullptr);
        DataBlock_MPI_Functions<double>::MPI_Scatter_vector_to_subvectors_alloc(3,block2,false,true, omp_get_default_device(), &ctx,&policy,rootrank,rank==rootrank? &B1:nullptr);
        DataBlock_MPI_Functions<double>::MPI_Scatter_vector_to_subvectors_alloc(3,block3,false,true,omp_get_default_device(), &ctx,&policy,rootrank,rank==rootrank? &C1:nullptr);

        block1.printtensors();
        block2.printtensors();

        Math_Functions_MPI<double>::Matrix_Vector_multiply_Distributed(block1,block2,block3);
        block3.printtensors();
        DataBlock<double> A1copy;

        DataBlock_MPI_Functions<double>::MPI_Gather_vector_from_subvectors_alloc(block3,rootrank,rank==rootrank? &A1copy:nullptr,  false, false, -1);

        if(rank==rootrank)
        {
            A1copy.printtensor();
            DataBlock_MPI_Functions<double>::MPI_Free_DataBlock(A1copy,false);
        }

        if(block1.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block1);
        }

        if(block2.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block2);
        }
        if(block3.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block3);
        }

        MPI_Comm_free(&cart_comm);


    }
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rootrank=0;
        if(rank==rootrank)
            cout<<"\n\n\n this example scatters and gathers a rank 4 tensor into rank 2 blocks over a two dimensional process grid\n\n";


        MPI_Comm cart_comm;

        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);

        int periods[2] = {0, 0};

        MPI_Cart_create(
            MPI_COMM_WORLD,
            2,
            dims,
            periods,
            0,
            &cart_comm
        );


        DataBlock<double> A1;
        size_t extents[4];
        size_t strides[4];
        std::vector<double> A_data;


        if (rank == rootrank)
        {
            extents[0] = 5;
            extents[1] = 4;
            extents[2] = 3;
            extents[3] = 2;

            size_t n = 1;
            for(int i=0; i<4; i++)
                n *= extents[i];

            A_data.resize(n);

            std::iota(A_data.begin(), A_data.end(), 0.0);

            A1 = DataBlock<double>(
                     A_data.data(),
                     0,
                     false,
                     4,
                     extents,
                     strides,
                     true,
                     true,
                     false,
                     -1);

            A1.printtensor();
        }

        DistributedDataBlock<double> block;
        size_t blockrank = 2;

        size_t blockextents[2] = {2,3};

        MPI_CartesianContext ctx=MPI_CartesianContext(cart_comm);
        BlockMappingPolicy policy=BlockMappingPolicy(ctx.gridrank);
        DataBlock_MPI_Functions<double>::MPI_Scatter_tensor_to_subtensors_alloc(blockrank,blockextents,block,false, true,omp_get_default_device(), &ctx,&policy,rootrank,rank==rootrank?&A1:nullptr);
        block.printtensors();

        DataBlock<double> A1copy;
        DataBlock_MPI_Functions<double>::MPI_Gather_tensor_from_subtensors_alloc(block,rootrank,rank==rootrank?&A1copy:nullptr,  false, false, -1);
        if(rank==rootrank)
        {
            A1copy.printtensor();
            DataBlock_MPI_Functions<double>::MPI_Free_DataBlock(A1copy,false);
        }

        if(block.local_blocknumber()>0)
        {
            DataBlock_MPI_Functions<double>::MPI_Free_DistributedDataBlock(block);
        }

    }

    MPI_Finalize();
    return 0;


}
