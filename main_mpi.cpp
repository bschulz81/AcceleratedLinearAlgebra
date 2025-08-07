#include <stdio.h>
#include <mpi.h>
#include <vector>
#include "mdspan_omp.h"
using namespace std;

int main(int argc, char** argv)
{
    int process_Rank, size_Of_Cluster;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);

    size_t rows = 4, cols = 4;

    if(process_Rank == 0)
    {
        vector<double>A_data(16,1);
        mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows, cols});
        A.MPI_Send_mdspan(1,1,MPI_COMM_WORLD);
        cout<<"Message Sent:\n";
        printmatrix(A);

        vector<double>A2_data(16,2);
        mdspan<double, std::vector<size_t>> A2(A2_data.data(), true, {rows, cols});
        MPI_send_datastruct(A2.get_datastruct(),1,1,MPI_COMM_WORLD);
        cout<<"Message Sent:\n";
        printmatrix(A2.get_datastruct());


    }

    else if(process_Rank == 1)
    {
        mdspan<double, std::vector<size_t>> B(true,0,1,MPI_COMM_WORLD);
        cout<<"Message Received:\n";
        printmatrix(B);

        datastruct<double> dB=MPI_recv_alloc_datastruct<double>(false,false,false,0,0,1,MPI_COMM_WORLD);
        cout<<"Message Received\n";
        printmatrix(dB);
        datastruct_free(dB,false,false,false,0);


    }

//Strassen algorithm does not work yet with MPI
//    if(process_Rank == 0)
//    {
//
//
//        // Define some data for matrix multiplication A*B=C;
//        vector<double>A3_data(16,0);
//        vector<double>B3_data(16,0);
//        vector<double>C3_data(16,1);
//
//        // Allocate data
//        for (size_t i = 0; i < rows * cols; ++i)
//        {
//            A3_data[i] = i + 1; // Example initialization
//            B3_data[i] = i ; // Example initialization
//        }
//
//
//
//        mdspan<double, std::vector<size_t>> A3(A3_data.data(), true, {rows, cols});
//        mdspan<double, std::array<size_t,2>> B3(B3_data.data(), true, {rows, cols});
//        mdspan<double, std::vector<size_t>> C3(C3_data.data(), true, {rows, cols});
//
//        printmatrix(A3);
//        printmatrix(B3);
//
//        matrix_multiplication_parameters par;
//        par.comm=MPI_COMM_WORLD;
//        par.mpi=true;
//        strassen_multiply(A3, B3, C3,par);
//        printmatrix(C3);
//    }
//    else
//    {
//
//
//
//        MPI_listener<double>(MPI_COMM_WORLD);
//
//    }
    MPI_Finalize();
    return 0;
}
