// main.cpp is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
//
// main.cpp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with mdspan.h. If not,
// see <https://www.gnu.org/licenses/>.
//


#include "mdspan_data.h"
#include "mathfunctions.h"
#include "mathfunctions_mpi.h"
using namespace std;



// Main function
int main()
{



    size_t rows = 4, cols = 4;


    cout<< "We create a 4x4 matrix that owns its own data buffer in a memapped file and then fill the buffer and print it"<<endl;
    mdspan_data<double, std::vector<size_t>> O( true,true, {rows, cols});


    for (size_t i=0; i<16; i++)
    {
        O.data()[i]=(double)i;
    }


    O.printmatrix();


    cout<<"now we create a 4x4 matrix with data in a separate vector"<<endl;

    vector<double>O2_data(16,2);
    mdspan<double, std::vector<size_t>> O2(O2_data.data(), true, {rows, cols});
    O2.printmatrix();



    cout<< "now we make a shallow copy of the first matrix on the second"<<endl;

    O2=O;
    O2.printmatrix();


    cout<<"We test the shallow copy by setting the first element of the first matrix to 42 and then print the first and second matrix"<<endl;
    O.data()[0]=42;

    O.printmatrix();
    O2.printmatrix();

    cout<< "On termination, the shared ptr variable with dummy ref counter should call a deleter that removes the created memory (on device, on the memmapped file, or on heap)"<<endl;

// Define some data for matrix multiplication A*B=C;
    vector<double>A_data(64,0);
    vector<double>B_data(64,0);
    vector<double>C_data(64,1);
   size_t rowsA = 8, colsA = 8;
  //   Allocate data
    for (size_t i = 0; i < rowsA * colsA; ++i)
    {
        A_data[i] = i + 1; // Example initialization
        B_data[i] = i ; // Example initialization
    }





//the same code base can have the strides and extents on heap(vector) or on the stack(array)
//Here, we define row-major data (column major support is included but was not tested).
//the array class has many constructors for various initialization methods and supports higher ranks than 2,
// but this was just tested in the beginning and has currently not much support.
    mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rowsA, colsA});
    mdspan<double, std::array<size_t,2>> B(B_data.data(), true, {rowsA, colsA});
    mdspan<double, std::vector<size_t>> C(C_data.data(), true, {rowsA, colsA});





    cout<<"Ordinary matrix multiplication, on gpu"<<std::endl;


    A.printmatrix();
    B.printmatrix();
//CPU_ONLY lets it multiply on CPU.
//AUTO lets the library decide based on whether the data is already
//on gpu, the algorithm, and the data size.

    Math_Functions_Policy p1(Math_Functions_Policy::AUTO);
//supplying nullptr instead of p1 lets the library use a global default that
//can be configured.
    Math_Functions<double>::matrix_multiply_dot(A, B, C,&p1);
//
//per default
//p1.update_host is set to true.
//If one has several calculations on gpu, this may not be desired and can be switched to false
//


    C.printmatrix();
//
    cout<<"We can also use the Strassen algorithm or its Winograd variant for the multiplication."<<std::endl;
    cout<<"It may offload on gpu. With the Message Passing Interface enabled, it can do so in parallel. "<<std::endl;
    cout<<"otherwise it offloads sequentially. The algorithm can also work entirely on device"<<std::endl;

    cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
    cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
    cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
    cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
    cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl;
    cout <<std::endl;
    std::fill(C_data.begin(),C_data.end(),0);

    Math_MPI_RecursiveMultiplication_Policy p6(Math_Functions_Policy::GPU_ONLY,false,false);
    p6.size_to_stop_recursion=16;

    Math_Functions_MPI<double>::winograd_multiply(A, B, C,&p6);

    C.printmatrix();


//
//Another set of data
   vector<double>A2_data= {4,12,-16,12,37,-43,-16,-43,98};
    vector<double>L2_data(9,0);
    size_t rows2 = 3, cols2 = 3;
////
//
//we want to do a cholesky decomposition with it on CPU
    cout<<"Now a cholesky decomposition on CPU"<<std::endl;

    mdspan<double, std::vector<size_t>> A2(A2_data.data(), true, {rows2, cols2});
    mdspan<double, std::vector<size_t>> L2(L2_data.data(), true, {rows2, cols2});

    A2.printmatrix();
    Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);

    Math_Functions<double>::cholesky_decomposition(A2,L2,&p2);

    L2.printmatrix();


//
//
//
//
//
////
////the first boolean flag implies that the entire cholesky decomposition (not just the multiplication) is now done on gpu
    cout<<"Now the cholesky decomposition is entirely done on GPU"<<std::endl;
////set the results to zero again.
    std::fill(L2_data.begin(),L2_data.end(),0);
    Math_Functions_Policy p3(Math_Functions_Policy::GPU_ONLY);
    Math_Functions<double>::cholesky_decomposition(A2,L2,&p3);


   L2.printmatrix();



//works only with unified shared memory enabled
cout <<"works on GPU only with omp unified shared memory"<<std::endl;
cout<<"With the advanced algorithms on cpu"<<std::endl;
    std::fill(L2_data.begin(),L2_data.end(),0);
    Math_MPI_Decomposition_Policy p9(
        Math_Functions_Policy::CPU_ONLY,
        true,
        false,
        Math_MPI_Decomposition_Policy::Naive);

    Math_Functions_MPI<double>::cholesky_decomposition(A2,L2,&p9);
    L2.printmatrix();

    cout<< "Now we do the same with the lu decomposition"<<std::endl;
    vector<double>A3_data= {1,-2,-2,-3,3,-9,0,-9,-1,2,4,7,-3,-6,26,2};
    vector<double>L3_data(16,0);
    vector<double>U3_data(16,0);
    size_t rows3 = 4, cols3 = 4;
    mdspan<double, std::vector<size_t>> A3(A3_data.data(), true, {rows3, cols3});
    mdspan<double, std::vector<size_t>> L3(L3_data.data(), true, {rows3, cols3});
    mdspan<double, std::vector<size_t>> U3(U3_data.data(), true, {rows3, cols3});

    Math_Functions_Policy p4(Math_Functions_Policy::CPU_ONLY);
    A3.printmatrix();
    cout<<"on CPU"<<std::endl;
    Math_Functions<double>::lu_decomposition(A3,L3,U3,&p4);
    L3.printmatrix();
    U3.printmatrix();
//
    cout<<"Entirely on gpu"<<std::endl;

    std::fill(L3_data.begin(),L3_data.end(),0);
    std::fill(U3_data.begin(),U3_data.end(),0);
    p4.mode=Math_Functions_Policy::GPU_ONLY;
    Math_Functions<double>::lu_decomposition(A3,L3,U3,&p4);
    L3.printmatrix();
    U3.printmatrix();

//works only with unified shared memory enabled
cout <<"works on GPU only with ompunified shared memory"<<std::endl;
cout<<"With the advanced algorithms on CPU"<<std::endl;
    std::fill(L3_data.begin(),L3_data.end(),0);
    std::fill(U3_data.begin(),U3_data.end(),0);
    Math_MPI_Decomposition_Policy p9a(
        Math_Functions_Policy::CPU_ONLY,
        false,
        false,
        Math_MPI_Decomposition_Policy::Naive);


    Math_Functions_MPI<double>::lu_decomposition(A3,L3,U3,&p9a);
    L3.printmatrix();



    cout<< "Now we do the same with the qr decomposition"<<std::endl;

    vector<double>A4_data= {12,-51,4,6,167,-68,-4,24,-41};
    vector<double>Q4_data(9,0);
    vector<double>R4_data(9,0);
    size_t rows4 = 3, cols4 = 3;

    mdspan<double, std::vector<size_t>> A4(A4_data.data(), true, {rows4, cols4});
    mdspan<double, std::vector<size_t>> Q4(Q4_data.data(), true, {rows4, cols4});
    mdspan<double, std::vector<size_t>> R4(R4_data.data(), true, {rows4, cols4});
    Math_Functions_Policy p5(Math_Functions_Policy::CPU_ONLY);
    p5.memmapped_files=true;
    A4.printmatrix();

    Math_Functions<double>::qr_decomposition(A4,Q4,R4,&p5);
    Q4.printmatrix();
    R4.printmatrix();


    std::fill(Q4_data.begin(),Q4_data.end(),0);
    std::fill(R4_data.begin(),R4_data.end(),0);

    cout<<"On gpu"<<std::endl;
    p5.mode=Math_Functions_Policy::GPU_ONLY;
    p5.memmapped_files=false;

    Math_Functions<double>::qr_decomposition(A4,Q4,R4,&p5);
    Q4.printmatrix();
    R4.printmatrix();

    cout<<"Works on GPU only with omp unified shared memory enabled"<<std::endl;
    cout<<"with the advanced algorithms "<<std::endl;
    std::fill(Q4_data.begin(),Q4_data.end(),0);
    Math_MPI_Decomposition_Policy p9b(
        Math_Functions_Policy::CPU_ONLY,
        false,
        false,
        Math_MPI_Decomposition_Policy::Naive);

    p9b.size_to_stop_recursion=16;


    Math_Functions_MPI<double>::qr_decomposition(A4,Q4,R4,&p9b);
    Q4.printmatrix();



}

