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


    {


        size_t rows = 4, cols = 4;


        cout<< "We create a 4x4 matrix that owns its own data buffer in a memapped file and then fill the buffer and print it"<<endl;
        mdspan_data<double, std::vector<size_t>> O( true,true, {rows, cols});


        for (size_t i=0; i<16; i++)
        {
            O.data()[i]=(double)i;
        }


        O.printtensor();


        cout<<"now we create a 4x4 matrix with data in a separate vector"<<endl;

        vector<double>O2_data(16,2);
        mdspan<double, std::vector<size_t>> O2(O2_data.data(), true, {rows, cols});
        O2.printtensor();



        cout<< "now we make a shallow copy of the first matrix on the second"<<endl;

        O2=O;
        O2.printtensor();


        cout<<"We test the shallow copy by setting the first element of the first matrix to 42 and then print the first and second matrix"<<endl;
        O.data()[0]=42;

        O.printtensor();
        O2.printtensor();

        cout<< "On termination, the shared ptr variable with dummy ref counter should call a deleter that removes the created memory (on device, on the memmapped file, or on heap)"<<endl;
    }


    {

        vector<double>A_data(12*12,0);
        vector<double>B_data(12*12,0);
        vector<double>C_data(12*12,1);
        size_t rowsA = 12, colsA = 12;
        A_data= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11,
                 11, 9, 7, 5, 3, 1, 12, 10, 8, 6, 4, 2,
                 3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10,
                 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
                 4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9,
                 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10,
                 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12,
                 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5,
                 6, 1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11,
                 11, 2, 9, 4, 12, 7, 3, 10, 5, 1, 8, 6
                };
        B_data= {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10,
                 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
                 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 12,
                 12, 9, 6, 3, 10, 7, 4, 1, 8, 5, 2, 11,
                 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11,
                 11, 8, 5, 2, 9, 6, 3, 12, 7, 4, 1, 10,
                 3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10,
                 10, 7, 4, 1, 11, 8, 5, 2, 12, 9, 6, 3,
                 4, 8, 12, 3, 7, 11, 2, 6, 10, 1, 5, 9,
                 9, 5, 1, 7, 3, 11, 8, 4, 12, 6, 2, 10
                };


//the same code base can have the strides and extents on heap(vector) or on the stack(array)
//Here, we define row-major data (column major support is included but was not tested).
//the array class has many constructors for various initialization methods and supports higher ranks than 2,
// but this was just tested in the beginning and has currently not much support.
        mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rowsA, colsA});
        mdspan<double, std::array<size_t,2>> B(B_data.data(), true, {rowsA, colsA});
        mdspan<double, std::vector<size_t>> C(C_data.data(), true, {rowsA, colsA});





        cout<<"Ordinary matrix multiplication, on gpu"<<std::endl;


        A.printtensor();
        B.printtensor();

        cout <<"CPU_ONLY lets it multiply on CPU.AUTO lets the library decide based on whether the data is already on gpu, the algorithm, and the data size."<<endl;

        Math_Functions_Policy p1(Math_Functions_Policy::AUTO);
        cout<<"supplying nullptr instead of a pointer to Math_Functions_Policy lets the library use a global default that can be configured."<<endl;
        Math_Functions<double>::matrix_multiply_dot(A, B, C,&p1);
        cout<<"per default update_host is set to true. If one has several calculations on gpu, this may not be desired and can be switched to false"<<endl;
//

        C.printtensor();
//
        cout<<"We can also use the Strassen algorithm or its Winograd variant for the multiplication."<<std::endl;
        cout<<"It may offload on gpu. With the Message Passing Interface enabled, it can do so in parallel. "<<std::endl;
        cout<<"otherwise it offloads sequentially. The algorithm can also work entirely on device with devicepointers to the data"<<std::endl;

        cout<<"in auto mode, the following default treshholds are set in mathfunctions.h and can be changed for convenience"<<std::endl;
        cout << "max_problem_size_for_gpu;" << "This is the size of the gpu memory, data larger than this is not offloaded"<< std::endl;
        cout <<" default_cubic_treshold = 256;"<< "The default number of elements at which matrices are auto offloaded in multiplication"<< std::endl;
        cout<< " default_square_treshold = 1000;"<<"The default number of elements at which matrices are auto offloaded for addition"<< std::endl;
        cout <<" default_linear_treshold = 1000000;"<<"The default number of elements at which vectors are auto offloaded for addition"<<std::endl;
        cout <<std::endl;
        std::fill(C_data.begin(),C_data.end(),0);

        cout<<"we now set it on gpu and set the size when to stop recursion to 2, per default, this is at 64"<<endl;

        Math_MPI_RecursiveMultiplication_Policy p(Math_Functions_Policy::GPU_ONLY,false,false);
        p.size_to_stop_recursion=2;

        Math_Functions_MPI<double>::strassen_multiply(A, B, C,&p);

        C.printtensor();

    }

    {

        cout<< "Now some tests whether the library accepts row and column major data and can extract rows and columns with the same code. "<<endl;
        cout<<" Note that this tests only the datastruct class, which can be offloaded to gpu. it is non owning, "<<endl;
        cout<<" compared to the mdspan class which owns strides and extents and mdspan_data, which owns the data as well"<<endl<<endl;
        {

            vector<double>A_data(3*7,0);
            A_data = {1,2,3,4,5,6,7,
                      8,9,10,11,12,13,14,
                      15,16,17,18,19,20,21
                     };

            size_t rows=3,cols=7;
            mdspan<double, std::vector<size_t>> A(A_data.data(), true, rows, cols);
            cout<<"A"<<endl;
            A.printtensor();
            size_t extaa[2]= {3,7};
            size_t straa[2];
            datastruct<double>Aaa(A_data.data(),0, true,2,extaa,straa,true,true,false);
            cout<<"A"<<Aaa.datalength()<<endl;
            Aaa.printtensor();

            size_t extaaa[2];
            size_t straaa[2];
            datastruct<double>Aaaa(A_data.data(),0, true,3,7,extaaa,straaa,true,true,false);
            cout<<"A"<<Aaaa.datalength()<<endl;
            Aaaa.printtensor();

            cout<<"column"<<endl;
            size_t exta[1];
            size_t stra[1];
            datastruct<double>Aa=A.column(1,exta,stra);
            cout <<"C"<<endl;
            Aa.printtensor();

            size_t exta2[2],stra2[2];
            datastruct<double>Ab= A.subspanmatrix(1,1,2,4,exta2,stra2);
            cout<<"subspanmatrixA"<<endl;
            Ab.printtensor();


            size_t exta3[1],stra3[1];
            double newda3[7];
            cout<<"column 1 of A with data c"<<endl;
            datastruct<double>Ac=A.column_s(1, exta3,stra3,newda3);
            Ac.printtensor();


            size_t exta4[2],stra4[2];
            double newda4[8];
            datastruct<double>Ad= A.subspanmatrix_s(1,1,2,4,exta4,stra4,newda4);
            cout<<"subspanmatrixA with data copy"<<endl;
            Ad.printtensor();
//
            size_t exta5[2],stra5[2];
            datastruct<double>Ae= A.transpose(exta5,stra5);
            cout<<"transpose"<<endl;
            Ae.printtensor();
//
            size_t exta6[2],stra6[2];
            double dataa6[21];
            datastruct<double>Af= A.transpose_s(exta6,stra6,dataa6);
            cout<<"transpose with data copy"<<endl;
            Af.printtensor();
//
//

            std::vector<double> data_rowmajor =
            {
                //   block 0 (first 3x4 matrix)
                1,2,3,4,
                5,6,7,8,
                9,10,11,12,
                // block 1 (second 3x4 matrix)
                13,14,15,16,
                17,18,19,20,
                21,22,23,24
            };

            size_t extents[3] = {2,3,4};
            size_t strides[3]; // will be computed

            datastruct<double> T_row(data_rowmajor.data(),
                                     data_rowmajor.size(),
                                     true, // row-major
                                     3,    // rank
                                     extents,
                                     strides,
                                     true, // compute_datalength
                                     true, // compute_strides
                                     false // data is not device ptr
                                    );

            size_t offsets[3]    = {1,0,0}; // start at block 1
            size_t sub_extents[3]= {1,3,4}; // take 1 block of full 3x4
            size_t sub_strides[3];
            datastruct<double> subT_view =T_row.subspan_v(offsets, sub_extents, sub_strides);

            std::cout << "Subtensor view (row-major):\n";
            subT_view.printtensor();


            double buffer1[12];
            datastruct<double> subC_view2A =T_row.subspan_v(offsets, sub_extents, sub_strides,buffer1);

            std::cout << "Subtensor view (row-major) with buffer:\n";
            subC_view2A.printtensor();

            size_t num_dims = subC_view2A.count_noncollapsed_dims();
            size_t* extentsA = new size_t[num_dims];
            size_t* stridesA = new size_t[num_dims];

            datastruct<double> coll=subC_view2A.collapsed_view(num_dims,extentsA, stridesA);
            std::cout<<"with collapsed dims"<<endl;
            coll.printtensor();

            delete[]extentsA;
            delete[]stridesA;


            vector<double> B_data_colmajor =
            {
                1, 8, 15,
                2, 9, 16,
                3, 10, 17,
                4, 11, 18,
                5, 12, 19,
                6, 13, 20,
                7, 14, 21
            };



            mdspan<double, std::vector<size_t>> B(B_data_colmajor.data(), false, rows, cols);
            cout<<"B"<<endl;
            B.printtensor();

//
            size_t extbb[2]= {3,7};
            size_t strbb[2];
            datastruct<double>Bbb(B_data_colmajor.data(),21, false,2,extbb,strbb,true,true,false);
            cout<<"B"<< Bbb.datalength()<<endl<<endl;
            Bbb.printtensor();
            cout<<"B"<< Bbb.datalength()<<endl<<endl;
            size_t extbbb[2];
            size_t strbbb[2];
            datastruct<double>Bbbb(B_data_colmajor.data(),0, false,3,7,extbbb,strbbb,true,true,false);
            Bbbb.printtensor();

            cout<<"column"<<endl;

            size_t extb[1],strb[1];
            datastruct<double>Ba= B.column(1,extb,strb);
            Ba.printtensor();
            size_t extb2[2],strb2[2];


            cout<<"subspanmatrx B"<<endl;
            datastruct<double>Bb= B.subspanmatrix(1,1,2,4,extb2,strb2);
            Bb.printtensor();

            size_t extb3[1],strb3[1];
            double newdb3[7];
            cout<< "column1  of B with data copy"<<endl;

            datastruct<double> Bc=B.column_s(1, extb3,strb3,  newdb3);
            Bc.printtensor();

            size_t extb4[2],strb4[2];
            double newdb4[8];
            datastruct<double>Bd= B.subspanmatrix_s(1,1,2,4,extb4,strb4,newdb4);
            cout<<"subspanmatrixB with data copy"<<endl;
            Bd.printtensor();

            size_t extb5[2],strb5[2];
            datastruct<double>Be= B.transpose(extb5,strb5);
            cout<<"transpose"<<endl;
            Be.printtensor();

            size_t extb6[2],strb6[2];
            double datab6[12];
            datastruct<double>Bf= B.transpose_s(extb6,strb6,datab6);
            cout<<"transpose with data copy"<<endl;
            Bf.printtensor();
////
//
            std::vector<double> data_colmajor =
            {
                1,13,
                2,14,
                3,15,
                4,16,

                5,17,
                6,18,
                7,19,
                8,20,

                9,21,
                10,22,
                11,23,
                12,24
            };

            size_t extentsC[3] = {2,3,4};
            size_t stridesC[3];

            datastruct<double> T_col(data_colmajor.data(),
                                     data_colmajor.size(),
                                     false, // column-major
                                     3,
                                     extentsC,
                                     stridesC,
                                     true,
                                     true,
                                     false);

            size_t offsetsC[3]     = {1,0,0};
            size_t sub_extentsC[3] = {1,3,4};
            size_t sub_stridesC[3];

            datastruct<double> subC_view =
                T_col.subspan_v(offsetsC, sub_extentsC, sub_stridesC);
            std::cout << "Subtensor view (col-major):\n";
            subC_view.printtensor();

            double buffer[12];
            datastruct<double> subC_view2 =T_col.subspan_v(offsetsC, sub_extentsC, sub_stridesC,buffer);
            std::cout << "Subtensor view (col-major) with buffer:\n";
            subC_view2.printtensor();



            size_t num_dimsB = subC_view2.count_noncollapsed_dims();
            size_t* extentsB = new size_t[num_dimsB];
            size_t* stridesB = new size_t[num_dimsB];

            datastruct<double> collB=subC_view2.collapsed_view(num_dims,extentsB, stridesB);
            std::cout<<"with collapsed dims"<<endl;
            collB.printtensor();

            delete[]extentsB;
            delete[]stridesB;


        }

        cout<< "Now some tests whether the library accepts row and column major data and can extract rows and columns with the same code. "<<endl;
        cout<<" Note that this tests only mdspan class. it owns strides and extents, the mdspan_data class owns the data as well"<<endl<<endl;
        {

            vector<double>A_data(3*7,0);
            A_data = {1,2,3,4,5,6,7,
                      8,9,10,11,12,13,14,
                      15,16,17,18,19,20,21
                     };

            size_t rows=3,cols=7;
            mdspan<double,array<size_t,2>> A(A_data.data(), true, rows, cols);
            cout<<"A"<<endl;
            A.printtensor();

            cout<<"row 1"<<endl;

            mdspan<double, std::vector<size_t>> Aa=A.row(1);
            cout <<"C"<<endl;
            Aa.printtensor();
//

            mdspan<double, std::array<size_t,2>> Ab= A.subspanmatrix(1,1,2,4);
            std::cout<<Ab.rank();
            cout<<"subspanmatrixA"<<endl;


//
//
//
            Ab.printtensor();

////
//
            double newda3[7];
            cout<<"column 1 of A with data c"<<endl;
            mdspan<double, std::vector<size_t>> Ac=A.column(1,newda3);
            Ac.printtensor();



////
////
//
            double newda4[8];
            mdspan<double, std::array<size_t,2>> Ad= A.subspanmatrix(1,1,2,4,newda4);
            cout<<"subspanmatrixA with data copy"<<endl;
            Ad.printtensor();
//
//
            mdspan<double, std::array<size_t,2>> Ae= A.transpose();
            cout<<"transpose"<<endl;
            Ae.printtensor();
//
//
            double dataa6[21];
            mdspan<double, std::array<size_t,2>> Af= A.transpose(dataa6);
            cout<<"transpose with data copy"<<endl;
            Af.printtensor();
//
//


            std::vector<double> data_rowmajor =
            {
                // block 0 (first 3x4 matrix)
                1,2,3,4,
                5,6,7,8,
                9,10,11,12,
                // block 1 (second 3x4 matrix)
                13,14,15,16,
                17,18,19,20,
                21,22,23,24
            };

            vector<size_t> extents = {2,3,4};


            mdspan<double, std::vector<size_t>> T_row(data_rowmajor.data(),true, extents );

            vector<size_t> offsets   = {1,0,0};
            vector<size_t> sub_extents= {1,3,4};
            mdspan<double, std::vector<size_t>>  subT_view =T_row.subspan(offsets, sub_extents);

            std::cout << "Subtensor view (row-major):\n";
            subT_view.printtensor();


            double buffer1[12];
            mdspan<double, std::vector<size_t>> subC_view2A =T_row.subspan(offsets, sub_extents,buffer1);

            std::cout << "Subtensor view (row-major) with buffer:\n";
            subC_view2A.printtensor();

            size_t num_dims = subC_view2A.count_noncollapsed_dims();
            size_t* extentsA = new size_t[num_dims];
            size_t* stridesA = new size_t[num_dims];

            datastruct<double> coll=subC_view2A.collapsed_view(num_dims,extentsA, stridesA);
            std::cout<<"with collapsed dims"<<endl;
            coll.printtensor();

            delete[]extentsA;
            delete[]stridesA;




            A.device_upload(true,0);
            mdspan<double, std::array<size_t,2>>ShallowCopyofA=A;


            vector<double> B_data_colmajor =
            {
                1, 8, 15,
                2, 9, 16,
                3, 10, 17,
                4, 11, 18,
                5, 12, 19,
                6, 13, 20,
                7, 14, 21
            };
            mdspan<double, std::vector<size_t>> B(B_data_colmajor.data(), false, rows, cols);
            cout<<"B"<<endl;
            B.printtensor();



            cout<<"column"<<endl;


            mdspan<double, std::vector<size_t>>Ba= B.column(1);
            Ba.printtensor();

            cout<<"subspanmatrx B"<<endl;
            mdspan<double, std::vector<size_t>>Bb= B.subspanmatrix(1,1,2,4);
            Bb.printtensor();

            double newdb3[7];
            cout<< "column1  of B with data copy"<<endl;

            mdspan<double, std::vector<size_t>> Bc=B.column(1, newdb3);
            Bc.printtensor();

            double newdb4[8];
            mdspan<double, std::vector<size_t>>Bd= B.subspanmatrix(1,1,2,4,newdb4);
            cout<<"subspanmatrixB with data copy"<<endl;
            Bd.printtensor();

            mdspan<double, std::vector<size_t>>Be= B.transpose();
            cout<<"transpose"<<endl;
            Be.printtensor();

            double datab6[12];
            mdspan<double, std::vector<size_t>>Bf= B.transpose(datab6);
            cout<<"transpose with data copy"<<endl;
            Bf.printtensor();


            std::vector<double> data_colmajor =
            {
                1,13,
                2,14,
                3,15,
                4,16,

                5,17,
                6,18,
                7,19,
                8,20,

                9,21,
                10,22,
                11,23,
                12,24
            };

            vector<size_t> extentsC = {2,3,4};

            mdspan<double, std::vector<size_t>> T_col(data_colmajor.data(),false,extentsC);
            vector<size_t> offsetsC     = {1,0,0};
            vector<size_t> sub_extentsC = {1,3,4};

            mdspan<double, std::vector<size_t>> subC_view =T_col.subspan(offsetsC, sub_extentsC);
            std::cout << "Subtensor view (col-major):\n";
            subC_view.printtensor();

            double buffer[12];

            mdspan<double, std::vector<size_t>> subC_view2 =T_col.subspan(offsetsC, sub_extentsC,buffer);
            std::cout << "Subtensor view (col-major) with buffer:\n";
            subC_view2.printtensor();


            mdspan<double, std::vector<size_t>> collB=subC_view2.collapsed_view();
            std::cout<<"with collapsed dims"<<endl;
            collB.printtensor();


        }
    }


    {

        vector<double>A_data= {210, -92, 68, -33, -34, -4, 118, -6, -92, 318, -100, 130, -153, -64, 160, 33, 68, -100, 204, -96, 41, -69, -16, -26, -33, 130, -96, 338, -152, -51, 12, 22, -34, -153, 41, -152, 346, 11, -30, -25, -4, -64, -69, -51, 11, 175, -79, 5, 118, 160, -16, 12, -30, -79, 320, 7, -6, 33, -26, 22, -25, 5, 7, 239};

        size_t rows2 = 8, cols2 = 8;

        cout<<endl<<endl<<endl<<endl;
        {

            cout<<"Now a cholesky decomposition on CPU"<<std::endl;
            vector<double>L_data(A_data.size(),0);
            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows2, cols2});
            mdspan<double, std::vector<size_t>> L(L_data.data(), true, {rows2, cols2});

            cout<<"with another dataset"<<endl;

            A.printtensor();
            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);

            Math_Functions<double>::cholesky_decomposition(A,L,&p);

            L.printtensor();

            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            vector<double>verify_data(L_data.size(),0);
            mdspan<double, std::vector<size_t>> verify(verify_data.data(), true, {rows2, cols2});

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            size_t newext[2],newstr[2];

            datastruct<double>m=L.transpose(newext,newstr);
            Math_Functions<double>::matrix_multiply_dot(L,m, verify,&p2);
            verify.printtensor();
        }


        {

            cout<<"Now the cholesky decomposition is entirely done on GPU"<<std::endl;
            vector<double>L_data(A_data.size(),0);
            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows2, cols2});
            mdspan<double, std::vector<size_t>> L(L_data.data(), true, {rows2, cols2});

            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);

            Math_Functions<double>::cholesky_decomposition(A,L,&p);

            L.printtensor();

            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            vector<double>verify_data(L_data.size(),0);
            mdspan<double, std::vector<size_t>> verify(verify_data.data(), true, {rows2, cols2});

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            size_t newext[2],newstr[2];

            datastruct<double>m=L.transpose(newext,newstr);
            Math_Functions<double>::matrix_multiply_dot(L,m, verify,&p2);
            verify.printtensor();

        }

        {

            cout<<"With the advanced algorithms on GPU"<<std::endl;

            vector<double>L_data(A_data.size(),0);

            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows2, cols2});
            mdspan<double, std::vector<size_t>> L(L_data.data(), true, {rows2, cols2});

            A.printtensor();

            Math_MPI_Decomposition_Policy p(
                Math_Functions_Policy::GPU_ONLY,
                false,
                false,
                Math_MPI_Decomposition_Policy::Naive);
            p.size_to_stop_recursion=2;
            Math_Functions_MPI<double>::cholesky_decomposition(A,L,&p);
            L.printtensor();


            cout<<"we can verify the cholesky decomposition by multiplication"<<endl;
            vector<double>verify_data(L_data.size(),0);
            mdspan<double, std::vector<size_t>> verify(verify_data.data(), true, {rows2, cols2});

            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            size_t newext[2],newstr[2];

            datastruct<double>m=L.transpose(newext,newstr);
            Math_Functions<double>::matrix_multiply_dot(L,m, verify,&p2);
            verify.printtensor();

        }

    }


    {

        cout<< "Now we do the same with the lu decomposition"<<std::endl;
        vector<double>A_data= {-3,3,-3,5,2,7,4,2,-2,4,2,-10,-4,-2,-10,1,-3,0,8,6,-3,-8,-8,-10,-6,-1,-4,-2,-4,-2,-3,1,-9,-10,5,-6,-8,1,-3,-8,-10,-8,-6,4,3,-8,-10,-6,3,-4,-2,4,4,-1,2,8,-4,6,9,-7,-6,-4,2,4};
        size_t rows3 = 8, cols3 = 8;

        {
            vector<double>L_data(64,0);
            vector<double>U_data(64,0);

            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows3, cols3});
            mdspan<double, std::vector<size_t>> L(L_data.data(), true, {rows3, cols3});
            mdspan<double, std::vector<size_t>> U(U_data.data(), true, {rows3, cols3});

            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);
            A.printtensor();

            cout<<"on CPU"<<std::endl;

            Math_Functions<double>::lu_decomposition(A,L,U,&p);
            L.printtensor();
            U.printtensor();

            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            vector<double>verify_data(64,0);
            mdspan<double, std::vector<size_t>> verify(verify_data.data(), true, {rows3, cols3});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(L,U, verify,&p2);
            verify.printtensor();

        }

        {



            vector<double>L_data(64,0);
            vector<double>U_data(64,0);


            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows3, cols3});
            mdspan<double, std::vector<size_t>> L(L_data.data(), true, {rows3, cols3});
            mdspan<double, std::vector<size_t>> U(U_data.data(), true, {rows3, cols3});

            cout<<"Entirely on gpu"<<std::endl;
            Math_Functions_Policy p(Math_Functions_Policy::GPU_ONLY);
            Math_Functions<double>::lu_decomposition(A,L,U,&p);
            L.printtensor();
            U.printtensor();

            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            vector<double>verify_data(64,0);
            mdspan<double, std::vector<size_t>> verify(verify_data.data(), true, {rows3, cols3});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(L,U, verify,&p2);
            verify.printtensor();
        }

        {
            vector<double>L_data(64,0);
            vector<double>U_data(64,0);


            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows3, cols3});
            mdspan<double, std::vector<size_t>> L(L_data.data(), true, {rows3, cols3});
            mdspan<double, std::vector<size_t>> U(U_data.data(), true, {rows3, cols3});

            cout<<"With the advanced algorithms on GPU"<<std::endl;

            Math_MPI_Decomposition_Policy p(
                Math_Functions_Policy::GPU_ONLY,
                false,
                false,
                Math_MPI_Decomposition_Policy::Strassen);

            p.size_to_stop_recursion=2;
            Math_Functions_MPI<double>::lu_decomposition(A,L,U,&p);
            L.printtensor();


            cout<<"we can verify the lu decomposition by multiplication"<<endl;
            vector<double>verify_data(64,0);
            mdspan<double, std::vector<size_t>> verify(verify_data.data(), true, {rows3, cols3});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(L,U, verify,&p2);
            verify.printtensor();

        }
    }
    {

        cout<< "Now we do the same with the qr decomposition"<<std::endl;
        vector<double>A_data= {-4, 9, 4, 0, -3, -4, 8, 0, 0, -7, -3, -8, -9, 1, -5, -9, -10, 1, 1, 6, -1, 5, 4, 4, 8, 1, 9, -8, -6, 8, -4, -2, -4, 7, -7, 3, 7, -2, -9, 9, 4, -4, 1, -3, 4, -8, 3, 6, -7, 7, -3, -7, -9, -5, -1, -7, 7, 1, -9, -1, -7, 3, 5, 4};
        size_t rows4 = 8, cols4 = 8;
        {

            vector<double>Q_data(64,0);
            vector<double>R_data(64,0);

            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows4, cols4});
            mdspan<double, std::vector<size_t>> Q(Q_data.data(), true, {rows4, cols4});
            mdspan<double, std::vector<size_t>> R(R_data.data(), true, {rows4, cols4});


            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);
            A.printtensor();

            cout<<"On cpu"<<std::endl;
            Math_Functions<double>::qr_decomposition(A,Q,R,&p);
            Q.printtensor();
            R.printtensor();

            vector<double>verifydata(64,0);

            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            mdspan<double, std::vector<size_t>> verify(verifydata.data(), true, {rows4, cols4});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(Q,R, verify,&p2);
            verify.printtensor();
        }


        {

            vector<double>Q_data(64,0);
            vector<double>R_data(64,0);
            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows4, cols4});
            mdspan<double, std::vector<size_t>> Q(Q_data.data(), true, {rows4, cols4});
            mdspan<double, std::vector<size_t>> R(R_data.data(), true, {rows4, cols4});


            cout<<"On gpu"<<std::endl;
            Math_Functions_Policy p(Math_Functions_Policy::CPU_ONLY);

            Math_Functions<double>::qr_decomposition(A,Q,R,&p);
            Q.printtensor();
            R.printtensor();

            vector<double>verifydata(64,0);

            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            mdspan<double, std::vector<size_t>> verify(verifydata.data(), true, {rows4, cols4});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(Q,R, verify,&p2);
            verify.printtensor();

        }

        {
            cout<<"with the advanced algorithms on gpu "<<std::endl;
            vector<double>Q_data(64,0);
            vector<double>R_data(64,0);

            mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows4, cols4});
            mdspan<double, std::vector<size_t>> Q(Q_data.data(), true, {rows4, cols4});
            mdspan<double, std::vector<size_t>> R(R_data.data(), true, {rows4, cols4});

            Math_MPI_Decomposition_Policy p(
                Math_Functions_Policy::GPU_ONLY,
                false,
                false,
                Math_MPI_Decomposition_Policy::Naive);

            p.size_to_stop_recursion=2;


            Math_Functions_MPI<double>::qr_decomposition(A,Q,R,&p);
            Q.printtensor();
            R.printtensor();
            vector<double>verifydata(64,0);

            cout<<"we can verify the qr decomposition by multiplication"<<endl;
            mdspan<double, std::vector<size_t>> verify(verifydata.data(), true, {rows4, cols4});
            Math_Functions_Policy p2(Math_Functions_Policy::CPU_ONLY);
            Math_Functions<double>::matrix_multiply_dot(Q,R, verify,&p2);
            verify.printtensor();

        }
    }
}





