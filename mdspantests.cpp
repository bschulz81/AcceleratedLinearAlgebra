


#include <iostream>
#include "datablock.h"


#include "mdspan_omp.h"
#include "mdspan_data.h"


#include "mathutilitiesdatablock.h"
#include "mathutilitiesmdspan.h"



int main()
{

    {

        cout<< "Now some tests whether the library accepts row and column major data and can extract rows and columns with the same code. "<<endl;
        cout<<" Note that this tests only mdspan_class. it owns strides and extents, the mdspan_data class owns the data as well"<<endl<<endl;
        {

            vector<double>A_data(3*7,0);
            A_data = {1,2,3,4,5,6,7,
                      8,9,10,11,12,13,14,
                      15,16,17,18,19,20,21
                     };

            ptrdiff_t rows=3,cols=7;
            cout<<"with the create_matrix function"<<endl;
            auto A = mdspan_utilities::create_matrix<double,dynamic_tag>(A_data.data(), rows, cols, DataBlockConfig{});

            cout<<"A"<<endl;
            A.print();
//
            cout<<"with the constructor"<<endl;
            mdspan_t<double, dynamic_tag>  A3 (A_data.data(), {rows, cols}, DataBlockConfig{});
            cout<<"A2"<<endl;
            A3.print();
//
            cout<<"instead of this long designation, there is also a type mdspan_t with a dynamic tag for the vector constructor and a static tag for the array constructor"<<endl;
            mdspan_t<double, dynamic_tag> A4(A_data.data(), {rows, cols}, DataBlockConfig{});
            mdspan_t<double, static_tag<2>> A5(A_data.data(), {rows, cols}, DataBlockConfig{});


            cout<<"row 1"<<endl;
            auto Aa=mdspan_utilities::matrix_row(A4,1);
            Aa.print();
//

            auto Ab= mdspan_utilities::matrix_subspan(A,1,1,2,4);
            std::cout<<Ab.rank();
            cout<<"matrix_subspanA"<<endl;
            Ab.print();
//
            auto Ae= mdspan_utilities::matrix_transpose(A);
            cout<<"transpose"<<endl;
            Ae.print();
//

//
            std::vector<double> data_rowmajor =
            {
//                 block 0 (first 3x4 matrix)
                1,2,3,4,
                5,6,7,8,
                9,10,11,12,
                //               block 1 (second 3x4 matrix)
                13,14,15,16,
                17,18,19,20,
                21,22,23,24
            };
//
            vector<ptrdiff_t> extents = {2,3,4};


            mdspan_t<double, dynamic_tag> T_row(data_rowmajor.data(), extents,DataBlockConfig{} );
            cout<<"A tensor"<<endl;
            T_row.print();

            vector<ptrdiff_t> offsets   = {1,0,0};
            vector<ptrdiff_t> sub_extents= {1,3,4};
            auto  subT_view =mdspan_utilities::tensor_subspan(T_row,offsets, sub_extents);

            std::cout << "Subtensor view (row-major):\n";
            subT_view.print();

            ptrdiff_t num_dims = DataBlockUtilities::count_noncollapsed_dims(subT_view);
            ptrdiff_t * extentsA = new ptrdiff_t[num_dims];
            ptrdiff_t* stridesA = new ptrdiff_t[num_dims];

            DataBlock<double> coll=DataBlockUtilities::collapsed_view(subT_view,num_dims,extentsA, stridesA);
            std::cout<<"with collapsed dims"<<endl;
            coll.print();

            delete[]extentsA;
            delete[]stridesA;

            cout<<"Upload the data"<<endl;
            A.device_data_upload(true);

            A.print();
            auto ShallowCopyofA=A;
            A.print();
            cout<<"print Shallow Copy on device"<<endl;
            ShallowCopyofA.print();
            cout<<"change data on host and copy data of A to device"<<endl;
            A_data[0]=42;
            A.device_data_update();
            cout<<"print shallow copy of A on device"<<endl;
            ShallowCopyofA.print();

            cout<<"Verify A is on device"<<A.data_is_devptr()<<endl;
            cout<<"remove A from device";
            A.device_data_download_release();
            cout<<"copy A to host and remove A from device"<<endl;
            cout<<"Verify A is on device"<<A.data_is_devptr()<<endl;


            auto subspan_of_A= mdspan_utilities::matrix_subspan(A,1,1,2,2);
            cout<<"this is a submatrix of A"<<endl;
            subspan_of_A.print();
            cout<<"now we offload this submatrix"<<endl;
            subspan_of_A.device_data_upload(true);

            cout<<"now we try to offload the tensor A. this would habe an overlap with the submatrix, so should be stopped by the library"<<endl;
            bool b=A.device_data_upload(true);
            cout<<"verify if the entire tensor A is on device. Would forbidden by the openmp standard."<<endl;
            cout<<"offload procedure returned"<< b<<"Verify A is on device"<<A.data_is_devptr()<<endl;

            cout<< "now we download the submatrix of A and delete it on device"<<endl;
            subspan_of_A.device_data_download_release();

            cout<<"now we try to offload A again. this should now work"<<endl;
            bool bb=A.device_data_upload(true);
            cout<<"verify if the entire tensor A is on device. now this should work."<<endl;
            cout<<"offload procedure returned"<< bb<<"Verify A is on device"<<A.data_is_devptr()<<endl;



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


            auto B = mdspan_utilities::create_matrix<double, dynamic_tag>(B_data_colmajor.data(), rows, cols, DataBlockConfig{.dprowmajor = false});

            cout<<"B"<<endl;
            B.print();



            cout<<"column"<<endl;


            auto Ba= mdspan_utilities::matrix_column(B,1);
            Ba.print();
            cout <<"Rank"<<Ba.rank()<<endl;
            cout<<"subspanmatrx B"<<endl;
            auto Bb= mdspan_utilities::matrix_subspan(B,1,1,1,4);
            Bb.print();
            cout <<"Rank"<<Bb.rank()<<endl;




            auto Be= mdspan_utilities::matrix_transpose(B);
            cout<<"transpose"<<endl;
            Be.print();




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

            vector<ptrdiff_t> extentsC = {2,3,4};

            mdspan_t<double, dynamic_tag> T_col(data_colmajor.data(),extentsC,DataBlockConfig{.dprowmajor = false});
            vector<ptrdiff_t> offsetsC     = {1,0,0};
            vector<ptrdiff_t> sub_extentsC = {1,3,4};
            cout <<"Rank"<<T_col.rank()<<endl;
            auto subC_view =mdspan_utilities::tensor_subspan(T_col,offsetsC, sub_extentsC);
            std::cout << "Subtensor view (col-major):\n";
            subC_view.print();

        }

    }
    {

        cout<< "This demonstrates some functions of the mdspan data class, which can, in contrast to mdspan, manage and own data."<<endl;
        cout<<"mdpspan_data does not provied shallow copies, for this one has to use the base class of mdspan, to which mdspan_data provides an assignment operator "<<endl;

        {

//
            vector<double>A_data(3*7,0);
            A_data = {1,2,3,4,5,6,7,
                      8,9,10,11,12,13,14,
                      15,16,17,18,19,20,21
                     };

            ptrdiff_t rows=3,cols=7;

            cout<<"now rowmajordata on a memmap on harddrive, creation with factory function"<<endl;
            auto mdspan_data_matrix = mdspan_utilities::create_matrix<double,static_tag<2>>(rows, cols, ManagedDataBlockConfig{.memmap=true});

            cout<<"the utility function created an empty matrix. We fill it now by copying in the data field"<<endl;

            std::copy(begin(A_data),end(A_data),mdspan_data_matrix.data());
            mdspan_data_matrix.print();


            cout<<"creation with the constructor"<<endl;
            mdspan_data_t <double, dynamic_tag>  mdspan_data_matrix2 ({rows, cols}, ManagedDataBlockConfig{.memmap=true});
            cout<<"the constructor created an empty matrix. We fill it now by copying in the data field"<<endl;
            std::copy(begin(A_data),end(A_data),mdspan_data_matrix2.data());
            mdspan_data_matrix2.print();



            cout<<"instead of this long designation, there is also a type mdspan_t with a dynamic tag for the vector constructor and a static tag for the array constructor"<<endl;
            mdspan_data_t <double, dynamic_tag>  mdspan_data_matrix3 ({rows, cols}, ManagedDataBlockConfig{.memmap=true});
            mdspan_data_t <double,static_tag<2>>  mdspan_data_matrix4 ({rows, cols}, ManagedDataBlockConfig{.memmap=true});

//




            cout<<"copy of mdspan on device";
            ManagedDataBlockConfig m={.data_ondevice=true};
            auto newcopy=mdspan_data_matrix.copy(&m);

            newcopy.print();

            cout<<"verify that the copy has data on device "<<newcopy.data_is_devptr()<<endl;


            cout<<"define a tensor"<<endl;
            std::vector<double> data_rowmajor =
            {
                //   block 0 (first 3x4 matrix)
                1,2,3,4,
                5,6,7,8,
                9,10,11,12,
                //  block 1 (second 3x4 matrix)
                13,14,15,16,
                17,18,19,20,
                21,22,23,24
            };

            vector<ptrdiff_t> extents2 = {2,3,4};

            cout<<"We write the tensor as a memmap with rowmajor data"<<endl;
            mdspan_data_t<double,dynamic_tag> Tensor(extents2, ManagedDataBlockConfig{.memmap=true});
            std::copy(begin(data_rowmajor),end(data_rowmajor),Tensor.data());
            cout<<"A tensor"<<endl;

            Tensor.print();
            vector<ptrdiff_t> offsets1   = {1,0,0};
            vector<ptrdiff_t> sub_extents1= {1,3,4};


            cout<<"now an mdspan subtensor, which only shallow copies"<<endl;
            auto  subtensor2(mdspan_utilities::tensor_subspan(Tensor,offsets1, sub_extents1));
            subtensor2.print();

            cout<<"now we offload that subtensor to gpu"<<endl;
            subtensor2.device_data_upload(true);

            cout<<endl<<"verify that the copy has data on device: "<<subtensor2.data_is_devptr()<<endl;

            cout<<"now we try to offload the subtensor tensor to gpu, despite a subtensor (i.e. part of the data is alive, and offloaded. "<<endl;
            cout<<"the entire tensor would overlap with the subtensor, so the program should turn out false and forbid the offload"<<endl;
            bool cc=Tensor.device_data_upload(true);
            cout<<endl<<"result of the procedure: "<< cc <<"verify that the Tensor has data on device: "<<Tensor.data_is_devptr()<<endl;






        }
        {
            cout<<"Now tests with a column major tensor"<< endl;
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



            cout<<"We test the same tensor as column major data"<<endl;
            ptrdiff_t rowsB=3,colsB=7;
            auto mdspan_data_matrixB=mdspan_utilities::create_matrix<double,static_tag<2>> ( rowsB, colsB,  ManagedDataBlockConfig({.dprowmajor=false}));

            std::copy(begin(B_data_colmajor),end(B_data_colmajor),mdspan_data_matrixB.data());
            cout<<"mdspan_data matrix with the data of the Matrix B (A in colmajor)"<<endl;
            mdspan_data_matrixB.print();


            cout<<"mdspan_data transpose"<<endl;
            auto transpose=mdspan_utilities::matrix_transpose(mdspan_data_matrixB);
            transpose.print();

            cout<<"copy of a view of mdspan_data with help of a constructor";
            ManagedDataBlockConfig m2={.memmap=false,.data_ondevice=true};
            mdspan_data_t<double,dynamic_tag>newcopy(transpose,&m2);

            newcopy.print();
            cout<<"on device:" <<newcopy.data_is_devptr();

        }
    }

}
