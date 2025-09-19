// Main function


#include <iostream>

#include "mdspan_omp.h"
#include "mdspan_data.h"
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

            size_t rows=3,cols=7;
            mdspan<double,array<size_t,2>> A(A_data.data(),  {rows, cols},true);
            cout<<"A"<<endl;
            A.printtensor();

            cout<<"row 1"<<endl;
            mdspan<double, array<size_t,2>> Aa=A.row(1);
            Aa.printtensor();


            mdspan<double, std::array<size_t,2>> Ab= A.subspanmatrix(1,1,2,4);
            std::cout<<Ab.rank();
            cout<<"subspanmatrixA"<<endl;
            Ab.printtensor();

            mdspan<double, std::array<size_t,2>> Ae= A.transpose();
            cout<<"transpose"<<endl;
            Ae.printtensor();












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
            vector<size_t> extents = {2,3,4};


            mdspan<double, std::vector<size_t>> T_row(data_rowmajor.data(), extents,true );
            cout<<"A tensor"<<endl;
            T_row.printtensor();

            vector<size_t> offsets   = {1,0,0};
            vector<size_t> sub_extents= {1,3,4};
            mdspan<double, std::vector<size_t>>  subT_view =T_row.subspan(offsets, sub_extents);

            std::cout << "Subtensor view (row-major):\n";
            subT_view.printtensor();

            size_t num_dims = subT_view.count_noncollapsed_dims();
            size_t* extentsA = new size_t[num_dims];
            size_t* stridesA = new size_t[num_dims];

            datastruct<double> coll=subT_view.collapsed_view(num_dims,extentsA, stridesA);
            std::cout<<"with collapsed dims"<<endl;
            coll.printtensor();

            delete[]extentsA;
            delete[]stridesA;

            cout<<"Upload the data"<<endl;
            A.device_data_upload(true);

            A.printtensor();
            mdspan<double, std::array<size_t,2>>ShallowCopyofA=A;
            A.printtensor();
            cout<<"print Shallow Copy on device"<<endl;
            ShallowCopyofA.printtensor();
            cout<<"change data on host and copy data of A to device"<<endl;
            A_data[0]=42;
            A.device_data_update();
            cout<<"print shallow copy of A on device"<<endl;
            ShallowCopyofA.printtensor();

            cout<<"Verify A is on device"<<A.data_is_devptr()<<endl;
            cout<<"remove A from device";
            A.device_data_download_release();
            cout<<"copy A to host and remove A from device"<<endl;
            cout<<"Verify A is on device"<<A.data_is_devptr()<<endl;


            mdspan<double, std::array<size_t,2>> subspan_of_A= A.subspanmatrix(1,1,2,2);
            cout<<"this is a submatrix of A"<<endl;
            subspan_of_A.printtensor();
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
            mdspan<double, std::vector<size_t>> B(B_data_colmajor.data(), { rows, cols},false);
            cout<<"B"<<endl;
            B.printtensor();



            cout<<"column"<<endl;


            mdspan<double, std::vector<size_t>>Ba= B.column(1);
            Ba.printtensor();
            cout <<"Rank"<<Ba.rank()<<endl;
            cout<<"subspanmatrx B"<<endl;
            mdspan<double, std::vector<size_t>>Bb= B.subspanmatrix(1,1,1,4);
            Bb.printtensor();
            cout <<"Rank"<<Bb.rank()<<endl;




            mdspan<double, std::vector<size_t>>Be= B.transpose();
            cout<<"transpose"<<endl;
            Be.printtensor();




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

            mdspan<double, std::vector<size_t>> T_col(data_colmajor.data(),extentsC,false);
            vector<size_t> offsetsC     = {1,0,0};
            vector<size_t> sub_extentsC = {1,3,4};
            cout <<"Rank"<<T_col.rank()<<endl;
            mdspan<double, std::vector<size_t>> subC_view =T_col.subspan(offsetsC, sub_extentsC);
            std::cout << "Subtensor view (col-major):\n";
            subC_view.printtensor();







        }

    }
    {

        cout<< "This demonstrates some functions of the mdspan data class, which can, in contrast to mdspan, manage and own data."<<endl;
        cout<<"mdpspan_data does not provied shallow copies, for this one has to use the base class of mdspan, to which mdspan_data provides an assignment operator "<<endl;

        {


            vector<double>A_data(3*7,0);
            A_data = {1,2,3,4,5,6,7,
                      8,9,10,11,12,13,14,
                      15,16,17,18,19,20,21
                     };

            size_t rows=3,cols=7;

            cout<<"now rowmajordata on a memmap on harddrive"<<endl;
            mdspan_data<double,array<size_t,2>> mdspan_data_matrix( rows, cols,true,true);

            std::copy(begin(A_data),end(A_data),mdspan_data_matrix.data());

            cout<<"mdspan_data matrix with the data of the Matrix A"<<endl;
            mdspan_data_matrix.printtensor();
            cout<<"mdspan_data row copy"<<endl;
            mdspan_data<double,array<size_t,2>>rowcopy=mdspan_data_matrix.row_copy(1);
            rowcopy.printtensor();
            cout <<"rank:" <<rowcopy.rank();

            cout<<"mdspan_data column copy"<<endl;
            mdspan_data<double,array<size_t,2>>columncopy=mdspan_data_matrix.column_copy(1);
            columncopy.printtensor();
            cout<<"mdspan_data transpose copy on a memmap"<<endl;
            mdspan_data<double,array<size_t,2>>transposecopy=mdspan_data_matrix.transpose_copy(true);
            transposecopy.printtensor();

            cout<<"mdspan_data subspanmatrix copy on a memmap"<<endl;
            mdspan_data<double,array<size_t,2>>subspanmatrixcopy=mdspan_data_matrix.subspanmatrix_copy(1,2,2,2,false);
            subspanmatrixcopy.printtensor();

            cout<<"mdspan_data subspanmatrix copy on a memmap"<<endl;
            array<size_t,2>offs= {1,2};
            array<size_t,2>sub_extents= {2,2};
            mdspan_data<double,array<size_t,2>>subspan=mdspan_data_matrix.subspan_copy(offs,sub_extents,false);
            subspan.printtensor();
            cout<<"copy of mdspan on device";
            mdspan_data<double,array<size_t,2>>newcopy=mdspan_data_matrix.copy(false,true,true,0);

            newcopy.printtensor();
            cout<<"mdspan_data subspanmatrix copy on device"<<endl;

            mdspan_data<double,array<size_t,2>>newcopy_subspan=newcopy.subspanmatrix_copy(1,2,2,2,false);
            newcopy_subspan.printtensor();
            cout<<"verify that the copy has data on device "<<newcopy_subspan.is_dev_ptr()<<endl;


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

            vector<size_t> extents2 = {2,3,4};

            cout<<"We write the tensor as a memmap with rowmajor data"<<endl;
            mdspan_data<double, std::vector<size_t>> Tensor(extents2,true,true);
            std::copy(begin(data_rowmajor),end(data_rowmajor),Tensor.data());
            cout<<"A tensor"<<endl;

            Tensor.printtensor();
            vector<size_t> offsets1   = {1,0,0};
            vector<size_t> sub_extents1= {1,3,4};

            cout<<"now an mdspan_data subtensor"<<endl;
            mdspan_data<double, std::vector<size_t>>  subtensor =Tensor.subspan_copy(offsets1, sub_extents1);
            subtensor.printtensor();




            cout<<"now an mdspan subtensor, which only shallow copies"<<endl;
            mdspan<double, std::vector<size_t>>  subtensor2(Tensor.subspan(offsets1, sub_extents1));
            subtensor2.printtensor();

            cout<<"now we offload that subtensor to gpu"<<endl;
            subtensor2.device_data_upload(true);

            cout<<endl<<"verify that the copy has data on device: "<<subtensor2.is_dev_ptr()<<endl;

            cout<<"now we try to offload the subtensor tensor to gpu, despite a subtensor (i.e. part of the data is alive, and offloaded. "<<endl;
            cout<<"the entire tensor would overlap with the subtensor, so the program should turn out false and forbid the offload"<<endl;
            bool cc=Tensor.device_data_upload(true);
            cout<<endl<<"result of the procedure: "<< cc <<"verify that the Tensor has data on device: "<<Tensor.is_dev_ptr()<<endl;






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
            size_t rowsB=3,colsB=7;
            mdspan_data<double,array<size_t,2>> mdspan_data_matrixB( rowsB, colsB, false,false);

            std::copy(begin(B_data_colmajor),end(B_data_colmajor),mdspan_data_matrixB.data());
            cout<<"mdspan_data matrix with the data of the Matrix B (A in colmajor)"<<endl;
            mdspan_data_matrixB.printtensor();
            cout<<"mdspan_data row copy"<<endl;
            mdspan_data<double,array<size_t,2>>rowcopyB=mdspan_data_matrixB.row_copy(1);
            rowcopyB.printtensor();
            cout <<"rank:" <<rowcopyB.rank();

            cout<<"mdspan_data column copy"<<endl;
            mdspan_data<double,array<size_t,2>>columncopyB=mdspan_data_matrixB.column_copy(1);
            columncopyB.printtensor();
            cout<<"mdspan_data transpose copy on a memmap"<<endl;
            mdspan_data<double,array<size_t,2>>transposecopyB=mdspan_data_matrixB.transpose_copy(true);
            transposecopyB.printtensor();

        }


    }
//
//    }
    {

        cout<< "This demonstrates some functions of the mdspan data class, which can, in contrast to mdspan, manage and own data."<<endl;
        cout<<"mdpspan_data does not provied shallow copies, for this one has to use the base class of mdspan, to which mdspan_data provides an assignment operator "<<endl;

        {


            vector<double>A_data(3*7,0);
            A_data = {1,2,3,4,5,6,7,
                      8,9,10,11,12,13,14,
                      15,16,17,18,19,20,21
                     };

            size_t rows=3,cols=7;

            cout<<"now rowmajordata on a memmap on harddrive"<<endl;
            mdspan_data<double,array<size_t,2>> mdspan_data_matrix( rows, cols,true,true);

            std::copy(begin(A_data),end(A_data),mdspan_data_matrix.data());

            cout<<"mdspan_data matrix with the data of the Matrix A"<<endl;
            mdspan_data_matrix.printtensor();
            cout<<"mdspan_data row copy"<<endl;
            mdspan_data<double,array<size_t,2>>rowcopy=mdspan_data_matrix.row_copy(1);
            rowcopy.printtensor();
            cout <<"rank:" <<rowcopy.rank();

            cout<<"mdspan_data column copy"<<endl;
            mdspan_data<double,array<size_t,2>>columncopy=mdspan_data_matrix.column_copy(1);
            columncopy.printtensor();
            cout<<"mdspan_data transpose copy on a memmap"<<endl;
            mdspan_data<double,array<size_t,2>>transposecopy=mdspan_data_matrix.transpose_copy(true);
            transposecopy.printtensor();

            cout<<"mdspan_data subspanmatrix copy on a memmap"<<endl;
            mdspan_data<double,array<size_t,2>>subspanmatrixcopy=mdspan_data_matrix.subspanmatrix_copy(1,2,2,2,false);
            subspanmatrixcopy.printtensor();

            cout<<"mdspan_data subspanmatrix copy on a memmap"<<endl;
            array<size_t,2>offs= {1,2};
            array<size_t,2>sub_extents= {2,2};
            mdspan_data<double,array<size_t,2>>subspan=mdspan_data_matrix.subspan_copy(offs,sub_extents,false);
            subspan.printtensor();
            cout<<"copy of mdspan on device";
            mdspan_data<double,array<size_t,2>>newcopy=mdspan_data_matrix.copy(false,true,true,0);

            newcopy.printtensor();
            cout<<"mdspan_data subspanmatrix copy on device"<<endl;

            mdspan_data<double,array<size_t,2>>newcopy_subspan=newcopy.subspanmatrix_copy(1,2,2,2,false);
            newcopy_subspan.printtensor();
            cout<<"verify that the copy has data on device "<<newcopy_subspan.is_dev_ptr()<<endl;


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

            vector<size_t> extents2 = {2,3,4};

            cout<<"We write the tensor as a memmap with rowmajor data"<<endl;
            mdspan_data<double, std::vector<size_t>> Tensor(extents2,true,true);
            std::copy(begin(data_rowmajor),end(data_rowmajor),Tensor.data());
            cout<<"A tensor"<<endl;

            Tensor.printtensor();
            vector<size_t> offsets1   = {1,0,0};
            vector<size_t> sub_extents1= {1,3,4};

            cout<<"now an mdspan_data subtensor"<<endl;
            mdspan_data<double, std::vector<size_t>>  subtensor =Tensor.subspan_copy(offsets1, sub_extents1);
            subtensor.printtensor();




            cout<<"now an mdspan subtensor, which only shallow copies"<<endl;
            mdspan<double, std::vector<size_t>>  subtensor2(Tensor.subspan(offsets1, sub_extents1));
            subtensor2.printtensor();

            cout<<"now we offload that subtensor to gpu"<<endl;
            subtensor2.device_data_upload(true);

            cout<<endl<<"verify that the copy has data on device: "<<subtensor2.is_dev_ptr()<<endl;

            cout<<"now we try to offload the subtensor tensor to gpu, despite a subtensor (i.e. part of the data is alive, and offloaded. "<<endl;
            cout<<"the entire tensor would overlap with the subtensor, so the program should turn out false and forbid the offload"<<endl;
            bool cc=Tensor.device_data_upload(true);
            cout<<endl<<"result of the procedure: "<< cc <<"verify that the Tensor has data on device: "<<Tensor.is_dev_ptr()<<endl;

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
            size_t rowsB=3,colsB=7;
            mdspan_data<double,array<size_t,2>> mdspan_data_matrixB( rowsB, colsB, false,false);

            std::copy(begin(B_data_colmajor),end(B_data_colmajor),mdspan_data_matrixB.data());
            cout<<"mdspan_data matrix with the data of the Matrix B (A in colmajor)"<<endl;
            mdspan_data_matrixB.printtensor();
            cout<<"mdspan_data row copy"<<endl;
            mdspan_data<double,array<size_t,2>>rowcopyB=mdspan_data_matrixB.row_copy(1);
            rowcopyB.printtensor();
            cout <<"rank:" <<rowcopyB.rank();

            cout<<"mdspan_data column copy"<<endl;
            mdspan_data<double,array<size_t,2>>columncopyB=mdspan_data_matrixB.column_copy(1);
            columncopyB.printtensor();
            cout<<"mdspan_data transpose copy on a memmap"<<endl;
            mdspan_data<double,array<size_t,2>>transposecopyB=mdspan_data_matrixB.transpose_copy(true);
            transposecopyB.printtensor();

        }

    }
}
