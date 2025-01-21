
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

#include "mdspan_acc.h"

using namespace std;




// Main function
int main()
{

size_t rows = 4, cols = 4;

// Define some data for matrix multiplication A*B=C;
vector<double>A_data(16,0);
vector<double>B_data(16,0);
vector<double>C_data(16,0);

   // Allocate data
    for (size_t i = 0; i < rows * cols; ++i)
    {
        A_data[i] = i + 1; // Example initialization
        B_data[i] = i ; // Example initialization
    }

//the same code base can have the strides and extents on heap(vector) or on the stack(array)
//Here, we define row-major data (column major support is included but was not tested).
//the array class has many constructors for various initialization methods and supports higher ranks than 2,
// but this was just tested in the beginning and has currently not much support.
mdspan<double, std::vector<size_t>> A(A_data.data(), true, {rows, cols});
mdspan<double, std::array<size_t,2>> B(B_data.data(), true, {rows, cols});
mdspan<double, std::vector<size_t>> C(C_data.data(), true, {rows, cols});


//simple matrix multiplication, if the boolean flag is set to true, the multiplication is done on gpu.
cout<<"Ordinary matrix multiplication, on gpu"<<std::endl;
matrix_multiply_dot(A, B, C,true);

printmatrix(C);




//Another set of data
vector<double>A2_data={4,12,-16,12,37,-43,-16,-43,98};
vector<double>L2_data(9,0);
size_t rows2 = 3, cols2 = 3;


//we want to do a cholesky decomposition with it

mdspan<double, std::vector<size_t>> A2(A2_data.data(), true, {rows2, cols2});
mdspan<double, std::vector<size_t>> L2(L2_data.data(), true, {rows2, cols2});
cout<<"A Cholesky decomposition with the multiplication on gpu"<<std::endl;

//the cholesky decomposition involves a matrix multiplication.
//the parameter par can select a multiplication algorithm. by default it is the naive one.
// but one can also set the strassen or the winograd variand of strassen's algorithm,
//and advise it to use temporary files for intermediate results. Also, one can set the size for when orfinary multiplication should be used.
// the last boolean flag indicates that the multiplication should be done on gpu.
//if strassen's or Winograd's variant was selected, then, if the matrix sizes are small enough, the ordinary multiplication will also be done on gpu.
// the step size (here set to 0, is a parameter from https://arxiv.org/abs/1812.02056. if it is zero, an optimal value for strassen's algorithm will be chosen)
printmatrix(A2);
matrix_multiplication_parameters par2;
cholesky_decomposition(A2,L2,par2,0,false,true);

printmatrix(L2);

//the first boolean flag implies that the entire cholesky decomposition (not just the multiplication) is now done on gpu
cout<<"Now the cholesky decomposition is entirely done on gpu"<<std::endl;
cholesky_decomposition(A2,L2,par2,0,true,true);

printmatrix(L2);

cout<< "Now we do the same with the lu decomposition, entirely on gpu"<<std::endl;
vector<double>A3_data={1,-2,-2,-3,3,-9,0,-9,-1,2,4,7,-3,-6,26,2};
vector<double>L3_data(16,0);
vector<double>U3_data(16,0);
size_t rows3 = 4, cols3 = 4;
mdspan<double, std::vector<size_t>> A3(A3_data.data(), true, {rows3, cols3});
mdspan<double, std::vector<size_t>> L3(L3_data.data(), true, {rows3, cols3});
mdspan<double, std::vector<size_t>> U3(U3_data.data(), true, {rows3, cols3});
matrix_multiplication_parameters par3;
printmatrix(A3);
cout<<"Entirely on gpu"<<std::endl;
lu_decomposition(A3,L3,U3,par3,0,true,true);
printmatrix(L3);
printmatrix(U3);

cout<<"Just the multiplication on gpu"<<std::endl;
lu_decomposition(A3,L3,U3,par3,0,false,true);
printmatrix(L3);
printmatrix(U3);


cout<< "Now we do the same with the qr decomposition"<<std::endl;

vector<double>A4_data={12,-51,4,6,167,-68,-4,24,-41};
vector<double>Q4_data(9,0);
vector<double>R4_data(9,0);
size_t rows4 = 3, cols4 = 3;

mdspan<double, std::vector<size_t>> A4(A4_data.data(), true, {rows4, cols4});
mdspan<double, std::vector<size_t>> Q4(Q4_data.data(), true, {rows4, cols4});
mdspan<double, std::vector<size_t>> R4(R4_data.data(), true, {rows4, cols4});
matrix_multiplication_parameters par4;
printmatrix(A4);

cout<<"Just the multiplication on gpu"<<std::endl;

qr_decomposition(A4,Q4,R4,par4,0,false,true);
printmatrix(Q4);
printmatrix(R4);

cout<<"Entirely on gpu"<<std::endl;
qr_decomposition(A4,Q4,R4,par4,0,true,false);
printmatrix(Q4);
printmatrix(R4);


cout<<"In order to test the qr decomposition, we can use Strassen's algorith"<<endl;

vector<double>C4_data(9,0);
mdspan<double, std::vector<size_t>> C4(C4_data.data(), true, {rows4, cols4});

strassen_multiply(Q4, R4, C4,par4, true);



printmatrix(C4);
cout<<"or its Winograd variant, with the smaller matrices computed on gpu"<<endl;
winograd_multiply(Q4, R4, C4,par4, true);

printmatrix(C4);


}
