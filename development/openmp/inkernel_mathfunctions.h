
#ifndef INKERNELMATHFUNCTIONS
#define INKERNELMATHFUNCTIONS

#include "cmath"
#include "datablockindiceshelperfunctions.hpp"


using namespace std;

template<typename T>
class DataBlock;

template<typename T>
class BlockedDataView;

#pragma omp begin declare target

class In_Kernel_Mathfunctions
{
public:
    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void cholesky_decomposition(const DataBlock<T>& A, DataBlock<T>& L,bool initialize_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void lu_decomposition(const  DataBlock<T>& dA, DataBlock<T>& dL, DataBlock<T>& dU,bool initialize_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void qr_decomposition( const DataBlock<T>&A, DataBlock<T> Q, DataBlock<T> &R,bool initialize_to_zero=true,bool with_memmaps=false);

    template <typename T>
    inline static void cross_product( const DataBlock<T>& vec1,const   DataBlock<T>& vec2, DataBlock<T>& res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot( const DataBlock<T>& A,  const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_accumulate( const DataBlock<T>& A,  const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_accumulate_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_add( const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_add_accumulate( DataBlock<T>& A,const DataBlock<T>& B);


    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_subtract(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_subtract_accumulate( DataBlock<T>& A,const  DataBlock<T>& B);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T> &C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector_kahan( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector_kahan( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_add(const  DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_add_accumulate(  DataBlock<T>& vec1,const  DataBlock<T>& vec2);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_subtract(const  DataBlock<T>& vec1, const  DataBlock<T>& vec2, DataBlock<T> & res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_subtract_accumulate(  DataBlock<T>& vec1,const  DataBlock<T>& vec2);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static T dot_product( const DataBlock<T> &vec1,const  DataBlock<T> &vec2);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static T dot_product_kahan(const  DataBlock<T> &vec1, const DataBlock<T> &vec2);



    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_scalar( const  DataBlock<T>& M, const T V, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_scalar_accumulate(   DataBlock<T>& M, const T V );

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_multiply_scalar_accumulate(  DataBlock<T>& vec,const T scalar);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks, const BlockedDataView<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks, const DataBlock<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static  void matrix_vector_multiply_sparse(const BlockedDataView<T>& A, const DataBlock<T>& x,  DataBlock<T>& y,bool initialize_output_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static  void matrix_vector_multiply_sparse(const BlockedDataView<T>& A, const BlockedDataView<T>& x, DataBlock<T>& y,bool initialize_output_to_zero=true);

    template <typename T>
    inline static T kahan_sum(const T *arr,ptrdiff_t n);

    template <typename T>
    inline static T neumaier_sum(const T*arr,ptrdiff_t n);
};
#pragma omp end declare target

#include "inkernel_mathfunctions.hpp"
#endif
