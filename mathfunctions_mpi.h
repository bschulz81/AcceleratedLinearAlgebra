#ifndef MATHFUNCTIONS_MPI
#define MATHFUNCTIONS_MPI

#include "mpi.h"
#include <math.h>


template <typename T>
class DataBlock;

template<typename T>
class DistributedDataBlock;

#include "mathfunctions_mpipolicy.h"

using namespace std;



class Math_Functions_MPI
{
public:

    template <typename T>
    inline static void strassen_multiply(const DataBlock<T> &aA, const DataBlock<T> &aB,DataBlock<T>& aC,MPI_Comm pcomm, const Math_MPI_RecursiveMultiplication_Policy *par=nullptr);
    template <typename T>
    inline static void winograd_multiply(const DataBlock<T> &aA,const DataBlock<T> &aB,DataBlock<T>& aC,MPI_Comm pcomm, const Math_MPI_RecursiveMultiplication_Policy *par=nullptr);
    template <typename T>
    inline static void cholesky_decomposition(const DataBlock<T>& aA, DataBlock<T> & aL, MPI_Comm pcomm, Math_MPI_Decomposition_Policy *par=nullptr);
    template <typename T>
    inline static void lu_decomposition(const DataBlock<T> &aA, DataBlock<T> & aL,DataBlock<T> & aU, MPI_Comm pcomm,Math_MPI_Decomposition_Policy *par=nullptr);
    template <typename T>
    inline static void qr_decomposition(const DataBlock<T> &aA,DataBlock<T>& aQ, DataBlock<T> & aR,MPI_Comm pcomm,    Math_MPI_Decomposition_Policy *par=nullptr);
    template<typename T>
    inline static void MPI_recursive_multiplication_helper( MPI_Comm pcom,const Math_MPI_RecursiveMultiplication_Policy*par=nullptr);
    template<typename T>
    inline static void MPI_recursion_helper_end(MPI_Comm pcomm);


    template<typename T>
    inline static bool matrix_multiply_dot_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const T CoefficientB=T(1),const T CoefficientC=T(0),
        const Math_MPI_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static bool matrix_multiply_dot_Distributed(const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
       return  matrix_multiply_dot_Distributed(A,B,C,T(1),T(0),policy);
    }


    template<typename T>
    inline static bool matrix_multiply_vector_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& x,DistributedDataBlock<T>& y,
        const T Coefficientx=T(1),const T Coefficienty=T(0),
        const Math_MPI_Functions_Policy* policy=nullptr);

     template<typename T>
    inline static bool matrix_multiply_vector_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& x,DistributedDataBlock<T>& y,
        const Math_MPI_Functions_Policy* policy)
        {
           return matrix_multiply_vector_Distributed(A,x,y,T(1),T(0),policy);
        }


    template<typename T>
    inline static bool matrix_linear_combination_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const T CoefficientA=T(1),const T CoefficientB=T(1),const T CoefficientC=T(0),
        const Math_MPI_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static bool matrix_linear_combination_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,
        const T CoefficientA=T(1),const T CoefficientC=T(1),
        const Math_MPI_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static bool matrix_add_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
        return matrix_linear_combination_Distributed(A,C,T(1),T(1),policy);
    }




    template<typename T>
    inline static bool matrix_add_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
       return matrix_linear_combination_Distributed(A,B,C,T(1),T(1),T(0),policy);
    }

    template<typename T>
    inline static bool matrix_subtract_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
       return matrix_linear_combination_Distributed(A,B,C,T(1),-T(1),T(0),policy);
    }


    template<typename T>
    inline static bool vector_linear_combination_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const T CoefficientA=T(1),const T CoefficientB=T(1),const T CoefficientC=T(0),
        const Math_MPI_Functions_Policy* policy=nullptr);



    template<typename T>
    inline static bool vector_add_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
        return vector_linear_combination_Distributed(A,B,C,T(1),T(1),T(0),policy);
    }

    template<typename T>
    inline static bool vector_subtract_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
       return vector_linear_combination_Distributed(A,B,C,T(1),-T(1),T(0),policy);
    }



    template<typename T>
    inline static bool vector_add_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
        return vector_linear_combination_Distributed(A,C,T(1),T(1),policy);
    }

    template<typename T>
    inline static bool vector_subtract_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
        return vector_linear_combination_Distributed(A,C,-T(1),T(1),policy);
    }



    template<typename T>
    inline static bool vector_linear_combination_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,const T CoefficientA=T(1),const T CoefficientC=T(1),
        const Math_MPI_Functions_Policy* policy=nullptr);



    template<typename T>
    inline static bool matrix_multiply_scalar_Distributed(
        const DistributedDataBlock<T>& A,const T scalar,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static bool matrix_multiply_scalar_Distributed(
        DistributedDataBlock<T>& A,const T scalar,
        const Math_MPI_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static bool vector_multiply_scalar_Distributed(
        const DistributedDataBlock<T>& A,const T scalar,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static bool vector_multiply_scalar_Distributed(
        DistributedDataBlock<T>& A,const T scalar,
        const Math_MPI_Functions_Policy* policy=nullptr);



    template<typename T>
    inline static bool vector_dot_product_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,
        int root,T* result,
        const Math_MPI_Functions_Policy* policy=nullptr);

    template <typename T>
    inline static T vector_dot_product_Allreduce_Distributed(const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& B,   const Math_MPI_Functions_Policy* pol = nullptr);

    template <typename T>
    inline static bool vector_dot_product_localblock(const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,T* result,const Math_MPI_Functions_Policy* pol);


    template <typename T>
    inline static MPI_Comm create_summa_communicator(ptrdiff_t br,ptrdiff_t bc,const DataBlock<T>* A,const DataBlock<T>* B,const DataBlock<T>* C,int rootrank = 0, MPI_Comm parent = MPI_COMM_WORLD, SummaGridPolicy policy=SummaGridPolicy::Compatible,bool printgrid=false);

    inline static bool matrix_distribution_is_summa_compatible( ptrdiff_t grid_r, ptrdiff_t grid_c,  ptrdiff_t Pr,   ptrdiff_t Pc);

    template <typename T>
    inline static void conjugate_from_root(  DistributedDataBlock<T>& A,int rootrank, MPI_Comm com);
    template <typename T>
    inline static void  conjugate(  DistributedDataBlock<T>& A);

    template <typename T>
    inline static bool matrix_extents_equal(const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& B, const DistributedDataBlock<T>& C)
    {
        return (A.pglobal_extents[0] == B.pglobal_extents[0] &&
                A.pglobal_extents[1] == B.pglobal_extents[1] &&
                A.pglobal_extents[0] == C.pglobal_extents[0] &&
                A.pglobal_extents[1] == C.pglobal_extents[1] &&
                A.pblock_extents[0] == B.pblock_extents[0] &&
                A.pblock_extents[1] == B.pblock_extents[1] &&
                A.pblock_extents[0] == C.pblock_extents[0] &&
                A.pblock_extents[1] == C.pblock_extents[1]);
    }


    template <typename T>
    inline static bool matrix_extents_equal(const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& B)
    {
        return (A.pglobal_extents[0] == B.pglobal_extents[0]&&
                A.pglobal_extents[1] == B.pglobal_extents[1]&&
                A.pblock_extents[0]  == B.pblock_extents[0]&&
                A.pblock_extents[1]  == B.pblock_extents[1]);
    }

    template <typename T>
    inline static bool vector_extents_equal(const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& B, const DistributedDataBlock<T>& C)
    {
        return (A.pglobal_extents[0] == B.pglobal_extents[0] &&
                A.pglobal_extents[0] == C.pglobal_extents[0] &&
                A.pblock_extents[0] == B.pblock_extents[0] &&
                A.pblock_extents[0] == C.pblock_extents[0]);
    }

    template <typename T>
    inline static bool vector_extents_equal(const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& B )
    {
        return (A.pglobal_extents[0] == B.pglobal_extents[0] &&
                A.pblock_extents[0] == B.pblock_extents[0]);
    }




    template<typename T>
    inline static bool tensor_linear_combination_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const T CoefficientA=T(1),const T CoefficientB=T(1),const T CoefficientC=T(0),
        const Math_MPI_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static bool tensor_linear_combination_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,
        const T CoefficientA=T(1),const T CoefficientC=T(1),
        const Math_MPI_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static bool tensor_add_Distributed(
        const DistributedDataBlock<T>& A,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
        return tensor_linear_combination_Distributed(A,C,T(1),T(1),policy);
    }




    template<typename T>
    inline static bool tensor_add_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
       return tensor_linear_combination_Distributed(A,B,C,T(1),T(1),T(0),policy);
    }

    template<typename T>
    inline static bool tensor_subtract_Distributed(
        const DistributedDataBlock<T>& A,const DistributedDataBlock<T>& B,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy)
    {
       return tensor_linear_combination_Distributed(A,B,C,T(1),-T(1),T(0),policy);
    }

     template<typename T>
    inline static bool tensor_multiply_scalar_Distributed(
        const DistributedDataBlock<T>& A,const T scalar,DistributedDataBlock<T>& C,
        const Math_MPI_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static bool tensor_multiply_scalar_Distributed(
        DistributedDataBlock<T>& A,const T scalar,
        const Math_MPI_Functions_Policy* policy=nullptr);




inline static std::optional<Math_MPI_Decomposition_Policy> default_policy;



    static const Math_MPI_Decomposition_Policy& get_default_policy()
    {
        if (!default_policy.has_value())
        {

            default_policy.emplace(true,Math_Functions_Policy::AUTO,Math_MPI_RecursiveMultiplication_Policy::Naive);
        }
        return *default_policy;
    }

    static void set_default_policy(const Math_MPI_Decomposition_Policy& p)
    {
        default_policy = p;
    }

    static void reset_default_policy()
    {
        default_policy.reset();
    }
protected:
    template <typename T>
    inline static void strassen_multiply_h(const DataBlock<T> &aA,const DataBlock<T> &aB,DataBlock<T>& aC,bool ongpu, bool separate_device_memory,MPI_Comm pcom, const Math_MPI_RecursiveMultiplication_Policy &par);
    template <typename T>
    inline static void winograd_multiply_h(const DataBlock<T> &aA,const DataBlock<T> &aB,DataBlock<T>& aC,bool ongpu, bool separate_device_memory,MPI_Comm pcom, const Math_MPI_RecursiveMultiplication_Policy &par);
    template <typename T>
    inline static void cholesky_decomposition_h(const DataBlock<T>& aA, DataBlock<T> & aL,MPI_Comm pcom,  Math_MPI_Decomposition_Policy &par);
    template <typename T>
    inline static void lu_decomposition_h(const DataBlock<T> &aA, DataBlock<T> & aL,DataBlock<T> & aU, MPI_Comm pcom,Math_MPI_Decomposition_Policy &par);
    template <typename T>
    inline static void qr_decomposition_h(const DataBlock<T> &aA,DataBlock<T>& aQ, DataBlock<T> & aR, MPI_Comm pcom,   Math_MPI_Decomposition_Policy &par);

    template<typename T>
    inline static void scale_local_blocks(T* cdata,const ptrdiff_t* coffsets,const ptrdiff_t* cstrides,const ptrdiff_t* cextents,
                                          const ptrdiff_t numblocks,const T alpha,  const bool ongpu,   const int devnum);



};

#include "mathfunctions_mpi.hpp"
#endif

