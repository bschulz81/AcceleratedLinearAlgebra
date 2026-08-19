#ifndef MATHFUNCTIONS
#define MATHFUNCTIONS

#include "datablock.h"

#include "mathfunctionspolicy.h"


class Math_Functions
{
public:
    template<typename T>
    inline static void cholesky_decomposition(const DataBlock<T>& A,DataBlock<T>& L,
            const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static void lu_decomposition(const DataBlock<T>& A,DataBlock<T>& L,DataBlock<T>& U,
                                        const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static void qr_decomposition(const DataBlock<T>& A,DataBlock<T>& Q,DataBlock<T>& R,
                                        const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static void matrix_multiply_dot(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
                                           const T CoefficientB=T(1),const T CoefficientC=T(0),
                                           const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_multiply_dot(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
                                           const Math_Functions_Policy* policy)
    {
        matrix_multiply_dot(A,B,C,T(1),T(0),policy);
    }

    template<typename T>
    inline static void matrix_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
            const T CoefficientA=T(1),const T CoefficientB=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_linear_combination(const DataBlock<T>& A,DataBlock<T>& C,const T CoefficientA=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_add(const DataBlock<T>& A,DataBlock<T>& C,

                                  const Math_Functions_Policy* policy=nullptr)
    {
        matrix_linear_combination(A,C,T(1),T(1),policy);
    }

    template<typename T>
    inline static void matrix_add(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
                                  const Math_Functions_Policy* policy=nullptr)
    {
        matrix_linear_combination(A,B,C,T(1),T(1),T(0),policy);
    }

    template<typename T>
    inline static void matrix_subtract(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,

                                       const Math_Functions_Policy* policy=nullptr)
    {
        matrix_linear_combination(A,B,C,T(1),-T(1),T(0),policy);
    }

    template<typename T>
    inline static void matrix_subtract(const DataBlock<T>& A,DataBlock<T>& C,
                                       const Math_Functions_Policy* policy=nullptr)
    {
        matrix_linear_combination(A,C,-T(1),T(1),policy);
    }

   template<typename T>
    inline static void matrix_multiply_scalar(const DataBlock<T>& M,const T scalar,DataBlock<T>& C,
            const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static void matrix_multiply_scalar(DataBlock<T>& M,const T scalar,
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_multiply_vector(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
            const T Coefficientx=T(1),const T Coefficienty=T(0),
            const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static void matrix_multiply_vector(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
            const Math_Functions_Policy* policy)
    {
        matrix_multiply_vector(A,x,y,T(1),T(0),policy);
    }




    template<typename T>
    inline static void vector_multiply_scalar(const DataBlock<T>& vec,const T scalar,
            DataBlock<T>& res,
            const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static void vector_multiply_scalar(DataBlock<T>& vec,const T scalar,
            const Math_Functions_Policy* policy=nullptr);


    template<typename T>
    inline static T vector_dot_product(const DataBlock<T>& vec1,const DataBlock<T>& vec2,const Math_Functions_Policy* policy=nullptr);



    template<typename T>
    inline static void vector_linear_combination(const DataBlock<T>& vecA,const DataBlock<T>& vecB,DataBlock<T>& vecC,
            const T CoefficientA=T(1),const T CoefficientB=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void vector_linear_combination(const DataBlock<T>& A,DataBlock<T>& C,
                                          const T CoefficientA,const T CoefficientC,
                                          const Math_Functions_Policy* pol);

    template<typename T>
    inline static void vector_add(const DataBlock<T>& vecA,const DataBlock<T>& vecB,DataBlock<T>& vecC,
                                  const Math_Functions_Policy* policy)
    {
        vector_linear_combination(vecA,vecB,vecC,T(1),T(1),T(0),policy);
    }

    template<typename T>
    inline static void vector_subtract(const DataBlock<T>& vecA,const DataBlock<T>& vecB,DataBlock<T>& vecC,
                                       const Math_Functions_Policy* policy)
    {
        vector_linear_combination(vecA,vecB,vecC,T(1),-T(1),T(0),policy);
    }

    template<typename T>
    inline static void vector_add(const DataBlock<T>& vecA,DataBlock<T>& vecC,
                                  const Math_Functions_Policy* policy)
    {
        vector_linear_combination(vecA,vecC,T(1),T(1),policy);
    }

    template<typename T>
    inline static void vector_subtract(const DataBlock<T>& vecA,DataBlock<T>& vecC,
                                       const Math_Functions_Policy* policy)
    {
        vector_linear_combination(vecA,vecC,-T(1),T(1),policy);
    }

    template<typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks,DataBlock<T>& C,
            const T CoefficientB=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_multiply_dot_sparse(
        const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks,DataBlock<T>& C,
        const Math_Functions_Policy* policy)
    {
        matrix_multiply_dot_sparse(Ablocks,Bblocks,C,T(1),T(0),policy);
    }

    template<typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks,const DataBlock<T>& B,DataBlock<T>& C,
            const T CoefficientB=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks,const DataBlock<T>& B,DataBlock<T>& C,
            const Math_Functions_Policy* policy)
    {
        matrix_multiply_dot_sparse(Ablocks,B,C,T(1),T(0),policy);
    }


    template<typename T>
    inline static void matrix_multiply_vector_sparse(const BlockedDataView<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
            const T Coefficientx=T(1),const T Coefficienty=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void matrix_multiply_vector_sparse(const BlockedDataView<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
            const Math_Functions_Policy* policy)
    {
        matrix_multiply_vector_sparse(A,x,y,T(1),T(0),policy);
    }

    template<typename T>
    inline void matrix_multiply_vector_sparse(const BlockedDataView<T>& A,const BlockedDataView<T>& x,DataBlock<T>& y,
        const T Coefficientx,const T Coefficienty,const Math_Functions_Policy* pol);


    template<typename T>
    inline static void matrix_multiply_vector_sparse(const BlockedDataView<T>& A,const BlockedDataView<T>& x,DataBlock<T>& y,
    const Math_Functions_Policy* policy)
    {
        matrix_multiply_vector_sparse(A,x,y,T(1),T(0),policy);
    }

    template<typename T>
    inline static void tensor_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
            const T CoefficientA=T(1),const T CoefficientB=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void tensor_linear_combination(const DataBlock<T>& A,DataBlock<T>& C,const T CoefficientA=T(1),const T CoefficientC=T(0),
            const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void tensor_add(const DataBlock<T>& A,DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr)
    {
        tensor_linear_combination(A,C,T(1),T(1),policy);
    }

    template<typename T>
    inline static void tensor_add(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr)
    {
        tensor_linear_combination(A,B,C,T(1),T(1),T(0),policy);
    }

    template<typename T>
    inline static void tensor_subtract(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr)
    {
        tensor_linear_combination(A,B,C,T(1),-T(1),T(0),policy);
    }

    template<typename T>
    inline static void tensor_subtract(const DataBlock<T>& A,DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr)
    {
        tensor_linear_combination(A,C,-T(1),T(1),policy);
    }

    template<typename T>
    inline static void tensor_multiply_scalar(const DataBlock<T>& M,const T scalar,DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);

    template<typename T>
    inline static void tensor_multiply_scalar(DataBlock<T>& M,const T scalar,const Math_Functions_Policy* policy=nullptr);



    static const Math_Functions_Policy& get_default_policy()
    {
        if (!default_policy.has_value())
        {
            // only construct when needed
            default_policy.emplace(Math_Functions_Policy::AUTO);
        }
        return *default_policy;
    }

    static void set_default_policy(const Math_Functions_Policy& p)
    {
        default_policy = p; // assigns, overwrites if already constructed
    }

    static void reset_default_policy()
    {
        default_policy.reset(); // clear back to "uninitialized"
    }

protected:

    inline static std::optional<Math_Functions_Policy> default_policy;

};


#include "mathfunctions.hpp"


#endif
