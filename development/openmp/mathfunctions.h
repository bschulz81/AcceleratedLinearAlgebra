#ifndef MATHFUNCTIONS
#define MATHFUNCTIONS

#include "mathfunctionspolicy.h"




template <typename T>
class DataBlock;





class Math_Functions
{
public:
    template<typename T>
    inline static void matrix_multiply_dot(const  DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_dot_kahan( const DataBlock<T>& A, const DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_add(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_subtract(const DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_dot_accumulate(const  DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_dot_accumulate_kahan( const DataBlock<T>& A, const DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_add_accumulate(DataBlock<T>& A,const DataBlock<T>& B,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_subtract_accumulate(DataBlock<T>& A, const DataBlock<T>& B,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_vector(const  DataBlock<T>&M,const  DataBlock<T> V, DataBlock<T> C,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_vector(const  DataBlock<T>&M,const T*V,  DataBlock<T> & C, const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_scalar(const   DataBlock<T>& M,const T V, DataBlock<T>& C, const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void matrix_multiply_scalar_accumulate(  DataBlock<T>& M,const T V, const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_add( const DataBlock<T>& vec1,  const DataBlock<T>& vec2, DataBlock<T> & res,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_subtract( const DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,  const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_multiply_scalar_accumulate(  DataBlock<T>& vec,const T scalar,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_multiply_scalar(  DataBlock<T>& vec,const T scalar,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_add_accumulate(  DataBlock<T>& vec1,  const DataBlock<T>& vec2,const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void vector_subtract_accumulate(  DataBlock<T>& vec1, const DataBlock<T>& vec2,  const Math_Functions_Policy* policy=nullptr);
template<typename T>
    inline static T dot_product( const DataBlock<T> &vec1, const DataBlock<T> &vec2, const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void cholesky_decomposition(const DataBlock<T>& A, DataBlock<T> & L, const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void lu_decomposition(const DataBlock<T> &A, DataBlock<T> & L,DataBlock<T> & U, const Math_Functions_Policy* policy=nullptr);
    template<typename T>
    inline static void qr_decomposition(const DataBlock<T> &A,DataBlock<T>& Q, DataBlock<T> & R,  const Math_Functions_Policy* policy=nullptr);



    inline static std::optional<Math_Functions_Policy> default_policy;


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

};





#endif
