#ifndef EXPRESSION_TEMPLATES
#define EXPRESSION_TEMPLATES

#include <optional>
#include <type_traits>
#include <complex>
#include "mathfunctions.h"
#include "mdspan_omp.h"
#include "datablock.h"
// 1. Scalar Validation Traits
namespace expr
{

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template<typename T>
concept ValidNumericType = std::is_arithmetic_v<T> || is_complex_v<T>;


template<typename LHS, typename RHS>
struct AddExpr;

template<typename LHS, typename RHS>
struct SubtrExpr;

template<typename LHS, typename RHS>
struct MulExpr;

template<typename LHS, typename Scalar>
struct ScaleExpr;

template<typename LHS, typename RHS>
struct DotExpr;


// 3. Structural Type Identification Helpers
template<typename T>
struct is_datablock_type {
private:
    template <typename U> static std::true_type  test(const DataBlock<U>*);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<const T*>()))::value;
};

template<typename T>
inline constexpr bool is_datablock_type_v = is_datablock_type<std::remove_cvref_t<T>>::value;

// Catch-all trait to identify our expression structures or data containers safely
template<typename T>
struct is_expr_type : std::false_type {};
template<typename L, typename R> struct is_expr_type<AddExpr<L, R>> : std::true_type {};
template<typename L, typename R> struct is_expr_type<SubtrExpr<L, R>> : std::true_type {};
template<typename L, typename R> struct is_expr_type<MulExpr<L, R>> : std::true_type {};
template<typename L, typename S> struct is_expr_type<ScaleExpr<L, S>> : std::true_type {};
template<typename L, typename R> struct is_expr_type<DotExpr<L, R>> : std::true_type {};

template<typename T>
concept IsValidMathOperand = is_datablock_type_v<T> || is_expr_type<std::remove_cvref_t<T>>::value;


// 4. Expression Structures
template<typename LHS, typename RHS>
struct AddExpr
{
    const LHS& lhs;
    const RHS& rhs;
    const Math_Functions_Policy* policy = nullptr;

    template<typename T>
    void assign_to(DataBlock<T>& C, const Math_Functions_Policy* override = nullptr) const
    {
        auto pol = override ? override : policy;

        if (lhs.ObjectType() == DataBlock<T>::Matrix)
        {
            Math_Functions::matrix_add(lhs, rhs, C, pol);
        }
        else if (lhs.ObjectType() == DataBlock<T>::Vector)
        {
            Math_Functions::vector_add(lhs, rhs, C, pol);
        }
        else
        {
            throw std::runtime_error("Unsupported type for addition");
        }
    }
};

template<typename LHS, typename RHS>
struct SubtrExpr
{
    const LHS& lhs;
    const RHS& rhs;
    const Math_Functions_Policy* policy = nullptr;

    template<typename T>
    void assign_to(DataBlock<T>& C, const Math_Functions_Policy* override = nullptr) const
    {
        auto pol = override ? override : policy;

        if (lhs.ObjectType() == DataBlock<T>::Matrix)
            Math_Functions::matrix_subtract(lhs, rhs, C, pol);
        else if (lhs.ObjectType() == DataBlock<T>::Vector)
            Math_Functions::vector_subtract(lhs, rhs, C, pol);
        else
            throw std::runtime_error("Unsupported type for subtraction");
    }
};

template<typename LHS, typename Scalar>
struct ScaleExpr
{
    const LHS& lhs;
    const Scalar scalar;
    const Math_Functions_Policy* policy = nullptr;

    template<typename T>
    void assign_to(DataBlock<T>& C, const Math_Functions_Policy* override = nullptr) const
    {
        auto pol = override ? override : policy;

        switch(lhs.ObjectType())
        {
            case DataBlock<T>::Vector:
                Math_Functions::vector_multiply_scalar(lhs, scalar, C, pol);
                break;
            case DataBlock<T>::Matrix:
                Math_Functions::matrix_multiply_scalar(lhs, scalar, C, pol);
                break;
            default:
                throw std::runtime_error("Unsupported type for scalar multiplication");
        }
    }
};

template<typename LHS, typename RHS>
struct MulExpr
{
    const LHS& lhs;
    const RHS& rhs;
    const Math_Functions_Policy* policy = nullptr;

    template<typename T>
    void assign_to(DataBlock<T>& C, const Math_Functions_Policy* override = nullptr) const
    {
        auto pol = override ? override : policy;

        if (lhs.ObjectType() == DataBlock<T>::Matrix)
        {
            if (rhs.ObjectType() == DataBlock<T>::Matrix)
                Math_Functions::matrix_multiply_dot(lhs, rhs, C, pol);
            else if (rhs.ObjectType() == DataBlock<T>::Vector)
                Math_Functions::matrix_multiply_vector(lhs, rhs, C, pol);
            else
                throw std::runtime_error("Unsupported RHS for matrix multiplication");
        }
        else if (lhs.ObjectType() == DataBlock<T>::Vector && rhs.ObjectType() == DataBlock<T>::Vector)
        {
            throw std::runtime_error("Dot product is scalar, use dot() or eval_scalar()");
        }
        else
        {
            throw std::runtime_error("Unsupported type combination for multiplication");
        }
    }
};

template<typename LHS, typename RHS>
struct DotExpr {
    const LHS& lhs;
    const RHS& rhs;
    const Math_Functions_Policy* policy = nullptr;

    template<typename T>
    T eval_scalar(const Math_Functions_Policy* override = nullptr) const
    {
        auto pol = override ? override : policy;
        if (lhs.ObjectType() == DataBlock<T>::Vector && rhs.ObjectType() == DataBlock<T>::Vector)
        {
            return Math_Functions::dot_product(lhs, rhs, pol);
        }
        throw std::runtime_error("DotExpr only works for vectors");
    }

    template<typename T>
    operator T() const {
        return eval_scalar<T>();
    }
};


// 5. Global Clean Operator Overloads (Constrained via robust SFINAE/Traits)
template<typename LHS, typename Scalar>
requires IsValidMathOperand<LHS> && ValidNumericType<std::remove_cvref_t<Scalar>>
auto operator*(const LHS& lhs, Scalar scalar)
{
    return ScaleExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<Scalar>> {lhs, scalar};
}

template<typename Scalar, typename RHS>
requires ValidNumericType<std::remove_cvref_t<Scalar>> && IsValidMathOperand<RHS>
auto operator*(Scalar scalar, const RHS& rhs)
{
    return ScaleExpr<std::remove_cvref_t<RHS>, std::remove_cvref_t<Scalar>> {rhs, scalar};
}

template<typename LHS, typename RHS>
requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto operator*(const LHS& lhs, const RHS& rhs)
{
    return MulExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<typename LHS, typename RHS>
requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto operator+(const LHS& lhs, const RHS& rhs)
{
    return AddExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<typename LHS, typename RHS>
requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto operator-(const LHS& lhs, const RHS& rhs)
{
    return SubtrExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<typename LHS, typename RHS>
requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto dot(const LHS& lhs, const RHS& rhs)
{
    return DotExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>{lhs, rhs};
}

}
template<typename Expr>
auto with_policy(const Expr& expr, const Math_Functions_Policy* policy) {
    auto e = expr;
    e.policy = policy;
    return e;
}


#endif // EXPRESSION_TEMPLATES
