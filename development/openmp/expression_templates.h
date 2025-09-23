#ifndef EXPRESSION_TEMPLATES
#define EXPRESSION_TEMPLATES

#include <optional>
#include <type_traits>
#include "mathfunctions.h"
#include "mdspan_omp.h"
#include "datablock.h"

namespace expr
{


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

template<typename U>
concept HasObjectType = requires(const std::remove_cvref_t<U>& x)
{
    { x.ObjectType() };
};

template<typename U>
concept HasAssignTo = requires(const std::remove_cvref_t<U>& x, DataBlock<double>& C)
{
    { x.assign_to(C) };
};

template<typename U>
concept ExprOrDataBlock = HasObjectType<U> || HasAssignTo<U>;

template<typename Expr>
concept Expression = requires(const Expr& e, DataBlock<double>& C)
{
    { e.assign_to(C) };
} || requires(const Expr& e)
{
    { e.template eval_scalar<double>() };
};


template<typename LHS, typename RHS>
struct AddExpr
{
    const LHS& lhs;
    const RHS& rhs;
    const Math_Functions_Policy* policy = nullptr;

    template<typename T>
    void assign_to(DataBlock<T>& C, const Math_Functions_Policy* override = nullptr) const
    {
        auto pol = override ? override : policy;  // pick override if given

        if (lhs.ObjectType() == DataBlock<T>::Matrix)
            Math_Functions<T>::matrix_add(lhs, rhs, C, pol);
        else if (lhs.ObjectType() == DataBlock<T>::Vector)
            Math_Functions<T>::vector_add(lhs, rhs, C, pol);
        else
            throw std::runtime_error("Unsupported type for addition");
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
        auto pol = override ? override : policy;  // pick override if given

        if (lhs.ObjectType() == DataBlock<T>::Matrix)
            Math_Functions<T>::matrix_subtract(lhs, rhs, C, pol);
        else if (lhs.ObjectType() == DataBlock<T>::Vector)
            Math_Functions<T>::vector_subtract(lhs, rhs, C, pol);
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
        auto pol = override ? override : policy;  // pick override if given

        if constexpr (requires { lhs.ObjectType(); })
        {
            switch(lhs.ObjectType())
            {
                case DataBlock<T>::Vector:
                    Math_Functions<T>::vector_multiply_scalar(lhs, scalar, C, pol);
                    break;
                case DataBlock<T>::Matrix:
                    Math_Functions<T>::matrix_multiply_scalar(lhs, scalar, C, pol);
                    break;
                default:
                    throw std::runtime_error("Unsupported type for scalar multiplication");
            }
        }
        else
        {
            static_assert(std::is_same_v<LHS, void>, "ScaleExpr::assign_to: lhs must be DataBlock-like");
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
                Math_Functions<T>::matrix_multiply_dot(lhs, rhs, C, pol);
            else if (rhs.ObjectType() == DataBlock<T>::Vector)
                Math_Functions<T>::matrix_multiply_vector(lhs, rhs, C, pol);
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
            return Math_Functions<T>::dot_product(lhs, rhs, pol);
        }
        throw std::runtime_error("DotExpr only works for vectors");
    }

    template<typename T>
    operator T() const {
        return eval_scalar<T>();
    }
};

template<ExprOrDataBlock LHS, typename Scalar>
requires std::is_arithmetic_v<std::remove_cvref_t<Scalar>>
auto operator*(const LHS& lhs, Scalar scalar)
{
    return ScaleExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<Scalar>> {lhs, scalar};
}

template<typename Scalar, ExprOrDataBlock RHS>
requires std::is_arithmetic_v<std::remove_cvref_t<Scalar>>
auto operator*(Scalar scalar, const RHS& rhs)
{
    return ScaleExpr<std::remove_cvref_t<RHS>, std::remove_cvref_t<Scalar>> {rhs, scalar};
}

template<ExprOrDataBlock LHS, ExprOrDataBlock RHS>
auto operator*(const LHS& lhs, const RHS& rhs)
{
    return MulExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<ExprOrDataBlock LHS, ExprOrDataBlock RHS>
auto operator+(const LHS& lhs, const RHS& rhs)
{
    return AddExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<ExprOrDataBlock LHS, ExprOrDataBlock RHS>
auto operator-(const LHS& lhs, const RHS& rhs)
{
    return SubtrExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}


template<ExprOrDataBlock LHS, ExprOrDataBlock RHS>
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


#endif
