#ifndef EXPRESSION_TEMPLATES_IMPL_HPP
#define EXPRESSION_TEMPLATES_IMPL_HPP


#include "mdspan_data.h"
#include "datablock.h"
#include "mathfunctionspolicy.h"
namespace expr
{

template<typename T,
         typename Container,
         typename Expression>
auto evaluate_to_mdspan_data(
    const Expression& expr,
    const ExpressionExecutionPolicy* pl = nullptr)
{
    const ExpressionExecutionPolicy& policy =
        (pl != nullptr) ? *pl : get_default_policy();

    Container ext = {};

    const size_t r = expr.rank();

    if constexpr (DynamicContainer<Container>)
        ext.resize(r);

    for(size_t i = 0; i < r; ++i)
        ext[i] = expr.extent(i);


    ManagedDataBlockConfig placement =
        policy.temporary_placement;


    if (policy.follow_expression_location)
    {
        LocationCheckContext ctx;

        if (!expr.location_check(ctx))
            throw std::runtime_error(
                "Expression location mismatch");

        placement.data_ondevice = ctx.data_is_device;

        if (ctx.data_is_device)
            placement.devicenum = ctx.device_number;
    }


    mdspan_data<T, Container> temp(ext, placement);

    expr.assign_to(temp, &policy);

    return temp;
}



template<typename T, typename Container, typename Expr>
decltype(auto) materialize(const Expr& expr,const ExpressionExecutionPolicy* policy)
{
    using E = std::remove_cvref_t<Expr>;

    if constexpr (std::is_base_of_v<DataBlock<T>, E>)
    {
        return (expr);
    }
    else
    {
        return evaluate_to_mdspan_data<T, Container>(expr,policy);
    }
}


inline bool same_extents(const auto& a, const auto& b)
{
    if (a.rank() != b.rank())
        return false;


    for(size_t i=0;i<(size_t)a.rank();++i)
        if(a.extent(i)!=b.extent(i))
            return false;

    return true;
}


template<typename LHS, typename RHS>
template<typename T, typename Container>
AddExpr<LHS, RHS>::operator mdspan_data<T, Container>() const
{
    return expr::evaluate_to_mdspan_data<T, Container>(*this);
}

template<typename LHS, typename RHS>
template<typename T>
void AddExpr<LHS, RHS>::assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pl) const
{
    const ExpressionExecutionPolicy &policy =(pl!=nullptr)? *pl : get_default_policy();

    auto L = materialize<T,std::vector<ptrdiff_t>>(lhs,&policy);
    auto R = materialize<T,std::vector<ptrdiff_t>>(rhs,&policy);

     if (policy.check_sizes)
            if(!same_extents(L,R))
                throw std::runtime_error("Wrong Matrix extents");


    Math_Functions_Policy mathpol=policy.kernel_policy;
    if (this->DataShape() == DataBlockObject::Matrix)
    {
        Math_Functions::matrix_add(L, R, C, &mathpol);
    }
    else
    {
        Math_Functions::vector_add(L, R, C, &mathpol);
    }
}

template<typename LHS, typename RHS>
template<typename T, typename Container>
SubtrExpr<LHS, RHS>::operator mdspan_data<T, Container>() const
{
    return expr::evaluate_to_mdspan_data<T, Container>(*this);
}

template<typename LHS, typename RHS>
template<typename T>
void SubtrExpr<LHS, RHS>::assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pl) const
{
    const ExpressionExecutionPolicy &policy =(pl!=nullptr)? *pl : get_default_policy();

    auto L = materialize<T,std::vector<ptrdiff_t>>(lhs,&policy);
    auto R = materialize<T,std::vector<ptrdiff_t>>(rhs,&policy);

    if (policy.check_sizes)
            if(!same_extents(L,R))
                throw std::runtime_error("Wrong Matrix extents");

    Math_Functions_Policy mathpol=policy.kernel_policy;


    if (lhs.DataShape() == DataBlockObject::Matrix)
        Math_Functions::matrix_subtract(L, R, C, &mathpol);
    else if (lhs.DataShape() == DataBlockObject::Vector)
        Math_Functions::vector_subtract(L, R, C, &mathpol);
    else throw std::runtime_error("Unsupported type for subtraction");
}

template<typename LHS, typename Scalar>
template<typename T, typename Container>
ScaleExpr<LHS, Scalar>::operator mdspan_data<T, Container>() const
{
    return expr::evaluate_to_mdspan_data<T, Container>(*this);
}

template<typename LHS, typename Scalar>
template<typename T>
void ScaleExpr<LHS, Scalar>::assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pl) const
{
    const ExpressionExecutionPolicy &policy =(pl!=nullptr)? *pl : get_default_policy();
    auto L = materialize<T,std::vector<ptrdiff_t>>(lhs,&policy);
    Math_Functions_Policy mathpol=policy.kernel_policy;
    switch(lhs.DataShape())
    {
    case DataBlockObject::Vector:
        Math_Functions::vector_multiply_scalar(L, scalar, C, &mathpol);
        break;
    case DataBlockObject::Matrix:
        Math_Functions::matrix_multiply_scalar(L, scalar, C, &mathpol);
        break;
    default:
        throw std::runtime_error("Unsupported type for scalar multiplication");
    }
}

template<typename LHS, typename RHS>
template<typename T, typename Container>
MulExpr<LHS, RHS>::operator mdspan_data<T, Container>() const
{
    return expr::evaluate_to_mdspan_data<T, Container>(*this);
}

template<typename LHS, typename RHS>
template<typename T>
void MulExpr<LHS, RHS>::assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pl) const
{
    const ExpressionExecutionPolicy &policy =(pl!=nullptr)? *pl : get_default_policy();
    auto L = materialize<T,std::vector<ptrdiff_t>>(lhs,&policy);
    auto R = materialize<T,std::vector<ptrdiff_t>>(rhs,&policy);

    if (policy.check_sizes)
        if(L.extent(1)!=R.extent(0))
             throw std::runtime_error("Wrong Matrix extents");

    Math_Functions_Policy mathpol=policy.kernel_policy;
    if (lhs.DataShape() == DataBlockObject::Matrix)
    {

        if (rhs.DataShape() == DataBlockObject::Matrix)
        {
            Math_Functions::matrix_multiply_dot(L, R, C, &mathpol);
        }
        else if (rhs.DataShape() == DataBlockObject::Vector)
        {
            Math_Functions::matrix_multiply_vector(L, R, C, &mathpol);
        }
        else throw std::runtime_error("Unsupported RHS for matrix multiplication");
    }
    else if (lhs.DataShape() == DataBlockObject::Vector && rhs.DataShape() == DataBlockObject::Vector)
    {
        throw std::runtime_error("Dot product is scalar, use dot() or eval_scalar()");
    }
    else
    {
        throw std::runtime_error("Unsupported type combination for multiplication");
    }
}

template<typename LHS, typename RHS>
template<typename T>
T DotExpr<LHS, RHS>::eval_scalar(const ExpressionExecutionPolicy* pl) const
{
    const ExpressionExecutionPolicy &policy =(pl!=nullptr)? *pl : get_default_policy();
    auto L = materialize<T,std::vector<ptrdiff_t>>(lhs,&policy);
    auto R = materialize<T,std::vector<ptrdiff_t>>(rhs,&policy);

    Math_Functions_Policy mathpol=policy.kernel_policy;
    if (lhs.DataShape() == DataBlockObject::Vector && rhs.DataShape() == DataBlockObject::Vector)
    {
        if (policy.check_sizes)
            if(L.extent(0)!=R.extent(0))
                throw std::runtime_error("Wrong vector sizes");

        return Math_Functions::dot_product(L, R,&mathpol);
    }
    throw std::runtime_error("DotExpr only works for vectors");
}

} // namespace expr

#endif // EXPRESSION_TEMPLATES_IMPL_HPP
