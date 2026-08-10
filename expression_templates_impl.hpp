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
    const expr::ExpressionExecutionPolicy* pl = nullptr)
{
    const expr::ExpressionExecutionPolicy& policy =
        (pl != nullptr) ? *pl : get_default_policy();


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


    mdspan_data<T,Container> temp;

    temp.recreate(expr, placement);

    expr.assign_to(temp, &policy);

    return temp;
}



template<typename T, typename Expr>
void evaluate_into(
    const Expr& expr,
    DataBlock<T>& C,
    const expr::ExpressionExecutionPolicy& policy)
{
    using E = std::remove_cvref_t<Expr>;

    if constexpr (is_datablock_type_v<E>)
    {
        if (policy.check_sizes &&
                !same_extents(expr, C))
        {
            throw std::runtime_error("Wrong extents");
        }

        C.copy_from(expr);
    }
    else
    {
        expr.assign_to(C, &policy);
    }
}





inline bool same_extents(const auto& a, const auto& b)
{
    if (a.rank() != b.rank())
        return false;

    for(size_t i=0; i<(size_t)a.rank(); ++i)
        if(a.extent(i)!=b.extent(i))
            return false;

    return true;
}



template<typename T, typename Expr>
decltype(auto)
evaluate_materialized(
    const Expr& expr,
    const expr::ExpressionExecutionPolicy& policy)
{
    using E = std::remove_cvref_t<Expr>;

    if constexpr (std::is_base_of_v<DataBlock<T>, E>)
    {
        if (policy.debugoutput)
        {
            std::cout<< "[evaluate_materialized] DataBlock -> borrow existing object\n";
        }

        return (expr);
    }
    else
    {
        if (policy.debugoutput)
        {
            std::cout<< "[evaluate_materialized] Expression -> allocate temporary\n";
        }

        mdspan_data_t<T, dynamic_tag> result;

        ManagedDataBlockConfig placement =
            policy.temporary_placement;

        if (policy.follow_expression_location)
        {
            LocationCheckContext ctx;

            if (!expr.location_check(ctx))
                throw std::runtime_error("Expression location mismatch");

            placement.data_ondevice =ctx.data_is_device;

            if (ctx.data_is_device)
            {
                placement.devicenum =ctx.device_number;
            }

            if (policy.debugoutput)
            {
                std::cout<< "  expression location: " << (ctx.data_is_device? "device": "host");

                if (ctx.data_is_device)
                {
                    std::cout<<" "<< ctx.device_number;
                }

                std::cout << "\n";
            }
        }


        result.recreate(expr, placement);
        if (policy.debugoutput)
        {
            std::cout<< "  evaluating expression into temporary \n";
        }
        expr.assign_to(result, &policy);

        return result;
    }
}


template<typename LHS, typename RHS>
template<typename T, typename Container>
AddExpr<LHS, RHS>::operator mdspan_data<T, Container>() const
{
    return expr::evaluate_to_mdspan_data<T, Container>(*this);
}


template<typename T, typename Expr>
mdspan_data_t<T, dynamic_tag>make_accumulator(const Expr& source,const expr::ExpressionExecutionPolicy& policy)
{
    mdspan_data_t<T, dynamic_tag> result;

    ManagedDataBlockConfig placement =
        policy.temporary_placement;

    if (policy.follow_expression_location)
    {
        LocationCheckContext ctx;

        if (!source.location_check(ctx))
            throw std::runtime_error("Expression location mismatch");

        placement.data_ondevice =
            ctx.data_is_device;

        if (ctx.data_is_device)
            placement.devicenum =ctx.device_number;

        if (policy.debugoutput)
        {
            std::cout
                    << "[make_accumulator] location: "<< (ctx.data_is_device? "device": "host");

            if (ctx.data_is_device)
            {
                std::cout<< " "<< ctx.device_number;
            }

            std::cout << "\n";
        }
    }

    result.recreate(source, placement);
    using E = std::remove_cvref_t<Expr>;
    if constexpr (is_datablock_type_v<E>)
    {
        if (policy.debugoutput)
        {
            std::cout<< "[make_accumulator] DataBlock -> copy into owning accumulator \n";
        }


        DataBlockUtilities::copy(static_cast<DataBlock<T>&>(result),static_cast<const DataBlock<T>&>(source));

    }
    else
    {
        if (policy.debugoutput)
        {
            std::cout<< "[make_accumulator] Expression -> evaluate directly into owning accumulator \n";
        }

    }

    return result;
}



template<typename LHS, typename RHS>
template<typename T>
void AddExpr<LHS, RHS>::assign_to(
    DataBlock<T>& C,
    const expr::ExpressionExecutionPolicy* pl) const
{
    const auto& policy =(pl != nullptr)? *pl: get_default_policy();

    Math_Functions_Policy mathpol =
        policy.kernel_policy;

    const auto info = analyze(*this);

    /*
     * The result can live in the LHS.
     *
     * Therefore we create one owning accumulator from
     * the LHS and accumulate the RHS into it.
     */
    if (info.result_source == ResultSource::LHS)
    {
        auto L = make_accumulator<T>(lhs, policy);

        if constexpr (is_datablock_type_v<RHS>)
        {
            if (policy.check_sizes &&!same_extents(L, rhs))
            {
                throw std::runtime_error(
                    "Wrong extents");
            }

            switch (this->ObjectType())
            {
            case DataBlockObject::Matrix:
                Math_Functions::matrix_add(rhs, L, &mathpol);
                break;
            case DataBlockObject::Vector:
                Math_Functions::vector_add(rhs, L, &mathpol);
                break;

            default:
                throw std::runtime_error("Unsupported type for addition");
            }
        }
        else
        {
            /*
             * RHS is an expression.
             *
             * For now the kernel requires a materialized
             * RHS, so this may allocate another temporary.
             */
            auto R = evaluate_materialized<T>(rhs, policy);

            if (policy.check_sizes &&!same_extents(L, R))
            {
                throw std::runtime_error(
                    "Wrong extents");
            }

            switch (this->ObjectType())
            {
            case DataBlockObject::Matrix:
                Math_Functions::matrix_add(R, L, &mathpol);
                break;

            case DataBlockObject::Vector:
                Math_Functions::vector_add(R, L, &mathpol);
                break;

            default:
                throw std::runtime_error("Unsupported type for addition");
            }
        }

        DataBlockUtilities::copy(
            C,
            static_cast<const DataBlock<T>&>(L));

        return;
    }

    /*
     * The result belongs to this node.
     *
     * C has already been allocated/configured by the
     * assignment machinery, so evaluate both operands
     * and write the result directly into C.
     */

    auto L = evaluate_materialized<T>(lhs,policy);

    auto R = evaluate_materialized<T>(rhs,policy);

    if (policy.check_sizes &&!same_extents(L, R))
    {
        throw std::runtime_error("Wrong extents");
    }

    switch (this->ObjectType())
    {
    case DataBlockObject::Matrix:
        Math_Functions::matrix_add(L, R, C, &mathpol);
        break;

    case DataBlockObject::Vector:
        Math_Functions::vector_add(L, R, C, &mathpol);
        break;

    default:
        throw std::runtime_error(
            "Unsupported type for addition");
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
void SubtrExpr<LHS, RHS>::assign_to(
    DataBlock<T>& C,
    const expr::ExpressionExecutionPolicy* pl) const
{
    const auto& policy =(pl != nullptr)? *pl: get_default_policy();

    Math_Functions_Policy mathpol =policy.kernel_policy;

    const auto info = analyze(*this);

    /*
     * The result can live in the LHS.
     *
     * Create one owning accumulator from the LHS
     * and subtract the RHS from it.
     */
    if (info.result_source == ResultSource::LHS)
    {
        auto L = make_accumulator<T>(lhs, policy);

        if constexpr (is_datablock_type_v<RHS>)
        {
            if (policy.check_sizes &&
                    !same_extents(L, rhs))
            {
                throw std::runtime_error(
                    "Wrong extents");
            }

            switch (this->ObjectType())
            {
            case DataBlockObject::Matrix:
                Math_Functions::matrix_subtract(rhs, L, &mathpol);
                break;

            case DataBlockObject::Vector:
                Math_Functions::vector_subtract(rhs, L, &mathpol);
                break;

            default:
                throw std::runtime_error(
                    "Unsupported type for subtraction");
            }
        }
        else
        {
            auto R = evaluate_materialized<T>(rhs,policy);

            if (policy.check_sizes &&
                    !same_extents(L, R))
            {
                throw std::runtime_error(
                    "Wrong extents");
            }

            switch (this->ObjectType())
            {
            case DataBlockObject::Matrix:
                Math_Functions::matrix_subtract(R, L, &mathpol);
                break;

            case DataBlockObject::Vector:
                Math_Functions::vector_subtract(R, L, &mathpol);
                break;

            default:
                throw std::runtime_error("Unsupported type for subtraction");
            }
        }

        DataBlockUtilities::copy(C,static_cast<const DataBlock<T>&>(L));

        return;
    }

    /*
     * The result belongs to this node.
     *
     * C has already been allocated/configured.
     * Evaluate both operands and write L-R directly
     * into C.
     */
    auto L = evaluate_materialized<T>(lhs,policy);

    auto R = evaluate_materialized<T>(rhs,policy);

    if (policy.check_sizes &&!same_extents(L, R))
    {
        throw std::runtime_error("Wrong extents");
    }

    switch (this->ObjectType())
    {
    case DataBlockObject::Matrix:
        Math_Functions::matrix_subtract(L, R, C, &mathpol);
        break;

    case DataBlockObject::Vector:
        Math_Functions::vector_subtract(L, R, C, &mathpol);
        break;

    default:
        throw std::runtime_error("Unsupported type for subtraction");
    }
}


template<typename LHS, typename Scalar>
template<typename T, typename Container>
ScaleExpr<LHS, Scalar>::operator mdspan_data<T, Container>() const
{
    return expr::evaluate_to_mdspan_data<T, Container>(*this);
}



template<typename LHS, typename Scalar>
template<typename T>
void ScaleExpr<LHS, Scalar>::assign_to(
    DataBlock<T>& C,
    const expr::ExpressionExecutionPolicy* pl) const
{
    const auto& policy =(pl != nullptr)? *pl: get_default_policy();

    Math_Functions_Policy mathpol =
        policy.kernel_policy;

    const auto info = analyze(*this);

    /*
     * The result can live in the LHS.
     *
     * Create one owning accumulator and scale it
     * in place.
     */
    if (info.result_source == ResultSource::LHS)
    {
        auto L = make_accumulator<T>(lhs,policy);

        switch (this->ObjectType())
        {
        case DataBlockObject::Vector:
            Math_Functions::vector_multiply_scalar(L,scalar,&mathpol);
            break;

        case DataBlockObject::Matrix:
            Math_Functions::matrix_multiply_scalar(L,scalar,&mathpol);
            break;

        default:
            throw std::runtime_error(
                "Unsupported type for scalar multiplication");
        }

        /*
         * C has already been allocated/configured by
         * the assignment machinery.
         */
        DataBlockUtilities::copy(C,static_cast<const DataBlock<T>&>(L));

        return;
    }

    /*
     * The result belongs to this node.
     *
     * em the LHS and write the scaled
     * result directly into C.
     */
    auto L = evaluate_materialized<T>(lhs,policy);

    switch (this->ObjectType())
    {
    case DataBlockObject::Vector:
        Math_Functions::vector_multiply_scalar(L,scalar,C,&mathpol);
        break;

    case DataBlockObject::Matrix:
        Math_Functions::matrix_multiply_scalar(L,scalar,C,&mathpol);
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
void MulExpr<LHS, RHS>::assign_to(
    DataBlock<T>& C,
    const expr::ExpressionExecutionPolicy* pl) const
{
    const auto& policy =
        (pl != nullptr)
            ? *pl
            : get_default_policy();

    Math_Functions_Policy mathpol =
        policy.kernel_policy;

    /*
     * Neither operand can be used as the result
     * storage for matrix multiplication.
     *
     * Therefore materialize both operands and
     * write the result directly into C.
     */
    auto L = evaluate_materialized<T>(lhs,policy);

    auto R = evaluate_materialized<T>(rhs,policy);

    if (policy.check_sizes)
    {
        if (L.rank() != 2 || R.rank() != 2)
        {
            throw std::runtime_error("Wrong rank for matrix multiplication");
        }

        if (L.extent(1) != R.extent(0))
        {
            throw std::runtime_error("Wrong Matrix extents");
        }
    }

    if (L.ObjectType() == DataBlockObject::Matrix)
    {
        if (R.ObjectType() == DataBlockObject::Matrix)
        {
            Math_Functions::matrix_multiply_dot(L,R,C,&mathpol);
        }
        else if (R.ObjectType() == DataBlockObject::Vector)
        {
            Math_Functions::matrix_multiply_vector(L,R,C,&mathpol);
        }
        else
        {
            throw std::runtime_error("Unsupported RHS for matrix multiplication");
        }
    }
    else if (L.ObjectType() == DataBlockObject::Vector &&
             R.ObjectType() == DataBlockObject::Vector)
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
T DotExpr<LHS, RHS>::eval_scalar(
    const expr::ExpressionExecutionPolicy* pl) const
{
    const auto& policy =
        (pl != nullptr)
            ? *pl
            : get_default_policy();

    Math_Functions_Policy mathpol =
        policy.kernel_policy;


    auto L = evaluate_materialized<T>(lhs,policy);

    auto R = evaluate_materialized<T>(rhs,policy);

    if (L.ObjectType() != DataBlockObject::Vector ||
            R.ObjectType() != DataBlockObject::Vector)
    {
        throw std::runtime_error("DotExpr only works for vectors");
    }

    if (policy.check_sizes &&
        L.extent(0) != R.extent(0))
    {
        throw std::runtime_error("Wrong vector sizes");
    }

    return Math_Functions::dot_product(L,R,&mathpol);
}


}

#endif // EXPRESSION_TEMPLATES_IMPL_HPP
