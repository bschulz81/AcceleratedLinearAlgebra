#ifndef EXPRESSION_TEMPLATES
#define EXPRESSION_TEMPLATES


#include <type_traits>
#include <vector>

#include "datablock.h"

#include "mathfunctions.h"
#include "mathfunctionspolicy.h"

template<typename T>
class DataBlock;

template<typename T, typename Container>
class mdspan_data;


struct ManagedDataBlockConfig;

namespace expr
{



class ExpressionExecutionPolicy
{
public:
    Math_Functions_Policy kernel_policy;
    ManagedDataBlockConfig temporary_placement= {};
    bool check_locations = true;
    bool check_sizes = true;
    bool follow_expression_location = true;
};



// global default policy
inline std::optional<ExpressionExecutionPolicy> default_policy;

inline const ExpressionExecutionPolicy& get_default_policy()
{
    if (!default_policy)
    {
        default_policy.emplace();

        // initialize kernel policy from Math_Functions
        default_policy->kernel_policy =
            Math_Functions::get_default_policy();
    }

    return *default_policy;
}

inline void set_default_policy(const ExpressionExecutionPolicy& p)
{
    default_policy = p;
}

inline void reset_default_policy()
{
    default_policy.reset();
}

template<typename T,
         typename Container = std::vector<ptrdiff_t>,
         typename Expression>
auto evaluate(const Expression& expr,
              const ExpressionExecutionPolicy& policy)
{
    return evaluate_to_mdspan_data<T, Container>(expr, &policy);
}


struct LocationCheckContext
{
    bool check_started = false;

    bool data_is_device = false;
    int device_number = -INT_MAX;

    template<typename T>
    bool check(const DataBlock<T>& d)
    {
#if defined(Unified_Shared_Memory)
        return d.dpdata != nullptr;
#endif

        if (d.data() == nullptr)
            return false;

        bool this_is_device =
            d.data_is_devptr();

        if (!check_started)
        {
            check_started = true;

            data_is_device = this_is_device;

            if (this_is_device)
                device_number = d.devptr_num();

            return true;
        }

        if (data_is_device != this_is_device)
            return false;

        if (data_is_device &&
                device_number != d.devptr_num())
            return false;

        return true;


    }
};


template<typename Derived>
class ExpressionInterface
{
public:

    template<typename Expr>
    Derived& operator=(const Expr& expr)
    {
        return static_cast<Derived*>(this)->assign(expr,nullptr);
    }


    template<typename Expr>
    Derived& assign(
        const Expr& expr,
        const expr::ExpressionExecutionPolicy* policy=nullptr)
    {
        const auto& pol = policy ? *policy : expr::get_default_policy();

        if (pol.check_locations)
        {
            expr::LocationCheckContext ctx;

            if (!expr.location_check(ctx))
                throw std::runtime_error("Expression location mismatch");
        }


        expr.assign_to(static_cast<Derived&>(*this), &pol);

        return static_cast<Derived&>(*this);
    }
};




template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;
template<typename T> concept ValidNumericType = std::is_arithmetic_v<T> || is_complex_v<T>;




template<typename LHS, typename RHS> struct AddExpr;
template<typename LHS, typename RHS> struct SubtrExpr;
template<typename LHS, typename RHS> struct MulExpr;
template<typename LHS, typename Scalar> struct ScaleExpr;
template<typename LHS, typename RHS> struct DotExpr;


template<typename T> struct is_datablock_type
{
private:
    template <typename U> static std::true_type test(const DataBlock<U>*);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<const T*>()))::value;
};
template<typename T> inline constexpr bool is_datablock_type_v = is_datablock_type<std::remove_cvref_t<T>>::value;

template<typename T> struct is_expr_type : std::false_type {};
template<typename L, typename R> struct is_expr_type<AddExpr<L, R>> : std::true_type {};
template<typename L, typename R> struct is_expr_type<SubtrExpr<L, R>> : std::true_type {};
template<typename L, typename R> struct is_expr_type<MulExpr<L, R>> : std::true_type {};
template<typename L, typename S> struct is_expr_type<ScaleExpr<L, S>> : std::true_type {};
template<typename L, typename R> struct is_expr_type<DotExpr<L, R>> : std::true_type {};

template<typename T> concept IsValidMathOperand = is_datablock_type_v<T> || is_expr_type<std::remove_cvref_t<T>>::value;


template<typename LHS, typename RHS>
struct AddExpr
{
    const LHS& lhs;
    const RHS& rhs;

    inline auto DataShape() const
    {
        return lhs.DataShape();
    }
    inline size_t rank() const
    {
        return lhs.rank();
    }
    inline size_t extent(size_t index) const
    {
        return lhs.extent(index);
    }
    inline size_t datalength() const
    {
        return lhs.datalength();
    }

    bool location_check(
        LocationCheckContext& state) const
    {
        return lhs.location_check(state) &&
               rhs.location_check(state);
    }

    template<typename T, typename Container = std::vector<ptrdiff_t>>
    operator mdspan_data<T, Container>() const;

    template<typename T>
    void assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pol = nullptr) const;
};

template<typename LHS, typename RHS>
struct SubtrExpr
{
    const LHS& lhs;
    const RHS& rhs;

    inline auto DataShape() const
    {
        return lhs.DataShape();
    }
    inline size_t rank() const
    {
        return lhs.rank();
    }
    inline size_t extent(size_t index) const
    {
        return lhs.extent(index);
    }
    inline size_t datalength() const
    {
        return lhs.datalength();
    }
    bool location_check(
        LocationCheckContext& state) const
    {
        return lhs.location_check(state) &&
               rhs.location_check(state);
    }

    template<typename T, typename Container = std::vector<size_t>>
    operator mdspan_data<T, Container>() const;

    template<typename T>
    void assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pol = nullptr) const;
};

template<typename LHS, typename Scalar>
struct ScaleExpr
{
    const LHS& lhs;
    const Scalar scalar;


    inline size_t rank() const
    {
        return lhs.rank();
    }
    inline size_t extent(size_t index) const
    {
        return lhs.extent(index);
    }
    inline size_t datalength() const
    {
        return lhs.datalength();
    }
    inline auto DataShape() const
    {
        return lhs.DataShape();
    }
    bool location_check(
        LocationCheckContext& state) const
    {
        return lhs.location_check(state);
    }

    template<typename T, typename Container = std::vector<size_t>>
    operator mdspan_data<T, Container>() const;

    template<typename T>
    void assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pol = nullptr) const;
};

template<typename LHS, typename RHS>
struct MulExpr
{
    const LHS& lhs;
    const RHS& rhs;

    inline auto DataShape() const
    {
        auto l = lhs.DataShape();
        auto r = rhs.DataShape();

        if (l == DataBlockObject::Matrix &&
                r == DataBlockObject::Matrix)
            return DataBlockObject::Matrix;

        if (l == DataBlockObject::Matrix &&
                r == DataBlockObject::Vector)
            return DataBlockObject::Vector;

        if (l == DataBlockObject::Vector &&
                r == DataBlockObject::Matrix)
            return DataBlockObject::Vector;
        if (l == DataBlockObject::Vector &&
                r == DataBlockObject::Vector)
            return DataBlockObject::Scalar;

        if (l == DataBlockObject::Scalar &&
                r == DataBlockObject::Scalar)
            return DataBlockObject::Scalar;


        return DataBlockObject::Tensor;
    }

    inline size_t rank() const
    {
        if (lhs.DataShape() == DataBlockObject::Matrix && rhs.DataShape() == DataBlockObject::Matrix)
        {
            return 2;
        }
        else if
        ((lhs.DataShape() == DataBlockObject::Matrix && rhs.DataShape() == DataBlockObject::Vector)||
                (lhs.DataShape() == DataBlockObject::Vector && rhs.DataShape() == DataBlockObject::Matrix))
        {
            return 1;
        }
        else
        {
            return lhs.rank();
        }
    }

    inline size_t extent(size_t index) const
    {
        auto lhs_type = lhs.DataShape();
        auto rhs_type = rhs.DataShape();

        // matrix * matrix
        if (lhs_type == DataBlockObject::Matrix &&
                rhs_type == DataBlockObject::Matrix)
        {
            if (index == 0)
                return lhs.extent(0);

            if (index == 1)
                return rhs.extent(1);
        }

        // matrix * vector
        if (lhs_type == DataBlockObject::Matrix &&
                rhs_type == DataBlockObject::Vector)
        {
            return lhs.extent(0);
        }

        // vector * matrix
        if (lhs_type == DataBlockObject::Vector &&
                rhs_type == DataBlockObject::Matrix)
        {
            return rhs.extent(1);
        }

        throw std::runtime_error("Invalid multiplication extent");
    }


    inline size_t datalength() const
    {
        return lhs.extent(0)*lhs.extent(1);
    }
    bool location_check(
        LocationCheckContext& state) const
    {
        return lhs.location_check(state) &&
               rhs.location_check(state);
    }
    template<typename T, typename Container = std::vector<size_t>>
    operator mdspan_data<T, Container>() const;

    template<typename T>
    void assign_to(DataBlock<T>& C, const ExpressionExecutionPolicy* pol = nullptr) const;
};


template<typename LHS, typename RHS>
struct DotExpr
{
    const LHS& lhs;
    const RHS& rhs;

    bool location_check(
        LocationCheckContext& state) const
    {
        return lhs.location_check(state) &&
               rhs.location_check(state);
    }

    template<typename T>
    T eval_scalar(const ExpressionExecutionPolicy* pol = nullptr) const;

    template<typename T>
    operator T() const
    {
        return eval_scalar<T>();
    }
};



template<typename LHS, typename Scalar> requires IsValidMathOperand<LHS> && ValidNumericType<std::remove_cvref_t<Scalar>>
        auto operator*(const LHS& lhs, Scalar scalar)
{
    return ScaleExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<Scalar>> {lhs, scalar};
}

template<typename Scalar, typename RHS> requires ValidNumericType<std::remove_cvref_t<Scalar>> && IsValidMathOperand<RHS>
        auto operator*(Scalar scalar, const RHS& rhs)
{
    return ScaleExpr<std::remove_cvref_t<RHS>, std::remove_cvref_t<Scalar>> {rhs, scalar};
}



template<typename LHS, typename RHS> requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto operator*(const LHS& lhs, const RHS& rhs)
{
    return MulExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<typename LHS, typename RHS> requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto operator+(const LHS& lhs, const RHS& rhs)
{
    return AddExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<typename LHS, typename RHS> requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto operator-(const LHS& lhs, const RHS& rhs)
{
    return SubtrExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

template<typename LHS, typename RHS> requires IsValidMathOperand<LHS> && IsValidMathOperand<RHS>
auto dot(const LHS& lhs, const RHS& rhs)
{
    return DotExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>> {lhs, rhs};
}

}

#include "expression_templates_impl.hpp"

#endif // EXPRESSION_TEMPLATES
