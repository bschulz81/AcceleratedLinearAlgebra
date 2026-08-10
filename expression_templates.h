#ifndef EXPRESSION_TEMPLATES
#define EXPRESSION_TEMPLATES


#include <type_traits>
#include <vector>

#include "datablock.h"
#include "mdspan_omp.h"
#include "mathfunctions.h"
#include "mathfunctionspolicy.h"

template<typename T>
class DataBlock;

template<typename T, typename Container>
class mdspan_data;


struct ManagedDataBlockConfig;

namespace expr
{



// Forward declarations of expressions
template<typename LHS, typename RHS> struct AddExpr;
template<typename LHS, typename RHS> struct SubtrExpr;
template<typename LHS, typename RHS> struct MulExpr;
template<typename LHS, typename Scalar> struct ScaleExpr;
template<typename LHS, typename RHS> struct DotExpr;

enum class ResultSource
{
    Node,   // allocate result storage here
    LHS,    // result is stored in lhs result
    RHS     // result is stored in rhs result
};

enum class StorageKind
{
    None,
    Destination,
    materialized_buffer
};


// Result type
struct EvaluationInfo
{
    int temporaries = 0;

    bool inplace_left = false;
    bool inplace_right = false;

    DataBlockObject shape;

    size_t rank = 0;

    bool is_scalar = false;

    const char* operation = "";

    ResultSource result_source = ResultSource::Node;
    StorageKind result_storage = StorageKind::None;
};

inline void print_indent(int level)
{
    for(int i=0; i<level; i++)
        std::cout << "  ";
}


template<typename Expr>
EvaluationInfo analyze(const Expr&);


// DataBlock
template<typename T>
EvaluationInfo analyze(const DataBlock<T>&);


template<typename T, typename Container>
EvaluationInfo analyze(const mdspan<T,Container>&);

template<typename LHS, typename RHS>
EvaluationInfo analyze(const AddExpr<LHS,RHS>&);

template<typename LHS, typename RHS>
EvaluationInfo analyze(const SubtrExpr<LHS,RHS>&);

template<typename LHS, typename RHS>
EvaluationInfo analyze(const MulExpr<LHS,RHS>&);

template<typename LHS, typename Scalar>
EvaluationInfo analyze(const ScaleExpr<LHS,Scalar>&);

template<typename LHS, typename RHS>
EvaluationInfo analyze(const DotExpr<LHS,RHS>&);


template<typename T>
EvaluationInfo analyze(const DataBlock<T>& d)
{
    EvaluationInfo info;

    info.temporaries = 0;
    info.shape = d.ObjectType();
    info.rank = d.rank();
    info.operation = "DataBlock";
    info.result_source = ResultSource::Node;
    info.result_storage = StorageKind::None;
    return info;
}


template<typename T, typename Container>
EvaluationInfo analyze(const mdspan<T,Container>& d)
{
    EvaluationInfo info;

    info.temporaries = 0;
    info.shape = d.ObjectType();
    info.rank = d.rank();
    info.operation = "mdspan";
    info.result_source = ResultSource::Node;
    info.result_storage = StorageKind::None;
    return info;
}

template<typename LHS, typename RHS>
EvaluationInfo analyze(const AddExpr<LHS, RHS>& e)
{
    auto l = analyze(e.lhs);
    auto r = analyze(e.rhs);

    EvaluationInfo result;

    result.shape = e.ObjectType();
    result.rank = e.rank();
    result.operation = "Add";

    result.inplace_left  = true;
    result.inplace_right = true;

    if (l.result_storage != StorageKind::None)
    {
        result.result_source  = ResultSource::LHS;
        result.result_storage = l.result_storage;

        result.temporaries =
            std::max(l.temporaries, r.temporaries);
    }
    else if (r.result_storage != StorageKind::None)
    {
        result.result_source  = ResultSource::RHS;
        result.result_storage = r.result_storage;

        result.temporaries =
            std::max(l.temporaries, r.temporaries);
    }
    else
    {
        result.result_source  = ResultSource::Node;
        result.result_storage = StorageKind::materialized_buffer;

        result.temporaries =
            std::max(l.temporaries, r.temporaries) + 1;
    }

    return result;
}

template<typename LHS, typename RHS>
EvaluationInfo analyze(const SubtrExpr<LHS, RHS>& e)
{
    auto l = analyze(e.lhs);
    auto r = analyze(e.rhs);

    EvaluationInfo result;

    result.shape = e.ObjectType();
    result.rank = e.rank();
    result.operation = "Subtract";

    result.inplace_left  = true;
    result.inplace_right = false;

    if (l.result_storage != StorageKind::None)
    {
        result.result_source  = ResultSource::LHS;
        result.result_storage = l.result_storage;

        result.temporaries =
            std::max(l.temporaries, r.temporaries);
    }
    else
    {
        result.result_source  = ResultSource::Node;
        result.result_storage = StorageKind::materialized_buffer;

        result.temporaries =
            std::max(l.temporaries, r.temporaries) + 1;
    }

    return result;
}

template<typename LHS, typename Scalar>
EvaluationInfo analyze(const ScaleExpr<LHS, Scalar>& e)
{
    auto result = analyze(e.lhs);

    result.operation = "Scale";
    result.inplace_left = true;

    if (result.result_storage != StorageKind::None)
    {
        result.result_source = ResultSource::LHS;
    }
    else
    {
        result.result_source  = ResultSource::Node;
        result.result_storage = StorageKind::materialized_buffer;
        result.temporaries += 1;
    }

    return result;
}

template<typename LHS, typename RHS>
EvaluationInfo analyze(const MulExpr<LHS, RHS>& e)
{
    auto l = analyze(e.lhs);
    auto r = analyze(e.rhs);

    EvaluationInfo result;

    result.shape = e.ObjectType();
    result.rank = e.rank();
    result.operation = "Mul";

    result.inplace_left  = false;
    result.inplace_right = false;

    result.result_source  = ResultSource::Node;
    result.result_storage = StorageKind::materialized_buffer;

    result.temporaries =
        l.temporaries +
        r.temporaries;

    return result;
}


class ExpressionExecutionPolicy
{
public:
    Math_Functions_Policy kernel_policy;
    ManagedDataBlockConfig temporary_placement= {};
    bool check_locations = true;
    bool check_sizes = true;
    bool follow_expression_location = true;
    bool debugoutput=false;
};

inline std::optional<ExpressionExecutionPolicy> default_policy;

inline const ExpressionExecutionPolicy& get_default_policy()
{
    if (!default_policy)
    {
        default_policy.emplace();

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


inline const char* result_source_name(ResultSource source)
{
    switch (source)
    {
    case ResultSource::Node:
        return "Node";
    case ResultSource::LHS:
        return "LHS";
    case ResultSource::RHS:
        return "RHS";
    }

    return "?";
}

inline const char* storage_kind_name(StorageKind storage)
{
    switch (storage)
    {
    case StorageKind::None:
        return "None";
    case StorageKind::Destination:
        return "Destination";
    case StorageKind::materialized_buffer:
        return "materialized_buffer";
    }

    return "?";
}


template<typename Expr>
void print_expression(
    const Expr& e,
    int level);


template<typename T, typename Container>
void print_expression(
    const mdspan<T,Container>& e,
    int level)
{
    auto info = analyze(e);

    print_indent(level);

    std::cout
            << "mdspan "
            << "source=" << result_source_name(info.result_source)
            << " storage=" << storage_kind_name(info.result_storage)
            << "\n";
}

template<typename LHS, typename RHS>
void print_expression(
    const AddExpr<LHS,RHS>& e,
    int level)
{
    auto info = analyze(e);

    print_indent(level);

    std::cout
            << "Add "
            << "peak_tmp=" << info.temporaries
            << " inplace=("
            << info.inplace_left << ","
            << info.inplace_right << ")"
            << " source=" << result_source_name(info.result_source)
            << " storage=" << storage_kind_name(info.result_storage)
            << "\n";

    print_expression(e.lhs, level + 1);
    print_expression(e.rhs, level + 1);
}

template<typename LHS, typename Scalar>
void print_expression(
    const ScaleExpr<LHS,Scalar>& e,
    int level)
{
    auto info = analyze(e);

    print_indent(level);

    std::cout
            << "Scale "
            << "peak_tmp=" << info.temporaries
            << " inplace=("
            << info.inplace_left << ","
            << info.inplace_right << ")"
            << " source=" << result_source_name(info.result_source)
            << " storage=" << storage_kind_name(info.result_storage)
            << "\n";

    print_expression(e.lhs, level + 1);
}


template<typename LHS, typename RHS>
void print_expression(
    const SubtrExpr<LHS,RHS>& e,
    int level)
{
    auto info = analyze(e);

    print_indent(level);

    std::cout
            << "Sub "
            << "peak_tmp=" << info.temporaries
            << " inplace=("
            << info.inplace_left << ","
            << info.inplace_right << ")"
            << " source=" << result_source_name(info.result_source)
            << " storage=" << storage_kind_name(info.result_storage)
            << "\n";

    print_expression(e.lhs, level + 1);
    print_expression(e.rhs, level + 1);
}

template<typename LHS, typename RHS>
void print_expression(
    const MulExpr<LHS,RHS>& e,
    int level)
{
    auto info = analyze(e);

    print_indent(level);

    std::cout
            << "Mul "
            << "peak_tmp=" << info.temporaries
            << " source=" << result_source_name(info.result_source)
            << " storage=" << storage_kind_name(info.result_storage)
            << "\n";

    print_expression(e.lhs, level + 1);
    print_expression(e.rhs, level + 1);
}


template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;
template<typename T> concept ValidNumericType = std::is_arithmetic_v<T> || is_complex_v<T>;

template<typename T> struct is_datablock_type
{
private:
    template <typename U> static std::true_type test(const DataBlock<U>*);
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test(std::declval<const T*>()))::value;
};


template<typename T>
inline constexpr bool is_datablock_type_v =
    is_datablock_type<std::remove_cvref_t<T>>::value;



template<typename Derived>
class ExpressionInterface
{
public:
    template<typename Expr>
    requires
    requires(
        const Expr& e,
        Derived& d,
        const ExpressionExecutionPolicy* p)
    {
        e.assign_to(d,p);
    }
    Derived& operator=(const Expr& expr)
    {
        return assign(expr,nullptr);
    }

    template<typename Expr>
    Derived& assign(
        const Expr& expr,
        const expr::ExpressionExecutionPolicy* policy=nullptr)
    {
        const auto& pol = policy ? *policy : expr::get_default_policy();


        ManagedDataBlockConfig placement =pol.temporary_placement;

        LocationCheckContext ctx;
        if (pol.check_locations)
        {
            if (!expr.location_check(ctx))
                throw std::runtime_error("Expression location mismatch");

           placement.data_ondevice = ctx.data_is_device;

            if (ctx.data_is_device)
            placement.devicenum = ctx.device_number;
        }


        auto info = expr::analyze(expr);
        if (pol.debugoutput)
        {

            std::cout << "[expression] analysis:\n";
            expr::print_expression(expr,0);

            std::cout
                    << "[expression]Peak temporaries = "
                    << info.temporaries
                    << "\n";
        }

        Derived& out = static_cast<Derived&>(*this);

        if (!has_same_layout(out, expr))
        {
            out.recreate(expr,placement);
        }


        expr.assign_to(static_cast<Derived&>(*this), &pol);

        return static_cast<Derived&>(*this);
    }
};

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

    inline ptrdiff_t rank() const
    {
        return lhs.rank();
    }

    inline const ptrdiff_t* extents_ptr() const
    {
        return lhs.extents_ptr();
    }

    inline const ptrdiff_t* strides_ptr() const
    {
        return lhs.strides_ptr();
    }

    inline bool rowmajor() const
    {
        return lhs.rowmajor();
    }


    inline ptrdiff_t datalength() const
    {
        return lhs.datalength();
    }

    inline auto ObjectType() const
    {
        return lhs.ObjectType();
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

    inline ptrdiff_t rank() const
    {
        return lhs.rank();
    }

    inline const ptrdiff_t* extents_ptr() const
    {
        return lhs.extents_ptr();
    }

    inline const ptrdiff_t* strides_ptr() const
    {
        return lhs.strides_ptr();
    }

    inline bool rowmajor() const
    {
        return lhs.rowmajor();
    }


    inline ptrdiff_t datalength() const
    {
        return lhs.datalength();
    }

    inline auto ObjectType() const
    {
        return lhs.ObjectType();
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

    inline ptrdiff_t rank() const
    {
        return lhs.rank();
    }

    inline const ptrdiff_t* extents_ptr() const
    {
        return lhs.extents_ptr();
    }

    inline const ptrdiff_t* strides_ptr() const
    {
        return lhs.strides_ptr();
    }

    inline bool rowmajor() const
    {
        return lhs.rowmajor();
    }


    inline ptrdiff_t datalength() const
    {
        return lhs.datalength();
    }

    inline auto ObjectType() const
    {
        return lhs.ObjectType();
    }

    bool location_check(LocationCheckContext& state) const
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

protected:
    ptrdiff_t pedatalength = 1;
    std::vector<ptrdiff_t> peextents;
    std::vector<ptrdiff_t> pestrides;
    bool perowmajor=true;

public:
    const LHS& lhs;
    const RHS& rhs;
    MulExpr(const LHS& l, const RHS& r)
        : lhs(l), rhs(r)
    {
        initialize_layout();
    }

    void initialize_layout()
    {
        auto l = lhs.ObjectType();
        auto r = rhs.ObjectType();

        if (l == DataBlockObject::Scalar)
        {
            copy_layout(rhs);
            return;
        }

        if (r == DataBlockObject::Scalar)
        {
            copy_layout(lhs);
            return;
        }

        if (l == DataBlockObject::Matrix &&r == DataBlockObject::Matrix)
        {
            peextents.resize(2);
            pestrides.resize(2);

            peextents[0] = lhs.extents_ptr()[0];
            peextents[1] = rhs.extents_ptr()[1];

            if (lhs.rowmajor())
            {
                pestrides[1] = 1;
                pestrides[0] = peextents[1];
                perowmajor=true;
            }
            else
            {
                pestrides[0] = 1;
                pestrides[1] = peextents[0];
                perowmajor=false;
            }
            ptrdiff_t n = 1;
            #pragma omp unroll partial
            for (ptrdiff_t i =0; i<peextents.size(); i++)
                n *= peextents[i];
            pedatalength=n;
            return;

        }

        if (l == DataBlockObject::Matrix &&r == DataBlockObject::Vector)
        {
            peextents.resize(1);
            pestrides.resize(1);

            peextents[0] = lhs.extents_ptr()[0];
            pestrides[0] = 1;
            perowmajor=true;
            pedatalength=peextents[0];
            return;
        }

        // vector * matrix
        if (l == DataBlockObject::Vector &&r == DataBlockObject::Matrix)
        {
            peextents.resize(1);
            pestrides.resize(1);

            peextents[0] = rhs.extents_ptr()[1];
            pestrides[0] = 1;
            perowmajor=true;

            pedatalength=peextents[0];
            return;
        }

        if (l == DataBlockObject::Vector &&r == DataBlockObject::Vector)
        {
            peextents.resize(1);
            pestrides.resize(1);
            peextents[0] = 1;
            pestrides[0] = 1;
            pedatalength=1;
            perowmajor=true;
            return;
        }

        throw std::runtime_error(
            "Unsupported multiplication expression");
    }

    void copy_layout(const auto& x)
    {
        ptrdiff_t r = x.rank();

        peextents.resize(r);
        pestrides.resize(r);

        const ptrdiff_t* ext = x.extents_ptr();
        const ptrdiff_t* str = x.strides_ptr();

        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < r; ++i)
        {
            peextents[i] = ext[i];
            pestrides[i] = str[i];
        }
        perowmajor = x.rowmajor();
        pedatalength = x.datalength();

    }
    inline ptrdiff_t rank() const
    {
        return peextents.size();
    }

    inline ptrdiff_t datalength() const
    {
        return pedatalength;
    }

    inline bool rowmajor() const
    {
        return perowmajor;
    }

    const ptrdiff_t* extents_ptr() const
    {
        return peextents.data();
    }

    const ptrdiff_t* strides_ptr() const
    {
        return pestrides.data();
    }

    inline auto ObjectType() const
    {

        auto l = lhs.ObjectType();
        auto r = rhs.ObjectType();

        if (l == DataBlockObject::Scalar)
            return r;

        if (r == DataBlockObject::Scalar)
            return l;

        if (l == DataBlockObject::Matrix &&r == DataBlockObject::Matrix)
            return DataBlockObject::Matrix;

        if (l == DataBlockObject::Matrix &&r == DataBlockObject::Vector)
            return DataBlockObject::Vector;

        if (l == DataBlockObject::Vector &&r == DataBlockObject::Matrix)
            return DataBlockObject::Vector;

        if (l == DataBlockObject::Vector &&r == DataBlockObject::Vector)
            return DataBlockObject::Scalar;


        return DataBlockObject::Tensor;
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
