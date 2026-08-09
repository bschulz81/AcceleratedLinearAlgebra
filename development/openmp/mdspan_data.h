#ifndef MDSPAN_DATAH
#define MDSPAN_DATAH

#include "string.h"
#include <atomic>
#include "datablock.h"
#include "mdspan_omp.h"

template <typename T>
class DataBlock;

template<typename T, typename Container>
class mdspan;









template <typename T,typename Container>
class mdspan_data : public mdspan<T,Container>,
    public expr::ExpressionInterface<mdspan_data<T,Container>>
{
public:

    friend class mdspan_utilities;

    mdspan_data() {};

    using expr::ExpressionInterface<mdspan_data<T,Container>>::operator=;

    mdspan_data(ptrdiff_t datalength, const Container& extents, const Container& strides,ManagedDataBlockConfig config);

    mdspan_data(const Container& extents, const Container& strides, ManagedDataBlockConfig config);

    mdspan_data(const Container& extents,ManagedDataBlockConfig config);

    mdspan_data( const DataBlock<T>& view, ManagedDataBlockConfig* alloc_config=nullptr) ;

    mdspan_data(const mdspan_data<T, Container>& other);
    mdspan_data<T, Container>&operator=(const mdspan_data<T,Container> & other);

    mdspan_data(mdspan_data<T, Container>&& other) noexcept;
    mdspan_data<T,Container>& operator=( mdspan_data<T, Container>&& other) noexcept;
    void allocate(const Container& extents,
                  const Container& strides,
                  const ManagedDataBlockConfig& config);

    template<typename Expr>
    void recreate( const Expr& expr, const ManagedDataBlockConfig& config);



    ~mdspan_data();

    mdspan_data<T,Container> copy( ManagedDataBlockConfig *config);

    void release_all_data();
    void allocate_storage(const ManagedDataBlockConfig& config);
protected:
    std::atomic<int>* p_ref_count = nullptr;

};


template<typename T, typename Tag>
using mdspan_data_t =mdspan_data<T,typename container_selector<Tag>::template container<ptrdiff_t>>;

#include "mdspan_data.hpp"
#endif

