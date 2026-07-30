#ifndef MDSPANUTILITIES
#define MDSPANUTILITIES

#include<vector>
#include "datablock.h"
template <typename T>
class DataBlock;

template<typename T, typename Container>class mdspan;
template<typename T, typename Container>
class mdspan;

class mdspan_utilities
{
public:

    template<typename T,typename Container>
    inline static mdspan<T,Container> tensor_subspan(const  mdspan<T,Container>  &d,const Container  &offsets,  Container &sub_extents);
    template<typename T,typename Container>
    inline static mdspan<T,Container> matrix_subspan(const  mdspan<T,Container>  &d,const ptrdiff_t row, const ptrdiff_t col,const  ptrdiff_t tile_rows,const  ptrdiff_t tile_cols );
    template<typename T,typename Container>
    inline static mdspan<T,Container> matrix_row(const  mdspan<T,Container>  &d,const ptrdiff_t row_index);
    template<typename T,typename Container>
    inline static mdspan<T,Container> matrix_column(const  mdspan<T,Container>  &d,const ptrdiff_t column_index);
    template<typename T,typename Container>
    inline static mdspan<T,Container> matrix_transpose(const  mdspan<T,Container>  &d);
    template<typename T,typename Container>
    inline static mdspan<T,Container> matrix_hermitian_transpose(const  mdspan<T,Container>  &d);


    template<typename T,typename Container>
    inline static mdspan_data<T,Container> tensor_subspan_copy(const  mdspan<T,Container>  &d, const Container& offsets, const Container &sub_extents, DataBlockConfig cfg);
    template<typename T,typename Container>
    inline static mdspan_data<T,Container> matrix_subspan_copy(const  mdspan<T,Container>  &d, const ptrdiff_t row, const ptrdiff_t col, const ptrdiff_t tile_rows, const ptrdiff_t tile_cols, DataBlockConfig cfg);
    template<typename T,typename Container>
    inline static mdspan_data<T,Container> matrix_transpose_copy(const  mdspan<T,Container>  &d,DataBlockConfig cfg);
    template<typename T,typename Container>
    inline static mdspan_data<T,Container> matrix_hermitian_transpose_copy(const  mdspan<T,Container>  &d,DataBlockConfig cfg);
    template<typename T, typename Container>
    inline static mdspan_data<T,Container> matrix_column_copy(const  mdspan<T, Container>  &d, const ptrdiff_t col_index, DataBlockConfig cfg);

    template<typename T,typename Container>
    inline static mdspan_data<T,Container> matrix_row_copy(const  mdspan<T,Container>  &d, const ptrdiff_t row_index, DataBlockConfig cfg);



    template <typename T, typename Container>
    inline static mdspan<T,std::vector<ptrdiff_t>> collapsed_view(mdspan<T,Container>&d);

    template<typename T, typename Container>
    inline static mdspan_data<T,Container>  copy(const mdspan<T,Container>& base,ManagedDataBlockConfig cfg);




    template<typename T,typename Tag>
    inline static auto create_matrix(T* data,  const ptrdiff_t rows, const ptrdiff_t cols, DataBlockConfig  config);

    template<typename T, typename Tag>
    inline static auto create_vector(T* data,  const ptrdiff_t rows, DataBlockConfig  config);


    template <typename T, typename Tag>
    inline static auto create_matrix(const ptrdiff_t rows, const ptrdiff_t cols, ManagedDataBlockConfig config);

    template <typename T, typename Tag>
    inline static auto create_vector(const ptrdiff_t rows, ManagedDataBlockConfig config);


};

#include "mathutilitiesmdspan.hpp"

#endif
