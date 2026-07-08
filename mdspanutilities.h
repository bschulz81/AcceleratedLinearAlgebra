#ifndef MDSPANUTILITIES
#define MDSPANUTILITIES
#include "datablock.h"
#include "datablockutilities.h"
#include "indiceshelperfunctions.h"
#include "mdspan_omp.h"
#include "mdspan_data.h"

class mdspan_utilities
{
public:

template<typename T,typename Container>
inline static mdspan<T, Container> tensor_subspan(const  mdspan<T, Container>  &d,const Container  &offsets,  Container &sub_extents);

template<typename T,typename Container>
inline static mdspan<T, Container> matrix_subspan(const  mdspan<T, Container>  &d,const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols );

template<typename T,typename Container>
inline static mdspan<T, Container> matrix_row(const  mdspan<T, Container>  &d,const size_t row_index);

template<typename T,typename Container>
inline static mdspan<T, Container> matrix_column(const  mdspan<T, Container>  &d,const size_t column_index);

template<typename T,typename Container>
inline static mdspan<T, Container> matrix_transpose(const  mdspan<T, Container>  &d);
template<typename T,typename Container>
inline static mdspan<T, Container> matrix_hermitian_transpose(const  mdspan<T, Container>  &d);

template<typename T,typename Container>
inline static mdspan_data<T, Container> tensor_subspan_copy(const  mdspan<T, Container>  &d, const Container& offsets, const Container &sub_extents, const bool memmap=false);
template<typename T,typename Container>
inline static mdspan_data<T, Container> matrix_subspan_copy(const  mdspan<T, Container>  &d, const size_t row, const size_t col, const size_t tile_rows, const size_t tile_cols, const bool memmap=false);
template<typename T,typename Container>
inline static mdspan_data<T, Container> matrix_transpose_copy(const  mdspan<T, Container>  &d, bool memmap=false);
template<typename T,typename Container>
inline static mdspan_data<T, Container> matrix_hermitian_transpose_copy(const  mdspan<T, Container>  &d, bool memmap=false);
template<typename T,typename Container>
inline static mdspan_data<T, Container> matrix_column_copy(const  mdspan<T, Container>  &d, const size_t col_index, const bool memmap=false);

template<typename T,typename Container>
inline static mdspan_data<T, Container> matrix_row_copy(const  mdspan<T, Container>  &d, const size_t row_index, const bool memmap=false);
};



template<typename T,typename Container>
mdspan<T, Container> mdspan_utilities::tensor_subspan( const mdspan<T, Container>&d,const Container &offsets, Container &sub_extents)
{
    size_t *tempstr=new size_t[offsets.size()];
    size_t *tempext=new size_t[offsets.size()];
    mdspan<T, Container> result(DataBlockUtilities::tensor_subspan(d, offsets.data(),sub_extents.data(),tempext, tempstr),d.mapping_manager);
    delete [] tempstr;
    delete [] tempext;
    return result;
}

template<typename T,typename Container>
mdspan<T, Container>  mdspan_utilities::matrix_subspan(const mdspan<T, Container>  &d,const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols )
{

    size_t tempext[2], tempstr[2];
    mdspan<T, Container> result(DataBlockUtilities::matrix_subspan(d,row,col,tile_rows,tile_cols, tempext, tempstr),d.mapping_manager);
    return result;
}

template<typename T,typename Container>
mdspan<T, Container>  mdspan_utilities:: matrix_row(const mdspan<T, Container>  &d,const size_t row_index)
{
    size_t tempext[1], tempstr[1];
    mdspan<T, Container> result(DataBlockUtilities::matrix_row(d,row_index,tempext, tempstr),d.mapping_manager);
    return result;
}

template<typename T,typename Container>
mdspan<T, Container>  mdspan_utilities::matrix_column(const mdspan<T, Container>  &d,const size_t column_index)
{
    size_t tempext[1], tempstr[1];
    mdspan<T, Container> result(DataBlockUtilities::matrix_column(d,column_index,tempext, tempstr),d.mapping_manager);
    return result;
}


template<typename T,typename Container>
mdspan<T, Container>  mdspan_utilities::matrix_transpose(const mdspan<T, Container>  &d)
{
    size_t tempext[2], tempstr[2];
    mdspan<T, Container> result(DataBlockUtilities::matrix_transpose(d,tempext,tempstr),d.mapping_manager);
    return result;
}

template<typename T,typename Container>
mdspan<T, Container>  mdspan_utilities::matrix_hermitian_transpose(const mdspan<T, Container>  &d)
{

    size_t tempext[2], tempstr[2];
    mdspan<T, Container> result(DataBlockUtilities::matrix_hermitian_transpose(d,tempext,tempstr),d.mapping_manager);
    return result;
}




template<typename T,typename Container>
mdspan_data<T, Container>  mdspan_utilities::tensor_subspan_copy(const mdspan<T, Container>  &d,const Container& offsets, const Container& sub_extents, const bool memmap)
{
    mdspan_data<T, Container> result(sub_extents, d.dprowmajor, memmap, d.dpdata_is_devptr, false, d.devptr_devicenum, d.pconjugate);
    DataBlock<T> temp = DataBlockUtilities::tensor_subspan_copy<OpenMPVariant::ParallelSimd>(d,offsets.data(), sub_extents.data(), result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = temp.dprank;
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container>  mdspan_utilities::matrix_subspan_copy(const  mdspan<T, Container> &d,const size_t row, const size_t col, const size_t tile_rows, const size_t tile_cols, const bool memmap)
{

    mdspan_data<T, Container> result(tile_rows, tile_cols, d.dprowmajor, memmap, d.dpdata_is_devptr, false, d.devptr_devicenum, d.pconjugate);
    DataBlockUtilities::matrix_subspan_copy<OpenMPVariant::ParallelSimd>(d,row, col, tile_rows, tile_cols, result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = 2;
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_transpose_copy(const  mdspan<T, Container> &d,bool memmap)
{
    mdspan_data<T, Container>  result(d.dpextents[1], d.dpextents[0], d.dprowmajor, memmap, d.dpdata_is_devptr, false, d.devptr_devicenum, d.pconjugate);
    DataBlockUtilities::matrix_transpose_copy<OpenMPVariant::ParallelSimd>(d,result.pextents.data(), result.pstrides.data(), result.dpdata);
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_hermitian_transpose_copy(const mdspan<T, Container> &d,bool memmap)
{
    mdspan_data<T, Container> result(d.dpextents[1],d.dpextents[0], d.dprowmajor, memmap,d.dpdata_is_devptr, false,d.devptr_devicenum, !d.pconjugate);
    DataBlockUtilities::matrix_hermitian_transpose_copy<OpenMPVariant::ParallelSimd>(d,result.pextents.data(), result.pstrides.data(), result.dpdata);
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_column_copy(const mdspan<T, Container> &d,const size_t col_index, const bool memmap)
{
    mdspan_data<T, Container> result(d.dpextents[0], 1, d.dprowmajor, memmap,d.dpdata_is_devptr, false, d.devptr_devicenum,d.pconjugate);
    DataBlockUtilities::matrix_column_copy<OpenMPVariant::ParallelSimd>(d,col_index, result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = 1;
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_row_copy(const mdspan<T, Container> &d,const size_t row_index, const bool memmap)
{

    mdspan_data<T, Container> result(d.dpextents[1], 1, d.dprowmajor, memmap, d.dpdata_is_devptr, false, d.devptr_devicenum,d.pconjugate);
    DataBlockUtilities::matrix_row_copy<OpenMPVariant::ParallelSimd>(d,row_index, result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = 1;
    return result;
}




#endif
