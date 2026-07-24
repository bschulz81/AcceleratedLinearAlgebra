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

    template<typename T,typename Container>
    inline static mdspan<T, Container> create_matrix(T* data,  const size_t rows, const size_t cols, DataBlockConfig  config);

    template<typename T,typename Container>
    inline static mdspan<T, Container>create_vector(T* data,  const size_t rows, DataBlockConfig  config);


    template <typename T, typename Container>
    inline static mdspan<T,std::vector<size_t>> collapsed_view(mdspan<T, Container>&d);

    template<typename T, typename Container>
    inline static mdspan_data<T, Container> copy(const mdspan<T, Container>& base,bool memmap, bool ondevice, bool defaultdevice, int devicenum);

    template <typename T, typename Container>
    inline static mdspan_data<T,Container>create_matrix(const size_t rows, const size_t cols, ManagedDataBlockConfig config);

    template <typename T, typename Container>
    inline static mdspan_data<T,Container>create_vector(const size_t rows,ManagedDataBlockConfig config);




};



template <typename T, typename Container>
mdspan_data<T,Container>mdspan_utilities::create_matrix(const size_t rows, const size_t cols, ManagedDataBlockConfig config)
{
    mdspan<T,Container> matrix_metadata= mdspan_utilities::create_matrix<T, Container>((T*)nullptr, rows,cols,config.Get_DataBlockConfig());
    return mdspan_data<T,Container>(matrix_metadata.pextents, matrix_metadata.pstrides, config);
}

template <typename T, typename Container>
mdspan_data<T,Container>mdspan_utilities::create_vector(const size_t rows,ManagedDataBlockConfig config)
{
    mdspan<T,Container> vector_metadata= mdspan_utilities::create_vector<T, Container>((T*)nullptr, rows, config.Get_DataBlockConfig());
    return mdspan_data<T,Container>(vector_metadata.pextents, vector_metadata.pstrides, config);
}


template<typename T,typename Container>
mdspan<T, Container> mdspan_utilities::create_matrix(T* data,  const size_t rows, const size_t cols,DataBlockConfig  config)
{

    const size_t r=2;
    Container pextents,pstrides;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
        pstrides = {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r);
        pstrides.resize(r);
    }
    pextents[0]=rows;
    pextents[1]=cols;
    pstrides[0] = config.dprowmajor ? cols : 1;
    pstrides[1] = config.dprowmajor ? 1 : rows;
    size_t  dpdatalength = (rows - 1) * pstrides[0] + (cols - 1) * pstrides[1] + 1;

    return mdspan(data, dpdatalength,pextents, pstrides, config);
}



template<typename T,typename Container>
mdspan<T, Container> mdspan_utilities::create_vector(T* data,  const size_t rows,DataBlockConfig  config)
{
    config.dprowmajor=true;
    Container pextents,pstrides;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(1);
        pstrides.resize(1);
    }

    pextents[0]=rows;
    pstrides[0]=1;

    size_t dpdatalength= (pextents[0] - 1) * pstrides[0] + 1;;

    return mdspan(data, dpdatalength,pextents, pstrides,  config);


}




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
    ManagedDataBlockConfig cfg=ManagedDataBlockConfig::SetConfig(memmap,false,d.dpconfig);

    mdspan_data<T, Container> result(sub_extents,cfg);
    DataBlock<T> temp = DataBlockUtilities::tensor_subspan_copy<OpenMPVariant::ParallelSimd>(d,offsets.data(), sub_extents.data(), result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = temp.dprank;
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container>  mdspan_utilities::matrix_subspan_copy(const  mdspan<T, Container> &d,const size_t row, const size_t col, const size_t tile_rows, const size_t tile_cols, const bool memmap)
{
    ManagedDataBlockConfig cfg=ManagedDataBlockConfig::SetConfig(memmap,false,d.dpconfig);

    mdspan_data<T,Container> result= mdspan_utilities::create_matrix<T, Container>( tile_rows,tile_cols,cfg);
    DataBlockUtilities::matrix_subspan_copy<OpenMPVariant::ParallelSimd>(d,row, col, tile_rows, tile_cols, result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = 2;
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_transpose_copy(const  mdspan<T, Container> &d,bool memmap)
{
    ManagedDataBlockConfig cfg=ManagedDataBlockConfig::SetConfig(memmap,false,d.dpconfig);
    mdspan_data<T,Container> result= mdspan_utilities::create_matrix<T, Container>( d.dpextents[1],d.dpextents[0],cfg);
    DataBlockUtilities::matrix_transpose_copy<OpenMPVariant::ParallelSimd>(d,result.pextents.data(), result.pstrides.data(), result.dpdata);
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_hermitian_transpose_copy(const mdspan<T, Container> &d,bool memmap)
{
    ManagedDataBlockConfig cfg=ManagedDataBlockConfig::SetConfig(memmap,false,d.dpconfig);
    mdspan_data<T,Container> result= mdspan_utilities::create_matrix<T, Container>( d.dpextents[1],d.dpextents[0],cfg);
    DataBlockUtilities::matrix_hermitian_transpose_copy<OpenMPVariant::ParallelSimd>(d,result.pextents.data(), result.pstrides.data(), result.dpdata);
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_column_copy(const mdspan<T, Container> &d,const size_t col_index, const bool memmap)
{
    ManagedDataBlockConfig cfg=ManagedDataBlockConfig::SetConfig(memmap,false,d.dpconfig);
    mdspan_data<T,Container> result= mdspan_utilities::create_vector<T, Container>( d.dpextents[0],cfg);
    DataBlockUtilities::matrix_column_copy<OpenMPVariant::ParallelSimd>(d,col_index, result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = 1;
    return result;
}

template<typename T,typename Container>
mdspan_data<T, Container> mdspan_utilities::matrix_row_copy(const mdspan<T, Container> &d,const size_t row_index, const bool memmap)
{
    ManagedDataBlockConfig cfg=ManagedDataBlockConfig::SetConfig(memmap,false,d.dpconfig);
    mdspan_data<T,Container> result= mdspan_utilities::create_vector<T, Container>( d.dpextents[1],cfg);
    DataBlockUtilities::matrix_row_copy<OpenMPVariant::ParallelSimd>(d,row_index, result.pextents.data(), result.pstrides.data(), result.dpdata);
    result.dprank = 1;
    return result;
}



template <typename T, typename Container>
mdspan<T,std::vector<size_t>> mdspan_utilities::collapsed_view(mdspan<T, Container>&d)
{
    size_t num_dims = d.count_noncollapsed_dims();
    size_t *tempext=new size_t[num_dims],
    *tempstr=new size_t[num_dims];
    mdspan<T, std::vector<size_t>> result(d.collapsed_view(num_dims,tempext, tempstr),d.mapping_manager);
    delete []tempext;
    delete []tempstr;
    return result;

}

template<typename T, typename Container>
mdspan_data<T, Container> mdspan_utilities::copy(const mdspan<T, Container>& base,bool memmap, bool ondevice, bool defaultdevice, int devicenum)
{
    if(defaultdevice)
        devicenum = omp_get_default_device();

    ManagedDataBlockConfig cfg(base.dpconfig, memmap, ondevice, defaultdevice, devicenum);


    mdspan_data<T, Container> result(base.pextents, base.pstrides, cfg);
    int targetdev, sourcedev;
    bool useomptargetmemcpy = false;

    if(ondevice && base.dpconfig.data_ondevice)
    {
        targetdev = devicenum;
        sourcedev = base.devicenum;
        useomptargetmemcpy = true;
    }
    else if(ondevice && !base.dpconfig.data_ondevice)
    {
        targetdev = devicenum;
        sourcedev = omp_get_initial_device();
        useomptargetmemcpy = true;
    }
    else if(!ondevice &&base.dpconfig.data_ondevice)
    {
        targetdev = omp_get_initial_device();
        sourcedev = base.devicenum;
        useomptargetmemcpy = true;
    }

    if(useomptargetmemcpy)
        omp_target_memcpy(result.dpdata, base.dpdata, sizeof(T) * base.dpdatalength, 0, 0, targetdev, sourcedev);
    else
        memcpy(result.dpdata, base.dpdata, sizeof(T) * base.dpdatalength);

    return result;
}

#endif
