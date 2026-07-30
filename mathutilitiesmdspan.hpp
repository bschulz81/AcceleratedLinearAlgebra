#ifndef MDSPANUTILITIEShpp
#define MDSPANUTILITIEShpp

#include "datablock.h"
#include "datablockconfigstructs.h"


template <typename T, typename Tag>
auto mdspan_utilities::create_matrix(const ptrdiff_t rows, const ptrdiff_t cols, ManagedDataBlockConfig config)
{
    using Container =typename container_selector<Tag>::template container<ptrdiff_t>;
    mdspan_t<T,Tag> matrix_metadata= mdspan_utilities::create_matrix<T,Tag>((T*)nullptr, rows,cols,config.Get_DataBlockConfig());
    return mdspan_data<T,Container>(matrix_metadata.pextents, matrix_metadata.pstrides, config);
}



template <typename T, typename Tag>
auto mdspan_utilities::create_vector(const ptrdiff_t rows,ManagedDataBlockConfig config)
{
    using Container =typename container_selector<Tag>::template container<ptrdiff_t>;
    mdspan_t<T,Tag> vector_metadata= mdspan_utilities::create_vector<T,Tag>((T*)nullptr, rows, config.Get_DataBlockConfig());
    return mdspan_data<T,Container>(vector_metadata.pextents, vector_metadata.pstrides, config);
}




template<typename T,typename Tag>
auto mdspan_utilities::create_matrix(T* data,  const ptrdiff_t rows, const ptrdiff_t cols,DataBlockConfig  config)
{

    const ptrdiff_t r=2;
    using Container =typename container_selector<Tag>::template container<ptrdiff_t>;
    Container pextents;
    Container pstrides;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
    }
    if constexpr (StaticContainer<Container>)
    {
        pstrides = {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r);
    }
    if constexpr (DynamicContainer<Container>)
        pstrides.resize(r);

    pextents[0]=abs(rows);
    pextents[1]=abs(cols);
    pstrides[0] = config.dprowmajor ? abs(cols) : 1;
    pstrides[1] = config.dprowmajor ? 1 : abs(rows);
    ptrdiff_t  dpdatalength = (abs(rows) - 1) * pstrides[0] + (abs(cols) - 1) * pstrides[1] + 1;

    return mdspan(data, dpdatalength,pextents, pstrides, config);
}



template<typename T,typename Tag>
auto  mdspan_utilities::create_vector(T* data,  const ptrdiff_t rows,DataBlockConfig  config)
{

    config.dprowmajor=true;
    using Container =typename container_selector<Tag>::template container<ptrdiff_t>;
    Container pextents;
    Container pstrides;

    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
    }
    if constexpr (StaticContainer<Container>)
    {
        pstrides= {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(1);
    }
    if constexpr (DynamicContainer<Container>)
    {
        pstrides.resize(1);
    }

    pextents[0]=abs(rows);
    pstrides[0]=1;

    ptrdiff_t dpdatalength= (abs(pextents[0]) - 1) * pstrides[0] + 1;;

    return mdspan(data, dpdatalength,pextents, pstrides,  config);


}





template<typename T,typename Container>
mdspan<T,Container> mdspan_utilities::tensor_subspan( const mdspan<T,Container>&d,const Container &offsets, Container &sub_extents)
{
    ptrdiff_t *tempstr=new ptrdiff_t[offsets.size()];
    ptrdiff_t *tempext=new ptrdiff_t[offsets.size()];
    mdspan<T,Container> result(DataBlockUtilities::tensor_subspan(d, offsets.data(),sub_extents.data(),tempext, tempstr),d.offload_registry);
    delete [] tempstr;
    delete [] tempext;
    return result;
}

template<typename T,typename Container>
mdspan<T,Container>  mdspan_utilities::matrix_subspan(const mdspan<T,Container>  &d,const ptrdiff_t row, const ptrdiff_t col,const  ptrdiff_t tile_rows,const  ptrdiff_t tile_cols )
{

    ptrdiff_t tempext[2];
    ptrdiff_t tempstr[2];
    mdspan<T,Container> result(DataBlockUtilities::matrix_subspan(d,row,col,tile_rows,tile_cols, tempext, tempstr),d.offload_registry);
    return result;
}

template<typename T,typename Container>
mdspan<T,Container>  mdspan_utilities:: matrix_row(const mdspan<T,Container>  &d,const ptrdiff_t row_index)
{
    ptrdiff_t tempext[1];
    ptrdiff_t tempstr[1];
    mdspan<T,Container> result(DataBlockUtilities::matrix_row(d,row_index,tempext, tempstr),d.offload_registry);
    return result;
}

template<typename T,typename Container>
mdspan<T,Container>  mdspan_utilities::matrix_column(const mdspan<T,Container>  &d,const ptrdiff_t column_index)
{
    ptrdiff_t tempext[1];
    ptrdiff_t tempstr[1];
    mdspan<T,Container> result(DataBlockUtilities::matrix_column(d,column_index,tempext, tempstr),d.offload_registry);
    return result;
}


template<typename T,typename Container>
mdspan<T,Container>  mdspan_utilities::matrix_transpose(const mdspan<T,Container>  &d)
{
    ptrdiff_t tempext[2];
    ptrdiff_t tempstr[2];
    mdspan<T,Container> result(DataBlockUtilities::matrix_transpose(d,tempext,tempstr),d.offload_registry);

    return result;
}

template<typename T,typename Container>
mdspan<T,Container>  mdspan_utilities::matrix_hermitian_transpose(const mdspan<T,Container>  &d)
{

    ptrdiff_t tempext[2];
    ptrdiff_t tempstr[2];
    mdspan<T,Container> result(DataBlockUtilities::matrix_hermitian_transpose(d,tempext,tempstr),d.offload_registry);
    return result;
}





template <typename T, typename Container>
mdspan<T,std::vector<ptrdiff_t>> mdspan_utilities::collapsed_view(mdspan<T,Container>&d)
{
    ptrdiff_t num_dims = d.count_noncollapsed_dims();
    ptrdiff_t *tempext=new ptrdiff_t[num_dims];
    ptrdiff_t *tempstr=new ptrdiff_t[num_dims];
    mdspan<T, std::vector<ptrdiff_t>> result(d.collapsed_view(num_dims,tempext, tempstr),d.offload_registry);
    delete []tempext;
    delete []tempstr;
    return result;

}

template<typename T, typename Container>
mdspan_data<T,Container> mdspan_utilities::copy(const mdspan<T,Container>& base,ManagedDataBlockConfig cfg)
{

    mdspan_data<T,Container> result(base.pextents, base.pstrides, cfg);
    int targetdev, sourcedev;
    bool useomptargetmemcpy = false;

    if(cfg.data_ondevice && base.dpconfig.data_is_devptr)
    {
        targetdev = cfg.devicenum;
        sourcedev = base.dpconfig.devicenum;
        useomptargetmemcpy = true;
    }
    else if(cfg.data_ondevice && !base.dpconfig.data_is_devptr)
    {
        targetdev = cfg.devicenum;
        sourcedev = omp_get_initial_device();
        useomptargetmemcpy = true;
    }
    else if(!cfg.data_ondevice &&base.dpconfig.data_is_devptr)
    {
        targetdev = omp_get_initial_device();
        sourcedev = base.dpconfig.devicenum;
        useomptargetmemcpy = true;
    }

    if(useomptargetmemcpy)
        omp_target_memcpy(result.dpdata, base.dpdata, sizeof(T) * base.dpdatalength, 0, 0, targetdev, sourcedev);
    else
        memcpy(result.dpdata, base.dpdata, sizeof(T) * base.dpdatalength);

    return result;
}

#endif
