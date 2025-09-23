#ifndef MDSPAN_DATAH
#define MDSPAN_DATAH

#include "mdspan_omp.h"
#include "string.h"

template <typename T, typename Container>
class mdspan_data: public mdspan<T,Container>
{
public:

    mdspan_data() {};

    mdspan_data( size_t datalength, const Container& extents, const Container& strides, bool rowm=true, bool memmap=false, bool ondevice=false,bool default_device=true,int devicenum=0 );
    mdspan_data( size_t datalength, std::initializer_list<size_t> ext,    std::initializer_list<size_t> str, bool rowm=true, bool memmap=false, bool ondevice=false,bool default_device=true,int devicenum=0 ):
        mdspan_data(  datalength,      Container(ext), Container(str),  rowm,  memmap,  ondevice, default_device, devicenum ){}

    mdspan_data(  const Container& extents,const  Container& strides,bool rowm=true, bool memmap=false,bool ondevice=false,bool default_device=true, int devicenum=0   );
    mdspan_data(std::initializer_list<size_t> ext,std::initializer_list<size_t> str,bool rowm=true, bool memmap=false,bool ondevice=false,bool default_device=true, int devicenum=0   ):
        mdspan_data(Container(ext), Container(str), rowm,  memmap, ondevice, default_device,  devicenum   ){}

    mdspan_data(  const Container& extents,bool rowm=true,bool memmap=false,bool ondevice=false,bool default_device=true,int devicenum=0  );
    mdspan_data(std::initializer_list<size_t> ext, bool rowm=true, bool memmap=false, bool ondevice=false,    bool default_device=true, int devicenum=0)
    : mdspan_data(Container(ext), rowm, memmap, ondevice, default_device, devicenum) {}


    mdspan_data(   size_t rows,  size_t cols,bool rowm=true,bool memmap=false,bool ondevice=false,bool default_device=true, int devicenum=0    );
    mdspan_data( size_t rows,bool rowm=true,bool memmap=false,   bool ondevice=false,bool default_device=true, int devicenum=0    );

    mdspan_data(const mdspan<T, Container>& base);
    mdspan_data(mdspan_data<T, Container>&& other) noexcept;
    mdspan_data(const mdspan_data<T, Container>& other);





    ~mdspan_data();

    mdspan<T,Container> &operator=(const mdspan_data<T,Container> & other);
    mdspan_data<T, Container>& operator=( mdspan_data<T, Container>&& other) noexcept;

    using DataBlock<T>::operator=;
//
    using DataBlock<T>::subspan_copy;
    mdspan_data<T, Container> subspan_copy(const Container& offsets, const Container& sub_extents, bool memmap=false ) ;
    mdspan_data<T, Container> subspanmatrix_copy( size_t row,  size_t col,  size_t tile_rows,  size_t tile_cols, bool memmap=false );
    mdspan_data<T, Container> transpose_copy(bool memmap=false );

    mdspan_data<T, Container> column_copy( size_t col_index,bool memmap=false );
    mdspan_data<T, Container> row_copy( size_t row_index,bool memmap=false   );
    mdspan_data<T, Container> copy( bool memmap=false, bool ondevice=false,bool defaultdevice=true,int devicenum=0);

    void release_all_data();
protected:
    bool pmemmap=false;
    void initialization_helper(bool ondevice=false,bool default_device=true, int devicenum=0, const bool memmap=false );
};


template<typename T, typename Tag>
using mdspan_data_t = mdspan_data<T, typename container_for_tag<Tag>::type>;

template <typename T, typename Container>
void mdspan_data<T,Container>::initialization_helper(bool ondevice,bool default_device,int devicenum, const bool memmap)
{
    if(ondevice)
    {
#if defined(Unified_Shared_Memory)
        if (memmap)
            this->dpdata = DataBlock_Host_Memory_Functions<T>::create_temp_mmap(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength];

        pmemmap=memmap;
        this->dpdata_is_devptr=false;
        this->p_has_offloaded_host_data=false;
#else
        if(default_device)
            devicenum=omp_get_default_device();
        this->dpdata=DataBlock_GPU_Memory_Functions<T>::alloc_device_ptr(this->dpdatalength,devicenum);
        this->devptr_devicenum=devicenum;
        this->dpdata_is_devptr=true;
        this->devptr_former_hostptr=nullptr;
        this->p_has_offloaded_host_data=false;
#endif
    }
    else
    {
        if (memmap)
            this->dpdata = DataBlock_Host_Memory_Functions<T>::create_temp_mmap(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength];
        pmemmap=memmap;
        this->dpdata_is_devptr=false;
        this->p_has_offloaded_host_data=false;
    }


}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data( size_t datalength,  const Container& extents, const Container& strides,bool rowm,bool memmap,
                                       bool ondevice, bool default_device,int devicenum    )
    : mdspan<T,Container>(nullptr,   extents,strides,rowm)
{
    initialization_helper(ondevice,default_device,memmap);
}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data( const Container& extents, const Container& strides, bool rowm,  bool memmap,
                                       bool ondevice, bool default_device,int devicenum)
    : mdspan<T,Container>(nullptr,   extents,strides,rowm)
{
    initialization_helper(ondevice,default_device,memmap);

}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(  const Container& extents,bool rowm, bool memmap,
                                       bool ondevice,bool default_device, int devicenum ):
    mdspan<T,Container>(nullptr,   extents,rowm)
{
    initialization_helper(ondevice,default_device,memmap);
}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(  size_t rows,size_t cols,bool rowm, bool memmap,
                                       bool ondevice,bool default_device, int devicenum ):
    mdspan<T,Container>(nullptr,   rows,cols,rowm)
{
    initialization_helper(ondevice,default_device,memmap);
}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(  size_t rows,bool rowm, bool memmap,
                                       bool ondevice,bool default_device, int devicenum ):
    mdspan<T,Container>(nullptr, rows,rowm)
{
    initialization_helper(ondevice,default_device,memmap);
}


template <typename T, typename Container>
void mdspan_data<T, Container>::release_all_data()
{
    if(this->p_has_offloaded_host_data)
    {
        this->device_data_release();
        if (pmemmap)
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(this->dpdata, this->dpdatalength);
        else
            delete[] this->dpdata;
    }
    else
    {
        if(this->dpdata_is_devptr)
            DataBlock_GPU_Memory_Functions<T>::free_device_ptr(this->dpdata,this->devptr_devicenum);
        else
        {
            if (pmemmap)
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(this->dpdata, this->dpdatalength);
            else
                delete[] this->dpdata;
        }
    }
}

template <typename T, typename Container>
mdspan_data<T, Container>::~mdspan_data()
{
    release_all_data();
}
//
template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::subspan_copy(const Container& offsets, const Container& sub_extents,const bool memmap)
{
    mdspan_data<T, Container>  result(  sub_extents,this->dprowmajor,memmap,this->dpdata_is_devptr,false,this->devptr_devicenum);
    DataBlock<T> temp= this->subspan_copy(offsets.data(),sub_extents.data(), result.pextents.data(),result.pstrides.data(), result.dpdata);
    result.dprank=temp.dprank;

    return result;
}


template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::subspanmatrix_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,const bool memmap)
{
    mdspan_data<T, Container>  result( tile_rows,tile_cols,this->dprowmajor,memmap, this->dpdata_is_devptr,false,this->devptr_devicenum);
    this->subspanmatrix_copy_w(row,col,tile_rows,tile_cols, result.pextents.data(),result.pstrides.data(), result.dpdata);
    result.dprank=2;
    return result;
}


template <typename T, typename Container>
mdspan_data<T, Container>  mdspan_data<T, Container>::transpose_copy( bool memmap )
{
    mdspan_data<T, Container>  result(this->dpextents[1],this->dpextents[0],this->dprowmajor,memmap,this->dpdata_is_devptr,false,this->devptr_devicenum);
    this->transpose_copy_w(result.pextents.data(),result.pstrides.data(), result.dpdata);
    return result;
}



template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::column_copy(const size_t col_index, const bool memmap )
{

    mdspan_data<T, Container>  result(this->dpextents[0],1,this->dprowmajor,memmap,this->dpdata_is_devptr,false,this->devptr_devicenum);
    this->column_copy_w(col_index, result.pextents.data(),result.pstrides.data(), result.dpdata);
    result.dprank=1;

    return result;

}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::row_copy(const size_t row_index, const bool memmap  )
{
    mdspan_data<T, Container>  result(this->dpextents[1],1,this->dprowmajor,memmap,this->dpdata_is_devptr,false,this->devptr_devicenum);
    this->row_copy_w(row_index,result.pextents.data(),result.pstrides.data(), result.dpdata);
    result.dprank=1;
    return result;

}

template<typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(const mdspan<T, Container>& base)
    : mdspan<T, Container>(base)
{

    this->mapping_manager = base.mapping_manager;
    this->p_has_offloaded_host_data = false;
    this->pextents=base.pmemmap;
}



template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::copy(bool memmap,bool ondevice,bool defaultdevice,int devicenum )
{
    if(defaultdevice)
        devicenum=omp_get_default_device();
    mdspan_data<T, Container>  result(this->pextents,this->pstrides,this->dprowmajor,memmap,ondevice,false,devicenum  );
    int targetdev,  sourcedev;
    bool useomptargetmemcpy=false;

    if(ondevice && this->dpdata_is_devptr)
    {
        targetdev=devicenum;
        sourcedev=this->devptr_devicenum;
        useomptargetmemcpy=true;
    }
    else
    {
        if(ondevice && !this->dpdata_is_devptr)
        {
            targetdev=devicenum;
            sourcedev=omp_get_initial_device();
            useomptargetmemcpy=true;
        }
        else
        {
            if(!ondevice && this->dpdata_is_devptr)
            {
                targetdev=omp_get_initial_device();
                sourcedev=this->devptr_devicenum;
                useomptargetmemcpy=true;
            }
        }
    }

    if(useomptargetmemcpy)
        omp_target_memcpy(result.dpdata,this->dpdata,sizeof(T)*this->dpdatalength,0,0,targetdev,sourcedev);
    else
        memcpy(result.dpdata,this->dpdata,sizeof(T)*this->dpdatalength);

    return result;
}


template <typename T, typename Container>
mdspan<T,Container>& mdspan_data<T, Container>:: operator=(const mdspan_data<T,Container> & other)
{

    if(this->dpdata!=other.dpdata)
    {
        release_all_data();
        this->p_has_offloaded_host_data = false;
    }

    this->mapping_manager=other.mapping_manager;

    this->pextents = other.pextents;
    this->pstrides = other.pstrides;

    this->dpextents        = this->pextents.data();
    this->dpstrides        = this->pstrides.data();

    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;

    this->devptr_devicenum=other.devptr_devicenum;
    this->devptr_former_hostptr=other.devptr_former_hostptr;
    return *this;
}


template<typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(const mdspan_data<T, Container>& other)
    : mdspan<T, Container>() // call base constructor with empty data
{
    this->dprank = other.dprank;
    this->dprowmajor = other.dprowmajor;
    this->p_has_offloaded_host_data = false;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->devptr_devicenum = other.devptr_devicenum;
    this->devptr_former_hostptr = nullptr; // no device data yet
    this->mapping_manager = other.mapping_manager;


    this->pextents = other.pextents;
    this->pstrides = other.pstrides;


    this->dpextents = this->pextents.data();
    this->dpstrides = this->pstrides.data();
    this->dpdatalength = other.dpdatalength;


    if (other.dpdata_is_devptr)
    {
        this->dpdata = DataBlock_GPU_Memory_Functions<T>::alloc_device_ptr(this->dpdatalength, other.devptr_devicenum);
        omp_target_memcpy(this->dpdata, other.dpdata, sizeof(T) * this->dpdatalength, 0, 0,
                          other.devptr_devicenum, other.devptr_devicenum);
    }
    else
    {
        if (other.pmemmap)
            this->dpdata = DataBlock_Host_Memory_Functions<T>::create_temp_mmap(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength];

        memcpy(this->dpdata, other.dpdata, sizeof(T) * this->dpdatalength);
        pmemmap=other.pmemmap;
    }
}




template<typename T, typename Container>
mdspan_data<T, Container>::mdspan_data(mdspan_data<T, Container>&& other) noexcept
{
    release_all_data();


    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprowmajor = other.dprowmajor;
    this->dprank = other.dprank;

    if constexpr (DynamicContainer<Container>)
    {
        this->pextents = std::move(other.pextents);
        this->pstrides = std::move(other.pstrides);
    }
    else
    {
        this->pextents=other.pextents;
        this->pstrides=other.pstrides;
    }


    this->dpextents = this->pextents.data();
    this->dpstrides = this->pstrides.data();

    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->devptr_devicenum = other.devptr_devicenum;
    this->devptr_former_hostptr = other.devptr_former_hostptr;
    this->p_has_offloaded_host_data = other.p_has_offloaded_host_data;
    this->mapping_manager = std::move(other.mapping_manager);

    pmemmap=other.pmemmap;

    other.dpdata = nullptr;
    other.dpdatalength = 0;
    other.p_has_offloaded_host_data = false;




}

// Move assignment
template<typename T, typename Container>
mdspan_data<T, Container>& mdspan_data<T, Container>::operator=(mdspan_data<T, Container>&& other) noexcept
{
    if(this != &other)
    {
        // Release current memory
        release_all_data();


        this->dpdata = other.dpdata;
        this->dpdatalength = other.dpdatalength;
        this->dprowmajor = other.dprowmajor;
        this->dprank = other.dprank;

        if constexpr (DynamicContainer<Container>)
        {
            this->pextents = std::move(other.pextents);
            this->pstrides = std::move(other.pstrides);
        }
        else
        {
            this->pextents=other.pextents;
            this->pstrides=other.pstrides;
        }


        this->dpextents = this->pextents.data();
        this->dpstrides = this->pstrides.data();

        this->dpdata_is_devptr = other.dpdata_is_devptr;
        this->devptr_devicenum = other.devptr_devicenum;
        this->devptr_former_hostptr = other.devptr_former_hostptr;
        this->p_has_offloaded_host_data = other.p_has_offloaded_host_data;
        this->mapping_manager = std::move(other.mapping_manager);
        pmemmap=other.pmemmap;

        other.dpdata = nullptr;
        other.dpdatalength = 0;
        other.p_has_offloaded_host_data = false;

    }
    return *this;
}
#endif

