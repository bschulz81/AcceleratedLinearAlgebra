#ifndef DATABLOCKGPUMEMHELPERS
#define DATABLOCKGPUMEMHELPERS

#include "datablock.h"
#include "datablock_host_memory_functions.h"


template<typename T>
class DataBlock_GPU_Memory_Functions
{
public:
    class OffloadHelper
    {
    protected:
        bool pupdate_host;
        DataBlock<T> &pdL;
        int pdevicenum;
    public:
        inline OffloadHelper(DataBlock<T>& dL, int devicenum, bool just_alloc, bool update_host_on_exit)
            :pupdate_host(update_host_on_exit), pdL(dL),pdevicenum(devicenum)
        {
#if !defined(Unified_Shared_Memory)
            if (just_alloc)
                DataBlock_GPU_Memory_Functions::create_out_struct(dL, devicenum);
            else
                DataBlock_GPU_Memory_Functions::create_in_struct(dL, devicenum);
#endif
        }
        inline OffloadHelper(const DataBlock<T>& dL, int devicenum, bool just_alloc)
            :pupdate_host(false), pdL(dL),pdevicenum(devicenum)
        {
#if !defined(Unified_Shared_Memory)
            if (just_alloc)
                DataBlock_GPU_Memory_Functions::create_out_struct(dL, devicenum);
            else
                DataBlock_GPU_Memory_Functions::create_in_struct(dL, devicenum);
#endif
        }

        inline  ~OffloadHelper()
        {
#if !defined(Unified_Shared_Memory)
            if (pupdate_host && !pdL.dpdata_is_devptr)
            {
                DataBlock_GPU_Memory_Functions::update_host(pdL, pdevicenum);
            }
            DataBlock_GPU_Memory_Functions::release_struct(pdL, pdevicenum);
#endif
        }

        OffloadHelper(const OffloadHelper&) = delete;
        OffloadHelper& operator=(const OffloadHelper&) = delete;
    };
class OffloadHelperConst
    {
    protected:
        const DataBlock<T> &pdL;
        int pdevicenum;
    public:
        inline OffloadHelperConst(const DataBlock<T>& dL, int devicenum, bool just_alloc)
            :pdL(dL),pdevicenum(devicenum)
        {
#if !defined(Unified_Shared_Memory)
            if (just_alloc)
                DataBlock_GPU_Memory_Functions::create_out_struct(dL, devicenum);
            else
                DataBlock_GPU_Memory_Functions::create_in_struct(dL, devicenum);
#endif
        }

        inline  ~OffloadHelperConst()
        {
#if !defined(Unified_Shared_Memory)
            DataBlock_GPU_Memory_Functions::release_struct(pdL, pdevicenum);
#endif
        }

        OffloadHelperConst(const OffloadHelperConst&) = delete;
        OffloadHelperConst& operator=(const OffloadHelperConst&) = delete;
    };


    inline static bool update_device(DataBlock<T>& dL,int devicenum);
    inline static bool update_host(DataBlock<T>& dL,int devicenum);
    inline static bool update_device_data(DataBlock<T>& dL,int devicenum);
    inline static void update_device_metadata(DataBlock<T>& dL,int devicenum);
    inline static bool update_host_data(DataBlock<T>& dL,int devicenum);
    inline static void update_host_metadata(DataBlock<T>& dL,int devicenum);
    inline static void set_data_to_device_ptr(DataBlock<T>& dL,int devicenum);
    inline static void set_data_to_host_ptr(DataBlock<T>& dL,int devicenum);
    inline static void create_out_struct(DataBlock<T>& dA,int devicenum);
    inline static void create_in_struct(DataBlock<T>& dA,int devicenum);
    inline static void exit_struct(DataBlock<T> &dA,int devicenum);
    inline static void release_struct(DataBlock<T> &dA,int devicenum);

    inline static void create_out_struct(const DataBlock<T>& dA,int devicenum);
    inline static void create_in_struct(const DataBlock<T>& dA,int devicenum);
    inline static void exit_struct(const DataBlock<T> &dA,int devicenum);
    inline static void release_struct(const DataBlock<T> &dA,int devicenum);


    inline static void copy_data_to_device_set_devptr(DataBlock<T>&dL,int devicenum);
    inline static void alloc_data_to_device_set_devptr(DataBlock<T>&dL,int devicenum);
    inline static void copy_data_to_host_set_host_ptr(DataBlock<T>&dL);
    inline static void free_device_data_set_host_ptr(DataBlock<T>&dL);


    inline static void copy_data_to_host_ptr(DataBlock<T>& dL);
    inline static void copy_data_to_device_ptr(DataBlock<T>& dL);
    inline static T* alloc_device_ptr(size_t length, int devicenum);
    inline static void free_device_ptr(T* deviceptr, int devicenum);


    inline static T* alloc_data_device_ptr(size_t datalength,bool with_memmap, int devicenum);
    inline static void free_data_device_ptr(T*pdata,size_t datalength,bool with_memmap, int devicenum);

    inline static DataBlock<T> alloc_data_copy_strides_extents_device(size_t datalength,bool rowmajor, size_t rank, size_t*extents,size_t *strides, bool with_memmap, int devicenum);
    inline static void free_copy_device(DataBlock<T>&m, bool with_memmap, int devicenum);

    inline static bool is_on_gpu(const DataBlock<T>&m,const int devicenum);
    inline static bool is_on_gpu_ptr(const T* pdata,const int devicenum);
    inline static bool is_on_gpu_ptr(const size_t* p,const int devicenum);

};

template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::is_on_gpu(const DataBlock<T> &m,const int devicenum)
{
    if(m.dpdata_is_devptr)
        return true;
    if (omp_target_is_present(m.dpdata,devicenum))
        return true;
    return false;
}

template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::is_on_gpu_ptr(const T* pdata,const int devicenum)
{
    if (omp_target_is_present(pdata,devicenum))
        return true;
    return false;
}
template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::is_on_gpu_ptr(const size_t* p,const int devicenum)
{
    if (omp_target_is_present(p,devicenum))
        return true;
    return false;
}

template<typename T>
T* DataBlock_GPU_Memory_Functions<T>::alloc_data_device_ptr(size_t datalength,bool with_memmap, int devicenum)
{

#if defined(Unified_Shared_Memory)
    return DataBlock_Host_Memory_Functions<T>::alloc_data_ptr(datalength, with_memmap);
#else
    return (T*)omp_target_alloc(sizeof(T)*datalength,devicenum);
#endif
}


template<typename T>
DataBlock<T> DataBlock_GPU_Memory_Functions<T>::alloc_data_copy_strides_extents_device(size_t datalength,bool rowmajor, size_t rank, size_t*extents,size_t *strides, bool with_memmap, int devicenum)
{
#if defined(Unified_Shared_Memory)
    return DataBlock_Host_Memory_Functions<T>::alloc_data_copy_strides_extents( datalength, rowmajor,  rank, extents, strides,  with_memmap);
#else

    size_t*pextents;
    size_t*pstrides;
    T* pdata;
    pextents=(size_t*) malloc(sizeof(size_t)*rank);
    memcpy(pextents,extents,sizeof(size_t)*rank);

    pstrides=(size_t*) malloc(sizeof(size_t)*rank);
    memcpy(pstrides,strides,sizeof(size_t)*rank);

    pdata=(T*)omp_target_alloc(sizeof(T)*datalength,devicenum);
    return DataBlock<T>(pdata,datalength,rowmajor,rank,pextents,pstrides,false, false,true);

#endif
}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::free_copy_device(DataBlock<T>&m, bool with_memmap,int devicenum)
{

#if defined(Unified_Shared_Memory)
    DataBlock_Host_Memory_Functions<T>::free_copy(m,with_memmap);
#else

    if(m.dpdata_is_devptr)
        omp_target_free(m.dpdata,devicenum);
    else
        free(m.dpdata);

    free(m.dpextents);
    free(m.dpstrides);
#endif
}

template<typename T>
void DataBlock_GPU_Memory_Functions<T>::free_data_device_ptr(T*pdata,size_t datalength,bool with_memmap, int devicenum)
{
#if defined(Unified_Shared_Memory)
    if(pdata!=nullptr)
        DataBlock_Host_Memory_Functions<T>::free_data_ptr(pdata,datalength,with_memmap);
#else
    if(pdata!=nullptr)
        omp_target_free(pdata,devicenum);
#endif
}







template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::update_device_data(DataBlock<T>& dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dL.dpdatalength;

    #pragma omp target update to (dL) device(devicenum)
    if(!dL.dpdata_is_devptr)
    {
        #pragma omp target update to (dL.dpdata[0:l])device(devicenum)
        return true;
    }
    else
        return false;

#endif
    return true;
}

template<typename T>
void DataBlock_GPU_Memory_Functions<T>::update_device_metadata(DataBlock<T>& dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t r=dL.dprank;
    #pragma omp target update to (dL) device(devicenum)
    #pragma omp target update to (dL.dpextents[0:r])device(devicenum)
    #pragma omp target update to (dL.dpstrides[0:r])device(devicenum)
#endif

}


template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::update_host_data(DataBlock<T>& dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dL.dpdatalength;

    if(!dL.dpdata_is_devptr)
    {
        #pragma omp target update from (dL.dpdata[0:l])device(devicenum)
        return true;
    }
    else
        return false;

#endif
    return true;
}

template<typename T>
void DataBlock_GPU_Memory_Functions<T>::update_host_metadata(DataBlock<T>& dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t r=dL.dprank;
    #pragma omp target update from (dL) device(devicenum)
    #pragma omp target update from (dL.dpextents[0:r])device(devicenum)
    #pragma omp target update from (dL.dpstrides[0:r])device(devicenum)
#endif
}

template<typename T>
void DataBlock_GPU_Memory_Functions<T>::copy_data_to_device_set_devptr(DataBlock<T>&dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    if(!dL.dpdata_is_devptr)
    {
        dL.devptr_former_hostptr=dL.dpdata;
        dL.dpdata=alloc_device_ptr(dL.dpdatalength,devicenum);
        dL.devptr_devicenum=devicenum;
        dL.dpdata_is_devptr=true;
        omp_target_memcpy(dL.dpdata,dL.devptr_former_hostptr,sizeof(T)* dL.dpdatalength,0,0,dL.devptr_devicenum, omp_get_initial_device());
    }
#endif
}

template<typename T>
void DataBlock_GPU_Memory_Functions<T>::alloc_data_to_device_set_devptr(DataBlock<T>&dL, int devicenum)
{
#if !defined(Unified_Shared_Memory)
    if(!dL.dpdata_is_devptr)
    {
        dL.former_host_ptr=dL.dpdata;
        dL.dpdata=alloc_device_ptr(dL.dpdatalength,devicenum);
        dL.dpdata_is_devptr=true;
        dL.devptr_devicenum=devicenum;
    }
#endif
}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::copy_data_to_host_set_host_ptr(DataBlock<T>&dL)
{
#if !defined(Unified_Shared_Memory)
    if(dL.dpdata_is_devptr)

    {
        omp_target_memcpy(dL.devptr_former_hostptr,dL.dpdata,sizeof(T)* dL.dpdatalength,0,0, omp_get_initial_device(),dL.devptr_devicenum);
        free_device_ptr(dL.dpdata, dL.devptr_devicenum);
        dL.dpdata=dL.devptr_former_hostptr;
        dL.dpdata_is_devptr=false;
        dL.devptr_devicenum=-1;
        dL.devptr_former_hostptr=nullptr;
    }
#endif
}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::free_device_data_set_host_ptr(DataBlock<T>&dL)
{
#if !defined(Unified_Shared_Memory)
    if(dL.dpdata_is_devptr)
    {
        omp_target_free(dL.dpdata,dL.devptr_devicenum);
        dL.dpdata=dL.devptr_former_hostptr;
        dL.dpdata_is_devptr=false;
        dL.devptr_devicenum=-1;
        dL.devptr_former_hostptr=nullptr;
    }
#endif
}

template<typename T>
T* DataBlock_GPU_Memory_Functions<T>::alloc_device_ptr(size_t length, int devicenum)
{
#if !defined(Unified_Shared_Memory)
    return (T*)omp_target_alloc(sizeof(T)*length, devicenum);
#else
    return (T*)malloc(sizeof(T)*length);
#endif

}
template<typename T>
void DataBlock_GPU_Memory_Functions<T>::free_device_ptr(T* deviceptr, int devicenum)
{
#if !defined(Unified_Shared_Memory)
    omp_target_free(deviceptr, devicenum);
#else
    free(deviceptr);
#endif

}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::copy_data_to_device_ptr(DataBlock<T>& dL)
{
#if !defined(Unified_Shared_Memory)
    if(dL.dpdata!=dL.devptr_former_hostptr)
        omp_target_memcpy(dL.dpdata,dL.devptr_former_hostptr,sizeof(T)*dL.dpdatalength,0,0,dL.devptr_devicenum,omp_get_initial_device());
#else
    if(dL.dpdata!=dL.devptr_former_hostptr)
        memcpy(dL.dpdata,dL.devptr_former_hostptr,sizeof(T)* dL.dpdatalength);
#endif

}

template<typename T>
void DataBlock_GPU_Memory_Functions<T>::copy_data_to_host_ptr(DataBlock<T>& dL)
{
#if !defined(Unified_Shared_Memory)
    if(dL.dpdata!=dL.devptr_former_hostptr)
        omp_target_memcpy(dL.devptr_former_hostptr,dL.dpdata,sizeof(T)*dL.dpdatalength,0,0,omp_get_initial_device(),dL.devptr_devicenum);
#else
    if(dL.dpdata!=dL.devptr_former_hostptr)
        memcpy(dL.devptr_former_hostptr,dL.dpdata,sizeof(T)* dL.dpdatalength);
#endif
}


template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::update_device(DataBlock<T>& dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dL.dpdatalength;
    size_t r=dL.dprank;

    #pragma omp target update to (dL) device(devicenum)
    #pragma omp target update to (dL.dpextents[0:r])device(devicenum)
    #pragma omp target update to (dL.dpstrides[0:r])device(devicenum)
    if(!dL.dpdata_is_devptr)
    {
        #pragma omp target update to (dL.dpdata[0:l])device(devicenum)
        return true;
    }
    else
        return false;
#endif
    return true;
}


template<typename T>
bool DataBlock_GPU_Memory_Functions<T>::update_host(DataBlock<T>& dL,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dL.dpdatalength;
    size_t r=dL.dprank;

    #pragma omp target update from (dL) device(devicenum)
    #pragma omp target update from (dL.dpstrides[0:r])device(devicenum)
    #pragma omp target update from (dL.dpextents[0:r])device(devicenum)
    if(!dL.dpdata_is_devptr)
    {
        #pragma omp target update from (dL.dpdata[0:l])device(devicenum)
        return true;
    }
    else
        return false;
#endif
    return true;
}




template<typename T>
void  DataBlock_GPU_Memory_Functions<T>::create_out_struct(DataBlock<T>& dA,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    #pragma omp target enter data map(to: dA) device(devicenum)
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target enter data map(alloc: dA.dpdata[0:l])device(devicenum)
    }
    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)
    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)


#endif
}


template<typename T>
void  DataBlock_GPU_Memory_Functions<T>::create_out_struct(const DataBlock<T>& dA,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    const size_t l=dA.dpdatalength;
    const size_t r=dA.dprank;
    #pragma omp target enter data map(to: dA) device(devicenum)
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target enter data map(alloc: dA.dpdata[0:l])device(devicenum)
    }
    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)
    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)


#endif
}




template<typename T>
void  DataBlock_GPU_Memory_Functions<T>::create_in_struct(DataBlock<T>& dA,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    #pragma omp target enter data map(to: dA)device(devicenum)
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)

    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)

#endif
}



template<typename T>
void  DataBlock_GPU_Memory_Functions<T>::create_in_struct(const DataBlock<T>& dA,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    const size_t l=dA.dpdatalength;
    const size_t r=dA.dprank;
    #pragma omp target enter data map(to: dA)device(devicenum)
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)

    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)

#endif
}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::exit_struct(DataBlock<T> &dA,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target exit data map(delete:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(delete:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA)device(devicenum)

#endif
}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::exit_struct(const DataBlock<T> &dA,int devicenum)
{
#if !defined(Unified_Shared_Memory)
    const size_t l=dA.dpdatalength;
    const size_t r=dA.dprank;
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target exit data map(delete:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(delete:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA)device(devicenum)

#endif
}



template<typename T>
void DataBlock_GPU_Memory_Functions<T>::release_struct(DataBlock<T> &dA,int devicenum)
{

#if !defined(Unified_Shared_Memory)
    size_t l=dA.dpdatalength;
    size_t r=dA.dprank;
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target exit data map(release:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(release:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA)device(devicenum)

#endif
}


template<typename T>
void DataBlock_GPU_Memory_Functions<T>::release_struct(const DataBlock<T> &dA,int devicenum)
{

#if !defined(Unified_Shared_Memory)
    const size_t l=dA.dpdatalength;
    const size_t r=dA.dprank;
    if(!dA.dpdata_is_devptr)
    {
        #pragma omp target exit data map(release:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(release:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA)device(devicenum)

#endif
}

#endif


