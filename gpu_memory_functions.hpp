

#ifndef OPENMP_GPU_MEMORY_FUNCTIONS_HPP
#define OPENMP_GPU_MEMORY_FUNCTIONS_HPP

#include <string.h>

template<typename T>
GPU_Memory_Functions::OffloadHelper<T>::OffloadHelper(DataBlock<T> &dL, int devicenum, bool just_alloc,
    bool update_host_on_exit):pupdate_host(update_host_on_exit), pdL(dL),pdevicenum(devicenum) {
#if !defined(Unified_Shared_Memory)
    if (just_alloc)
        GPU_Memory_Functions::create_out(dL, devicenum);
    else
        GPU_Memory_Functions::create_in(dL, devicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::OffloadHelper<T>::OffloadHelper(const DataBlock<T> &dL, int devicenum, bool just_alloc):pupdate_host(false), pdL(dL),pdevicenum(devicenum) {
#if !defined(Unified_Shared_Memory)
    if (just_alloc)
        GPU_Memory_Functions::create_out(dL, devicenum);
    else
        GPU_Memory_Functions::create_in(dL, devicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::OffloadHelper<T>::~OffloadHelper() {
#if !defined(Unified_Shared_Memory)
    if (pupdate_host && !pdL.dpconfig.data_is_devptr)
    {
        GPU_Memory_Functions::update_host(pdL, pdevicenum);
    }
    GPU_Memory_Functions::release(pdL, pdevicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::OffloadHelperConst<T>::
OffloadHelperConst(const DataBlock<T> &dL, int devicenum, bool just_alloc):pdL(dL),pdevicenum(devicenum) {
#if !defined(Unified_Shared_Memory)
    GPU_Memory_Functions::create_in(dL, devicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::OffloadHelperConst<T>::~OffloadHelperConst() {
#if !defined(Unified_Shared_Memory)
    GPU_Memory_Functions::release(pdL, pdevicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::BlockedDataViewOffloadHelper<T>::~BlockedDataViewOffloadHelper() {
#if !defined(Unified_Shared_Memory)
    GPU_Memory_Functions::release_blocked(pdL, pdevicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T>::~DataBlockArrayOffloadHelperConst() {
#if !defined(Unified_Shared_Memory)
    GPU_Memory_Functions::release(pArr, pdevicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::DataBlockdpdataoffloader<T>::DataBlockdpdataoffloader(const DataBlock<T> &block, int devicenum,
    bool is_output):
    m_block(block), m_copied(false), m_is_output(is_output) {
    if (!m_block.dpconfig.data_is_devptr)
    {
        if (m_is_output)
        {
            m_copied = GPU_Memory_Functions::alloc_data_to_device_set_devptr(m_block, devicenum);
        }
        else
        {
            m_copied = GPU_Memory_Functions::copy_data_to_device_set_devptr(m_block, devicenum);
        }
    }
}

template<typename T>
GPU_Memory_Functions::DataBlockdpdataoffloader<T>::DataBlockdpdataoffloader(DataBlock<T> &block, int devicenum,
    bool is_output): m_block(block), m_copied(false), m_is_output(is_output) {
    if (!m_block.dpconfig.data_is_devptr)
    {
        if (m_is_output)
        {
            m_copied = GPU_Memory_Functions::alloc_data_to_device_set_devptr(m_block, devicenum);
        }
        else
        {
            m_copied = GPU_Memory_Functions::copy_data_to_device_set_devptr(m_block, devicenum);
        }
    }
}

template<typename T>
const DataBlock<T> & GPU_Memory_Functions::DataBlockdpdataoffloader<T>::get() const { return m_block; }

template<typename T>
DataBlock<T> & GPU_Memory_Functions::DataBlockdpdataoffloader<T>::get_mutable() { return m_block; }

template<typename T>
GPU_Memory_Functions::DataBlockdpdataoffloader<T>::~DataBlockdpdataoffloader() {
    if (m_copied)
    {
        if (m_is_output)
        {
            GPU_Memory_Functions::copy_data_to_host_set_host_ptr(m_block);
        }
        else
        {
            GPU_Memory_Functions::free_device_data_set_host_ptr(m_block);
        }
    }
}

template<typename T>
GPU_Memory_Functions::BlockedDataViewOffloadHelper<T>::BlockedDataViewOffloadHelper(const BlockedDataView<T> &dL,
    int devicenum):pdL(dL),pdevicenum(devicenum) {
#if !defined(Unified_Shared_Memory)
    GPU_Memory_Functions::create_in_blocked(dL, devicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>::DataBlockArrayOffloadHelper(DataBlockArray<T> &arr, int devicenum,
    bool just_alloc, bool update_host_on_exit): pupdate_host(update_host_on_exit), pArr(arr), pdevicenum(devicenum) {
#if !defined(Unified_Shared_Memory)
    if (just_alloc)
        GPU_Memory_Functions::create_out(arr, devicenum);
    else
        GPU_Memory_Functions::create_in(arr, devicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::DataBlockArrayOffloadHelper<T>::~DataBlockArrayOffloadHelper() {
#if !defined(Unified_Shared_Memory)
    if (pupdate_host && !pArr.pdata_is_devptr)
    {
        GPU_Memory_Functions::update_host(pArr, pdevicenum);
    }
    GPU_Memory_Functions::release(pArr, pdevicenum);
#endif
}

template<typename T>
GPU_Memory_Functions::DataBlockArrayOffloadHelperConst<T>::DataBlockArrayOffloadHelperConst(
    const DataBlockArray<T> &arr, int devicenum): pArr(arr), pdevicenum(devicenum) {
#if !defined(Unified_Shared_Memory)
    GPU_Memory_Functions::create_in(arr, devicenum);
#endif
}

template<typename T>
void GPU_Memory_Functions::create_out(DataBlockArray<T>& arr, int devicenum)
{
    if (arr.pnumblocks == 0) return;
    if (arr.pdata_is_devptr) devicenum = arr.pdevnum;

    const ptrdiff_t l = arr.pdatalength;
    const ptrdiff_t nb = arr.pnumblocks;
    const ptrdiff_t rank = arr.ptensor_rank;


    #pragma omp target enter data map(to: arr) device(devicenum)


    if(!arr.pdata_is_devptr && arr.pdata != nullptr)
    {
        #pragma omp target enter data map(alloc: arr.pdata[0:l]) device(devicenum)
    }

    if (arr.pblock_offsets)
    {
        #pragma omp target enter data map(to: arr.pblock_offsets[0:nb]) device(devicenum)
    }
    if (arr.pextentsbuffer)
    {
        #pragma omp target enter data map(to: arr.pextentsbuffer[0:nb*rank]) device(devicenum)
    }
    if (arr.pstridesbuffer)
    {
        #pragma omp target enter data map(to: arr.pstridesbuffer[0:nb*rank]) device(devicenum)
    }
}

template<typename T>
void GPU_Memory_Functions::create_in(DataBlockArray<T>& arr, int devicenum)
{
    if (arr.pnumblocks == 0) return;
    if (arr.pdata_is_devptr) devicenum = arr.pdevnum;

    const ptrdiff_t l = arr.pdatalength;
    const ptrdiff_t nb = arr.pnumblocks;
    const ptrdiff_t rank = arr.ptensor_rank;

    #pragma omp target enter data map(to: arr) device(devicenum)
    if(!arr.pdata_is_devptr && arr.pdata != nullptr)
    {
        #pragma omp target enter data map(to: arr.pdata[0:l]) device(devicenum)
    }
    if (arr.pblock_offsets)
    {
        #pragma omp target enter data map(to: arr.pblock_offsets[0:nb]) device(devicenum)
    }
    if (arr.pextentsbuffer)
    {
        #pragma omp target enter data map(to: arr.pextentsbuffer[0:nb*rank]) device(devicenum)
    }
    if (arr.pstridesbuffer)
    {
        #pragma omp target enter data map(to: arr.pstridesbuffer[0:nb*rank]) device(devicenum)
    }
}

template<typename T>
void GPU_Memory_Functions::create_in(const DataBlockArray<T>& arr, int devicenum)
{
    if (arr.pnumblocks == 0) return;
    if (arr.pdata_is_devptr) devicenum = arr.pdevnum;

    const  ptrdiff_t l = arr.pdatalength;
    const  ptrdiff_t nb = arr.pnumblocks;
    const ptrdiff_t rank = arr.ptensor_rank;

    #pragma omp target enter data map(to: arr) device(devicenum)
    if(!arr.pdata_is_devptr && arr.pdata != nullptr)
    {
        #pragma omp target enter data map(to: arr.pdata[0:l]) device(devicenum)
    }
    if (arr.pblock_offsets)
    {
        #pragma omp target enter data map(to: arr.pblock_offsets[0:nb]) device(devicenum)
    }
    if (arr.pextentsbuffer)
    {
        #pragma omp target enter data map(to: arr.pextentsbuffer[0:nb*rank]) device(devicenum)
    }
    if (arr.pstridesbuffer)
    {
        #pragma omp target enter data map(to: arr.pstridesbuffer[0:nb*rank]) device(devicenum)
    }
}

template<typename T>
void GPU_Memory_Functions::update_host(DataBlockArray<T>& arr, int devicenum)
{
    if (arr.pnumblocks == 0 || arr.pdata_is_devptr || arr.pdata == nullptr) return;
    ptrdiff_t l = arr.pdatalength;
    #pragma omp target update from(arr.pdata[0:l]) device(devicenum)
}

template<typename T>
void GPU_Memory_Functions::release(const DataBlockArray<T>& arr, int devicenum)
{
    if (arr.pnumblocks == 0) return;
    if (arr.pdata_is_devptr) devicenum = arr.pdevnum;

    const ptrdiff_t l = arr.pdatalength;
    const ptrdiff_t nb = arr.pnumblocks;
    const ptrdiff_t rank = arr.ptensor_rank;

    if (arr.pstridesbuffer)
    {
        #pragma omp target exit data map(release: arr.pstridesbuffer[0:nb*rank]) device(devicenum)
    }
    if (arr.pextentsbuffer)
    {
        #pragma omp target exit data map(release: arr.pextentsbuffer[0:nb*rank]) device(devicenum)
    }
    if (arr.pblock_offsets)
    {
        #pragma omp target exit data map(release: arr.pblock_offsets[0:nb]) device(devicenum)
    }
    if(!arr.pdata_is_devptr && arr.pdata != nullptr)
    {
        #pragma omp target exit data map(release: arr.pdata[0:l]) device(devicenum)
    }
    #pragma omp target exit data map(release: arr) device(devicenum)
}



template<typename T>
bool GPU_Memory_Functions::is_on_gpu(const DataBlock<T> &m,const int devicenum)
{
    if(m.dpconfig.data_is_devptr)
        return true;
    if (omp_target_is_present(m.dpdata,devicenum))
        return true;
    return false;
}

template<typename T>
bool GPU_Memory_Functions::is_on_gpu(const DataBlockArray<T> &m,const int devicenum)
{
    if(m.pdata_is_devptr)
        return true;
    if (omp_target_is_present(m.pdata,devicenum))
        return true;
    return false;
}

template<typename T>
bool GPU_Memory_Functions::is_on_gpu_ptr(const T* pdata,const int devicenum)
{
    if (omp_target_is_present(pdata,devicenum))
        return true;
    return false;
}
template<typename T>
bool GPU_Memory_Functions::is_on_gpu_ptr(const ptrdiff_t* p,const int devicenum)
{
    if (omp_target_is_present(p,devicenum))
        return true;
    return false;
}

template<typename T>
T* GPU_Memory_Functions::alloc_data_device_ptr(ptrdiff_t datalength,bool with_memmap, int devicenum)
{

#if defined(Unified_Shared_Memory)
    return Host_Memory_Functions<T>::alloc_data_ptr(datalength, with_memmap);
#else
    return (T*)omp_target_alloc(sizeof(T)*datalength,devicenum);
#endif
}



template<typename T>
void GPU_Memory_Functions::free_data_device_ptr(T*&pdata,ptrdiff_t datalength,bool with_memmap, int devicenum)
{
#if defined(Unified_Shared_Memory)
    if(pdata!=nullptr)
        Host_Memory_Functions<T>::free_data_ptr(pdata,datalength,with_memmap);
#else
    if(pdata!=nullptr)
        omp_target_free(pdata,devicenum);
#endif
}

template<typename T>
bool GPU_Memory_Functions::update_device_data(DataBlock<T>& dL,int devicenum)
{

    if (dL.dpdata==nullptr)
        return false;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dL.dpdatalength;

    #pragma omp target update to (dL) device(devicenum)
    if(!dL.dpconfig.data_is_devptr)
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
void GPU_Memory_Functions::update_device_metadata(DataBlock<T>& dL,int devicenum)
{
    if (dL.dpextents==nullptr)
        return;
    if (dL.dpstrides==nullptr)
        return;
#if !defined(Unified_Shared_Memory)
    ptrdiff_t r=dL.dprank;
    #pragma omp target update to (dL) device(devicenum)
    #pragma omp target update to (dL.dpextents[0:r])device(devicenum)
    #pragma omp target update to (dL.dpstrides[0:r])device(devicenum)
#endif

}


template<typename T>
bool GPU_Memory_Functions::update_host_data(DataBlock<T>& dL,int devicenum)
{

    if (dL.dpdata==nullptr)
        return false;
#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dL.dpdatalength;

    if(!dL.dpconfig.data_is_devptr)
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
void GPU_Memory_Functions::update_host_metadata(DataBlock<T>& dL,int devicenum)
{
    if (dL.dpextents==nullptr)
        return;
    if (dL.dpstrides==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t r=dL.dprank;
    #pragma omp target update from (dL) device(devicenum)
    #pragma omp target update from (dL.dpextents[0:r])device(devicenum)
    #pragma omp target update from (dL.dpstrides[0:r])device(devicenum)
#endif
}

template<typename T>
bool GPU_Memory_Functions::copy_data_to_device_set_devptr(DataBlock<T>&dL,int devicenum)
{

#if !defined(Unified_Shared_Memory)
    if(!dL.dpconfig.data_is_devptr)
    {
        dL.devptr_former_hostptr=dL.dpdata;
        dL.dpdata=GPU_Memory_Functions::alloc_device_ptr<T>(dL.dpdatalength,devicenum);
        dL.dpconfig.devicenum=devicenum;
        dL.dpconfig.data_is_devptr=true;
        omp_target_memcpy(dL.dpdata,dL.devptr_former_hostptr,sizeof(T)* dL.dpdatalength,0,0,dL.dpconfig.devicenum, omp_get_initial_device());
        return true;
    }
#endif
    return false;
}

template<typename T>
bool GPU_Memory_Functions::alloc_data_to_device_set_devptr(DataBlock<T>&dL, int devicenum)
{

#if !defined(Unified_Shared_Memory)
    if(!dL.dpconfig.data_is_devptr)
    {
        dL.devptr_former_hostptr=dL.dpdata;
        dL.dpdata=alloc_device_ptr<T>(dL.dpdatalength,devicenum);
        dL.dpconfig.data_is_devptr=true;
        dL.dpconfig.devicenum=devicenum;
        return true;
    }
#endif
    return false;
}


template<typename T>
bool GPU_Memory_Functions::copy_data_to_host_set_host_ptr(DataBlock<T>&dL)
{
    if (dL.dpdata==nullptr)
        return false;
    if (dL.devptr_former_hostptr==nullptr)
        return false;
#if !defined(Unified_Shared_Memory)
    if(dL.dpconfig.data_is_devptr)
    {
        omp_target_memcpy(dL.devptr_former_hostptr,dL.dpdata,sizeof(T)* dL.dpdatalength,0,0, omp_get_initial_device(),dL.dpconfig.devicenum);
        free_device_ptr(dL.dpdata, dL.dpconfig.devicenum);
        dL.dpdata=dL.devptr_former_hostptr;
        dL.dpconfig.data_is_devptr=false;
        dL.dpconfig.devicenum=-INT_MAX;
        dL.devptr_former_hostptr=nullptr;
        return true;
    }
#endif
    return false;
}


template<typename T>
bool GPU_Memory_Functions::free_device_data_set_host_ptr(DataBlock<T>&dL)
{
    if (dL.dpdata==nullptr)
        return false;
    if (dL.devptr_former_hostptr==nullptr)
        return false;
#if !defined(Unified_Shared_Memory)
    if(dL.dpconfig.data_is_devptr)
    {
        omp_target_free(dL.dpdata,dL.dpconfig.devicenum);
        dL.dpdata=dL.devptr_former_hostptr;
        dL.dpconfig.data_is_devptr=false;
        dL.dpconfig.devicenum=-INT_MAX;
        dL.devptr_former_hostptr=nullptr;
        return true;
    }
#endif
    return false;
}

template<typename T>
T* GPU_Memory_Functions::alloc_device_ptr(ptrdiff_t length, int devicenum)
{
#if !defined(Unified_Shared_Memory)
    return (T*)omp_target_alloc(sizeof(T)*length, devicenum);
#else
    return (T*)malloc(sizeof(T)*length);
#endif

}
template<typename T>
void GPU_Memory_Functions::free_device_ptr(T* &deviceptr, int devicenum)
{
    if (deviceptr==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    omp_target_free(deviceptr, devicenum);
#else
    free(deviceptr);
#endif
}


template<typename T>
void GPU_Memory_Functions::copy_data_to_device_ptr(DataBlock<T>& dL)
{
    if (dL.dpdata==nullptr)
        return;
    if (dL.devptr_former_hostptr==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    if(dL.dpdata!=dL.devptr_former_hostptr)
        omp_target_memcpy(dL.dpdata,dL.devptr_former_hostptr,sizeof(T)*dL.dpdatalength,0,0,dL.dpconfig.devicenum,omp_get_initial_device());
#else
    if(dL.dpdata!=dL.devptr_former_hostptr)
        memcpy(dL.dpdata,dL.devptr_former_hostptr,sizeof(T)* dL.dpdatalength);
#endif

}

template<typename T>
void GPU_Memory_Functions::copy_data_to_host_ptr(DataBlock<T>& dL)
{
    if (dL.dpdata==nullptr)
        return;
    if (dL.devptr_former_hostptr==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    if(dL.dpdata!=dL.devptr_former_hostptr)
        omp_target_memcpy(dL.devptr_former_hostptr,dL.dpdata,sizeof(T)*dL.dpdatalength,0,0,omp_get_initial_device(),dL.dpconfig.devicenum);
#else
    if(dL.dpdata!=dL.devptr_former_hostptr)
        memcpy(dL.devptr_former_hostptr,dL.dpdata,sizeof(T)* dL.dpdatalength);
#endif
}


template<typename T>
bool GPU_Memory_Functions::update_device(DataBlock<T>& dL,int devicenum)
{

    if (dL.dpdata==nullptr)
        return false;
    if (dL.dpextents==nullptr)
        return false;
    if (dL.dpstrides==nullptr)
        return false;
#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dL.dpdatalength;
    ptrdiff_t r=dL.dprank;
    if(dL.dpconfig.data_is_devptr)
        devicenum=dL.dpconfig.devicenum;
    #pragma omp target update to (dL) device(devicenum)
    #pragma omp target update to (dL.dpextents[0:r])device(devicenum)
    #pragma omp target update to (dL.dpstrides[0:r])device(devicenum)
    if(!dL.dpconfig.data_is_devptr)
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
bool GPU_Memory_Functions::update_host(DataBlock<T>& dL,int devicenum)
{
    if (dL.dpdata==nullptr)
        return false;
    if (dL.dpextents==nullptr)
        return false;
    if (dL.dpstrides==nullptr)
        return false;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dL.dpdatalength;
    ptrdiff_t r=dL.dprank;
    if(dL.dpconfig.data_is_devptr)
        devicenum=dL.dpconfig.devicenum;
    #pragma omp target update from (dL) device(devicenum)
    #pragma omp target update from (dL.dpstrides[0:r])device(devicenum)
    #pragma omp target update from (dL.dpextents[0:r])device(devicenum)
    if(!dL.dpconfig.data_is_devptr)
    {
        #pragma omp target update from (dL.dpdata[0:l])device(devicenum)
        return true;
    }
    else
        return false;
#else
    return true;
#endif

}




template<typename T>
void GPU_Memory_Functions::create_out(DataBlock<T>& dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dA.dpdatalength;
    ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)
        devicenum=dA.dpconfig.devicenum;
    #pragma omp target enter data map(to: dA) device(devicenum)
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target enter data map(alloc: dA.dpdata[0:l])device(devicenum)
    }
    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)
    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)


#endif
}




template<typename T>
void GPU_Memory_Functions::create_in(DataBlock<T>& dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dA.dpdatalength;
    ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)
        devicenum=dA.dpconfig.devicenum;

    #pragma omp target enter data map(to: dA)device(devicenum)
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)

    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)

#endif
}





template<typename T>
void GPU_Memory_Functions::create_in(const DataBlock<T>& dA,int devicenum)
{

    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    const ptrdiff_t l=dA.dpdatalength;
    const ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)
        devicenum=dA.dpconfig.devicenum;

    #pragma omp target enter data map(to: dA)device(devicenum)
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)

    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)

#endif
}

template<typename T>
void GPU_Memory_Functions::create_in_blocked(const BlockedDataView<T>& dA,int devicenum)
{

    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;
    if (dA.block_shape==nullptr)
        return;
    if (dA.pooled_offsets_flat==nullptr)
        return;
    if (dA.pooled_offsets_starts==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t r=dA.dprank;
    ptrdiff_t count=dA.usedblocks;
    ptrdiff_t count2=r*count;
    const ptrdiff_t l=dA.dpdatalength;
    if(dA.dpconfig.data_is_devptr)
        devicenum=dA.dpconfig.devicenum;

    #pragma omp target enter data map(to: dA)device(devicenum)
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target enter data map(to: dA.dpdata[0:l])device(devicenum)
    }
    #pragma omp target enter data map(to: dA.dpextents[0:r])device(devicenum)
    #pragma omp target enter data map(to: dA.dpstrides[0:r])device(devicenum)
    #pragma omp target enter data map(to: dA.block_shape[0:r])device(devicenum)
    if(!dA.offsets_starts_is_devptr)
    {
        #pragma omp target enter data map(to: dA.pooled_offsets_flat[0:count2])device(devicenum)
        #pragma omp target enter data map(to: dA.pooled_offsets_starts[0:count+1])device(devicenum)
    }
#endif
}

template<typename T>
void GPU_Memory_Functions::exit_blocked(const BlockedDataView<T>& dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;
    if (dA.block_shape==nullptr)
        return;
    if (dA.pooled_offsets_flat==nullptr)
        return;
    if (dA.pooled_offsets_starts==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t r=dA.dprank;
    ptrdiff_t count=dA.usedblocks;
    ptrdiff_t count2=r*count;
    const ptrdiff_t l=dA.dpdatalength;
    if(dA.dpconfig.data_is_devptr)
        devicenum=dA.dblock.dpconfig.devicenum;
    if(!dA.offsets_starts_is_devptr)
    {
        #pragma omp target exit data map(delete: dA.pooled_offsets_flat[0:count2])device(devicenum)
        #pragma omp target exit data map(delete: dA.pooled_offsets_starts[0:count+1])device(devicenum)
    }
    #pragma omp target exit data map(delete: dA.block_shape[0:r])device(devicenum)
    #pragma omp target exit data map(delete: dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(delete: dA.dpextents[0:r])device(devicenum)
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target exit data map(delete: dA.dpdata[0:l])device(devicenum)
    }
    #pragma omp target exit data map(delete: dA)device(devicenum)

#endif
}


template<typename T>
void GPU_Memory_Functions::release_blocked(const BlockedDataView<T>& dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;
    if (dA.block_shape==nullptr)
        return;
    if (dA.pooled_offsets_flat==nullptr)
        return;
    if (dA.pooled_offsets_starts==nullptr)
        return;

#if !defined(Unified_Shared_Memory)
    ptrdiff_t r=dA.dprank;
    ptrdiff_t count=dA.usedblocks;
    ptrdiff_t count2=r*count;
    const ptrdiff_t l=dA.dpdatalength;
    if(!dA.offsets_starts_is_devptr)
    {
        #pragma omp target exit data map(release: dA.pooled_offsets_flat[0:count2])device(devicenum)
        #pragma omp target exit data map(release: dA.pooled_offsets_starts[0:count+1])device(devicenum)
    }
    #pragma omp target exit data map(release: dA.block_shape[0:r])device(devicenum)
    #pragma omp target exit data map(release: dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(release: dA.dpextents[0:r])device(devicenum)
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target exit data map(release: dA.dpdata[0:l])device(devicenum)
    }
    #pragma omp target exit data map(release: dA)device(devicenum)
#endif
}


template<typename T>
DataBlock<T> GPU_Memory_Functions::alloc_data_copy_strides_extents_device(ptrdiff_t datalength,bool rowmajor, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,  DataBlockConfig conf)
{
#if defined(Unified_Shared_Memory)
    return Host_Memory_Functions<T>::alloc_data_copy_strides_extents( datalength,   rank, extents, strides, conf);
#else

    ptrdiff_t*pextents;
    ptrdiff_t*pstrides;
    T* pdata;
    pextents=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);
    memcpy(pextents,extents,sizeof(ptrdiff_t)*rank);

    pstrides=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);
    memcpy(pstrides,strides,sizeof(ptrdiff_t)*rank);

    pdata=(T*)omp_target_alloc(sizeof(T)*datalength, conf.devicenum);
    conf.data_is_devptr=true;
    return DataBlock<T>(pdata,datalength,rank,pextents,pstrides,conf);
#endif
}

template<typename T>
DataBlock<T> GPU_Memory_Functions::alloc_data_strides_extents_device(ptrdiff_t datalength,bool rowmajor, ptrdiff_t rank, ptrdiff_t*extents,ptrdiff_t *strides,  DataBlockConfig conf)
{
#if defined(Unified_Shared_Memory)
    return Host_Memory_Functions<T>::alloc_data_copy_strides_extents( datalength,   rank, extents, strides, conf);
#else

    ptrdiff_t*pextents;
    ptrdiff_t*pstrides;
    T* pdata;
    pextents=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);
    pstrides=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*rank);

    pdata=(T*)omp_target_alloc(sizeof(T)*datalength, conf.devicenum);
    conf.data_is_devptr=true;
    return DataBlock<T>(pdata,datalength,rank,pextents,pstrides,conf);
#endif
}


template<typename T>
void GPU_Memory_Functions::free_copy_device(DataBlock<T>&m)
{

#if defined(Unified_Shared_Memory)
    Host_Memory_Functions<T>::free_copy(m);
#else

    if(m.dpdata!=nullptr)
    {
        if(m.dpconfig.data_is_devptr)
            omp_target_free(m.dpdata,m.dpconfig.devicenum);
        else
            free(m.dpdata);
    }
    if(m.dpextents!=nullptr)
        free(m.dpextents);

    if(m.dpstrides!=nullptr)
        free(m.dpstrides);

    m.dpconfig.devicenum=-INT_MAX;
    m.dpconfig.data_is_devptr=false;
#endif
}




template<typename T>
void GPU_Memory_Functions::exit(DataBlock<T> &dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;


#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dA.dpdatalength;
    ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)devicenum=dA.dpconfig.devicenum;
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target exit data map(delete:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(delete:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA)device(devicenum)

#endif
}


template<typename T>
void GPU_Memory_Functions::exit(const DataBlock<T> &dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;


#if !defined(Unified_Shared_Memory)
    const ptrdiff_t l=dA.dpdatalength;
    const ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)
        devicenum=dA.dpconfig.devicenum;
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target exit data map(delete:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(delete:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(delete:dA)device(devicenum)

#endif
}



template<typename T>
void GPU_Memory_Functions::release(DataBlock<T> &dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;


#if !defined(Unified_Shared_Memory)
    ptrdiff_t l=dA.dpdatalength;
    ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)devicenum=dA.dpconfig.devicenum;
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target exit data map(release:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(release:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA)device(devicenum)

#endif
}


template<typename T>
void GPU_Memory_Functions::release(const DataBlock<T> &dA,int devicenum)
{
    if (dA.dpdata==nullptr)
        return;
    if (dA.dpextents==nullptr)
        return;
    if (dA.dpstrides==nullptr)
        return;


#if !defined(Unified_Shared_Memory)
    const ptrdiff_t l=dA.dpdatalength;
    const ptrdiff_t r=dA.dprank;
    if(dA.dpconfig.data_is_devptr)devicenum=dA.dpconfig.devicenum;
    if(!dA.dpconfig.data_is_devptr)
    {
        #pragma omp target exit data map(release:dA.dpdata[0:l])device(devicenum)
    }

    #pragma omp target exit data map(release:dA.dpstrides[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA.dpextents[0:r])device(devicenum)
    #pragma omp target exit data map(release:dA)device(devicenum)

#endif

}


#endif //OPENMP_GPU_MEMORY_FUNCTIONS_HPP
