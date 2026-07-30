

#ifndef MDSPAN_HPP
#define MDSPAN_HPP


#include "gpu_memory_functions.h"
template <typename T,typename Container>
mdspan<T,Container>& mdspan<T, Container>:: operator=(const mdspan<T,Container> & other)
{
    if(this->dpdata!=other.dpdata)
    {
        if(p_owns_device_offload)
            this->device_data_release();
        p_owns_device_offload = false;
    }
    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprank           = other.dprank;
    this->dpconfig         = other.dpconfig;
    this->dpconjugate=other.dpconjugate;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;


    offload_registry=other.offload_registry;

    pextents = other.pextents;
    pstrides = other.pstrides;

    this->dpextents        = pextents.data();
    this->dpstrides        = pstrides.data();



    return *this;
}

template <typename T, typename Container>
mdspan<T, Container>&mdspan<T, Container>::operator=(const DataBlock<T> & other)
{

    if(this->dpdata!=other.dpdata)
    {
        if(p_owns_device_offload)
            this->device_data_release();
    }

    this->dpdata           = other.dpdata;
    this->dpdatalength      =other.dpdatalength;
    this->dprank            =other.dprank;
    this->dpconfig         =other.dpconfig;
    this->dpconjugate=other.dpconjugate;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;

    if(pextents.size()!=other.dprank)
        if constexpr (DynamicContainer<Container>)
            pextents.resize(other.dprank);

    if(pextents.data()!=other.dpextents)
        copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

    if(pstrides.size()!=other.dprank)
        if constexpr (DynamicContainer<Container>)
            pstrides.resize(other.dprank);

    if(pstrides.data()!=other.dpstrides)
        copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();



    return *this;
}


template<typename T , typename Container>
mdspan<T, Container>& mdspan<T, Container>::operator=( mdspan<T, Container>&& other)noexcept
{
    if(this->dpdata!=other.dpdata)
    {
        if(p_owns_device_offload)
            this->device_data_release();
    }


    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprank           = other.dprank;
    this->dpconfig         = other.dpconfig;
    this->dpconjugate=other.dpconjugate;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;


    if constexpr (DynamicContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            pextents  = std::move(other.pextents);
    }
    if constexpr (DynamicContainer<Container>)
        pstrides  = std::move(other.pstrides);

    if constexpr (StaticContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));
    }

    if constexpr (StaticContainer<Container>)
    {
        if(pstrides.data()!=other.dpstrides)
            copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));
    }
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();



    offload_registry=std::move(other.offload_registry);

    p_owns_device_offload  = other.p_owns_device_offload;
    other.p_owns_device_offload = false;


    other.dpdata               = nullptr;
    other.dpstrides            = nullptr;
    other.dpextents            = nullptr;
    other.devptr_former_hostptr=nullptr;
    other.dpconfig.dprowmajor=false;
    other.dpconfig.data_is_devptr=false;
    other.dpconfig.devicenum = -INT_MAX;
    other.dpconjugate=false;
    return *this;
}


template<typename T,typename Container>
mdspan<T, Container>::mdspan(const mdspan<T, Container>& other)
{
    p_owns_device_offload = false;
    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprank = other.dprank;
    this->dpconfig = other.dpconfig;
    this->devptr_former_hostptr = other.devptr_former_hostptr;
    this->dpconjugate=other.dpconjugate;
    this->offload_registry = other.offload_registry;


    if constexpr (DynamicContainer<Container>)
    {
        pextents = other.pextents;
    }
    if constexpr (DynamicContainer<Container>)
    {
        pstrides = other.pstrides;
    }

    if constexpr (StaticContainer<Container>)
    {
        if (pextents.data() != other.dpextents)
            std::copy(other.dpextents, other.dpextents + other.dprank, std::begin(pextents));
    }
    if constexpr (StaticContainer<Container>)
    {
        if (pstrides.data() != other.dpstrides)
            std::copy(other.dpstrides, other.dpstrides + other.dprank, std::begin(pstrides));
    }

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();
}


template <typename T,typename Container>
mdspan<T, Container>::mdspan(const DataBlock<T>& other, const shared_ptr<typename mdspan<T,Container>::DeviceOffloadRegistry>& m)
{
    p_owns_device_offload = false;
    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprank = other.dprank;
    this->dpconfig = other.dpconfig;
    this->devptr_former_hostptr = other.devptr_former_hostptr;
     this->dpconjugate=other.dpconjugate;
    this->offload_registry = m;


    if constexpr (DynamicContainer<Container>)
    {
        if (pextents.size() != (size_t)abs(other.dprank)) pextents.resize(abs(other.dprank));
    }
    if constexpr (DynamicContainer<Container>)
    {
        if (pstrides.size() != (size_t)abs(other.dprank)) pstrides.resize(abs(other.dprank));
    }

    if (pextents.data() != other.dpextents)
    {
        std::copy(other.dpextents, other.dpextents + other.dprank, std::begin(pextents));
    }
    if (pstrides.data() != other.dpstrides)
    {
        std::copy(other.dpstrides, other.dpstrides + other.dprank, std::begin(pstrides));
    }


    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(mdspan<T, Container>&& other)noexcept
{

    this->dpdata           = other.dpdata;
    this->dpdatalength      =other.dpdatalength;
    this->dprank            =other.dprank;
    this->dpconfig          =other.dpconfig;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;
    this->dpconjugate=other.dpconjugate;
    if constexpr (DynamicContainer<Container>)
    {
        pextents  = std::move(other.pextents);
    }

    if constexpr (DynamicContainer<Container>)
    {
        pstrides  = std::move(other.pstrides);
    }

    if constexpr (StaticContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));
    }

    if constexpr (StaticContainer<Container>)
    {
        if(pstrides.data()!=other.dpstrides)
            copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));
    }

    offload_registry=std::move(other.offload_registry);

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


    p_owns_device_offload  = other.p_owns_device_offload;

    other.p_owns_device_offload = false;
    other.dpdata               = nullptr;
    other.dpstrides            = nullptr;
    other.dpextents            = nullptr;
    other.devptr_former_hostptr=nullptr;
    other.dpconfig.dprowmajor=false;
    other.dpconfig.data_is_devptr=false;
    other.dpconfig.devicenum = -INT_MAX;
    other.dpconjugate=false;


}



template <typename T, typename Container>
mdspan<T, Container>::~mdspan()
{
    if(p_owns_device_offload)
        this->device_data_release();
}


// Access operator for multidimensional indices
template <typename T,typename Container>
inline T& mdspan<T, Container>::operator()(const Container& indices)
{


    ptrdiff_t offset = 0;
    #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * this->dpstrides[i];
    }
    return this->dpdata[offset];
}


// Access operator for multidimensional indices
template <typename T,typename Container>
T mdspan<T, Container>::operator()(const Container& indices)const
{

    ptrdiff_t offset = 0;
     #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * this->dpstrides[i];
    }
    if constexpr (is_complex<T>::value)
    {
        if (this->dpconfig.dpconjugate)
        {
            return std::conj( this->dpdata[offset]);
        }
    }

    return this->dpdata[offset];
}


template <typename T, typename Container>
void mdspan<T, Container>::compute_initialize_strides(const Container& extents,const bool rowmajor)
{
    const ptrdiff_t n = extents.size();
    if (n == 0)
    {
        this->dpstrides = pstrides.data();
        return;
    }

    if constexpr (StaticContainer<Container>)
    {
        pstrides = {};
    }
    if constexpr (DynamicContainer<Container>)
    {
        pstrides.resize(n);
    }

    if(n==1)
    {
        pstrides[0]=1;
        this->dpstrides = pstrides.data();
        return;
    }

    if (rowmajor)
    {
        pstrides[n - 1] = 1;
        #pragma omp unroll partial
        for (int i =(int) n - 2; i >= 0; --i)
        {
            pstrides[i] = pstrides[i + 1] * extents[i + 1];
        }
    }
    else
    {

        pstrides[0] = 1;
        #pragma omp unroll partial
        for (ptrdiff_t i = 1; i < n; ++i)
        {
            pstrides[i] = pstrides[i - 1] * extents[i - 1];
        }
    }

    this->dpstrides = pstrides.data();

}

template <typename T,typename Container>
void mdspan<T,Container>::initialize_extents_and_strides(const Container& extents, const Container& strides)
{
    const ptrdiff_t r = extents.size();

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
    {
        pstrides.resize(r);
    }

    #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < r; ++i)
    {
        pextents[i] = abs(extents[i]);
        pstrides[i] = strides[i];
    }
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


}


template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents(const Container& extents)
{
    const ptrdiff_t r = extents.size();
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
    {
        pstrides.resize(r);
    }

    #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < r; ++i)
    {
        pextents[i] = abs(extents[i]);
    }
    this->dpextents = pextents.data();
}


template <typename T,typename Container>
mdspan<T, Container>::mdspan(T* data, const  ptrdiff_t datalength, const Container& extents, const Container& strides,const DataBlockConfig  config)
    :DataBlock<T>(data,datalength,extents.size(),nullptr,nullptr, config)
{
    initialize_extents_and_strides(extents,strides);
}



template <typename T,typename Container>
mdspan<T, Container>::mdspan(T* data, const Container& extents, const Container& strides,const DataBlockConfig  config)
    : DataBlock<T>(data, 0,extents.size(),nullptr,nullptr,config)
{
    initialize_extents_and_strides(extents,strides);
    this->dpdatalength=compute_data_length(this->dpextents,this->dpstrides,this->dprank);
}



template <typename T,typename Container>
mdspan<T, Container>::mdspan(T* data, const  Container& extents,const DataBlockConfig  config)
    :  DataBlock<T>(data,0,extents.size(),nullptr,nullptr,  config)
{
    initialize_extents(extents);
    compute_initialize_strides(pextents,config.dprowmajor);
    this->dpdatalength=compute_data_length(this->dpextents,this->dpstrides,this->dprank);
}




template <typename T, typename Container>inline
bool mdspan<T,Container>:: device_data_upload(bool default_device,int devicenum)
{

    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    if(this->dpconfig.data_is_devptr && devicenum==this->dpconfig.devicenum )return false;

    if(offload_registry==nullptr)
    {
        offload_registry = std::make_shared<DeviceOffloadRegistry>();
    }


    if(!offload_registry->insert(devicenum, (intptr_t)this->dpdata, (intptr_t)(this->dpdata+this->dpdatalength)))return false;

    GPU_Memory_Functions::copy_data_to_device_set_devptr(*this,devicenum);

    p_owns_device_offload=true;
    return true;
}

template <typename T, typename Container>
inline
bool mdspan<T, Container>:: device_data_alloc(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    if(this->dpconfig.data_is_devptr && devicenum==this->dpconfig.devicenum)return false;

    if(offload_registry==nullptr)
        offload_registry = std::make_shared<DeviceOffloadRegistry>();

    if(!offload_registry->insert(devicenum, (intptr_t)this->dpdata, (intptr_t)(this->dpdata+this->dpdatalength)))return false;

    GPU_Memory_Functions::alloc_data_to_device_set_devptr(*this,devicenum);
    p_owns_device_offload=true;

    return true;
}

template <typename T,typename Container>
inline
bool mdspan<T, Container>:: device_data_download_release()
{

    if(!p_owns_device_offload)return false;
    if(offload_registry==nullptr) return false;
    if(!offload_registry->remove(this->dpconfig.devicenum, (intptr_t)this->devptr_former_hostptr, (intptr_t)(this->devptr_former_hostptr+this->dpdatalength)))
        return false;

    GPU_Memory_Functions::copy_data_to_host_set_host_ptr(*this);
    p_owns_device_offload=false;

    return true;
}





template <typename T, typename Container>
inline
bool mdspan<T, Container>:: device_data_release()
{
    if(!p_owns_device_offload)return false;
    if(offload_registry==nullptr) return false;

    if(!offload_registry->remove(this->dpconfig.devicenum, (intptr_t)this->devptr_former_hostptr, (intptr_t)(this->devptr_former_hostptr+this->dpdatalength)))
        return false;

    GPU_Memory_Functions::free_device_data_set_host_ptr(*this);
    p_owns_device_offload=false;
    return true;

}

template <typename T,typename Container>
inline
bool mdspan<T, Container>:: host_data_update()
{
    if(!this->dpconfig.data_is_devptr)return false;
    if(this->devptr_former_hostptr==nullptr)return false;

    GPU_Memory_Functions::copy_data_to_host_ptr(*this);
    return true;

}
template <typename T,typename Container>
inline
bool mdspan<T, Container>:: device_data_update()
{
    if(!this->dpconfig.data_is_devptr)return false;
    if(this->devptr_former_hostptr==nullptr)return false;

    GPU_Memory_Functions::copy_data_to_device_ptr(*this);
    return true;

}



#endif

