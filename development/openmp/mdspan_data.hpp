#ifndef MDSPANDATAHPP
#define MDSPANDATAHPP


#include "mdspan_omp.h"

template <typename T, typename Container>
void mdspan_data<T,Container>::initialization_helper(const ManagedDataBlockConfig& config)
{
    this->dpconfig.pmemmap = config.memmap;
    this->p_owns_device_offload = false;

    this->dpconfig= config.Get_DataBlockConfig();

    int target_device = config.devicenum;

    if (config.data_ondevice)
    {
#if defined(Unified_Shared_Memory)
        if (config.memmap)
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength]();

        this->dpconfig.data_is_devptr = false;
#else
        if (config.default_device)
        {
            target_device = omp_get_default_device();
        }
        this->dpdata = GPU_Memory_Functions::alloc_device_ptr<T>(this->dpdatalength, target_device);
        this->dpconfig.devicenum = target_device;
        this->dpconfig.data_is_devptr = true;
        this->devptr_former_hostptr = nullptr;
#endif
    }
    else
    {
        if (config.memmap)
        {
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        }
        else
            this->dpdata = new T[this->dpdatalength]();

        this->dpconfig.data_is_devptr = false;
    }

    p_ref_count = new std::atomic<int>(1);
}


template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(ptrdiff_t datalength, const Container& extents, const Container& strides, ManagedDataBlockConfig config)
    : mdspan<T,Container>(nullptr, extents, strides, config)
{
    initialization_helper(config);
}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(const Container& extents, const Container& strides, ManagedDataBlockConfig config)
    : mdspan<T,Container>(nullptr, extents, strides, config.Get_DataBlockConfig())
{
    initialization_helper(config);
}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(const Container& extents, ManagedDataBlockConfig config)
    : mdspan<T,Container>(nullptr, extents, config.Get_DataBlockConfig())
{
    initialization_helper(config);
}



template <typename T,typename Container>
void mdspan_data<T,Container>::release_all_data()
{
    if (p_ref_count)
    {
        if (p_ref_count->fetch_sub(1, std::memory_order_release) == 1)
        {
            std::atomic_thread_fence(std::memory_order_acquire);

            if (this->p_owns_device_offload)
            {
                this->device_data_release();
            }

            else if (this->dpconfig.data_is_devptr)
            {
                if (this->dpdata != nullptr)
                {
                    GPU_Memory_Functions::free_device_ptr(this->dpdata, this->dpconfig.devicenum);
                    this->dpdata = nullptr;
                }
            }
            if (this->dpdata != nullptr)
            {
                if (this->dpconfig.pmemmap)
                {
                    Host_Memory_Functions::delete_temp_mmap(this->dpdata, this->dpdatalength);
                }
                else
                {
                    delete[] this->dpdata;
                }
                this->dpdata = nullptr;
            }

            delete p_ref_count;
        }
        p_ref_count = nullptr;
    }
}

template <typename T,typename Container>
mdspan_data<T,Container>::~mdspan_data()
{
    release_all_data();
}

template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data( const DataBlock<T>& view,
                                       ManagedDataBlockConfig* alloc_config)
{

    this->dpconfig = view.dpconfig;
    if(alloc_config!=nullptr)
    {

        this->dpconfig.pmemmap = alloc_config->data_ondevice? false: alloc_config->memmap;
        this->dpconfig.data_is_devptr = alloc_config->data_ondevice;
        this->dpconfig.devicenum =alloc_config->default_device? omp_get_default_device(): alloc_config->devicenum;
    }

    this->p_owns_device_offload = false;
    this->dpdatalength = view.dpdatalength;
    this->dprank = view.dprank;
    this->dpconjugate=view.dpconjugate;

    if constexpr (StaticContainer<Container>)
    {
        this->pextents = {};
    }
    if constexpr (StaticContainer<Container>)
    {
        this->pstrides = {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        this->pextents.resize(this->dprank);
    }
    if constexpr (DynamicContainer<Container>)
    {
        this->pstrides.resize(this->dprank);
    }

    std::copy(view.dpextents, view.dpextents + this->dprank, std::begin(this->pextents));
    std::copy(view.dpstrides, view.dpstrides + this->dprank, std::begin(this->pstrides));


    this->dpextents = this->pextents.data();
    this->dpstrides = this->pstrides.data();

    if (this->dpconfig.data_is_devptr)
    {
#if defined(Unified_Shared_Memory)
        if (this->dpconfig.memmap)
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength]();
        this->dpconfig.data_is_devptr = false;
#else
        this->dpdata = GPU_Memory_Functions::alloc_device_ptr<T>(this->dpdatalength, this->dpconfig.devicenum);

#endif
    }
    else
    {
        if (this->dpconfig.pmemmap)
        {
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        }
        else
            this->dpdata = new T[this->dpdatalength]();
        this->dpconfig.data_is_devptr = false;
    }

    int sourcedev= view.dpconfig.data_is_devptr? view.dpconfig.devicenum:omp_get_initial_device();
    if (this->dpconfig.data_is_devptr)
    {
        cout<<"targetdev"<< this->dpconfig.devicenum<<"sourcedev"<< view.dpconfig.devicenum<<endl;
        omp_target_memcpy(this->dpdata,  view.dpdata,    sizeof(T) * this->dpdatalength,     0, 0,  this->dpconfig.devicenum,  sourcedev);
    }
    else
    {
        memcpy(this->dpdata, view.data(), sizeof(T) * this->dpdatalength);
    }


    p_ref_count = new std::atomic<int>(1);
}

template <typename T, typename Container>
mdspan_data<T,Container> mdspan_data<T,Container>::copy(ManagedDataBlockConfig *alloc_config)
{


    int targetdev, sourcedev;
    bool useomptargetmemcpy = false;
    ManagedDataBlockConfig mcfg;


    if (alloc_config!=nullptr)
    {
        mcfg=*alloc_config;
    }
    mcfg.dprowmajor=this->dpconfig.dprowmajor;

    if(mcfg.data_ondevice && this->dpconfig.data_is_devptr)
    {

        targetdev = alloc_config->default_device?omp_get_default_device(): mcfg.devicenum;
        sourcedev = this->dpconfig.devicenum;
        useomptargetmemcpy = true;
    }
    else if(mcfg.data_ondevice && !this->dpconfig.data_is_devptr)
    {

        targetdev =  alloc_config->default_device?omp_get_default_device(): mcfg.devicenum;
        sourcedev = omp_get_initial_device();
        useomptargetmemcpy = true;
    }
    else if(!mcfg.data_ondevice && this->dpconfig.data_is_devptr)
    {
        targetdev = omp_get_initial_device();
        sourcedev = this->dpconfig.devicenum;
        useomptargetmemcpy = true;
    }




    mdspan_data<T,Container> result(this->pextents, this->pstrides, mcfg);
    result.dpconjugate=this->dpconjugate;
    if(useomptargetmemcpy)
    {
        omp_target_memcpy(result.dpdata, this->dpdata, sizeof(T) * this->dpdatalength, 0, 0, targetdev, sourcedev);
    }
    else
    {
        memcpy(result.dpdata, this->dpdata, sizeof(T) * this->dpdatalength);
    }

    return result;
}


template <typename T, typename Container>
mdspan_data<T,Container>& mdspan_data<T, Container>::operator=(const mdspan_data<T,Container>& other)
{
    if(this->dpdata != other.dpdata)
    {
        release_all_data();
        this->p_owns_device_offload = false;
    }

    this->offload_registry = other.offload_registry;
    this->pextents = other.pextents;
    this->pstrides = other.pstrides;

    this->dpextents        = this->pextents.data();
    this->dpstrides        = this->pstrides.data();
    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;

    this->dprank           = other.dprank;
    this->dpconfig          =other.dpconfig;
    this->devptr_former_hostptr= other.devptr_former_hostptr;
    this->dpconjugate=other.dpconjugate;

    p_ref_count = other.p_ref_count;

    if (p_ref_count)
    {
        p_ref_count->fetch_add(1, std::memory_order_relaxed);
    }
    return *this;
}




template<typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(const mdspan_data<T,Container>& other)
    : mdspan<T,Container>(other),
      p_ref_count(other.p_ref_count)
{

    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprank = other.dprank;
    this->dpconfig = other.dpconfig;
    this->devptr_former_hostptr = other.devptr_former_hostptr;
     this->dpconjugate=other.dpconjugate;

    this->memmap = other.memmap;
    this->p_owns_device_offload = false;


    if (p_ref_count)
    {
        p_ref_count->fetch_add(1, std::memory_order_relaxed);
    }
}


template<typename T, typename Container>
mdspan_data<T,Container>&
mdspan_data<T,Container>::operator=(mdspan_data<T,Container>&& other) noexcept
{
    if(this != &other)
    {
        release_all_data();

        mdspan<T,Container>::operator=(std::move(other));

        p_ref_count = other.p_ref_count;
        other.p_ref_count = nullptr;
    }

    return *this;
}


template<typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(mdspan_data<T,Container>&& other) noexcept
    : mdspan<T,Container>(std::move(other)),
      p_ref_count(other.p_ref_count)
{
    other.p_ref_count = nullptr;
}


#endif
