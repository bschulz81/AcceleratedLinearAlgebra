#ifndef MDSPAN_DATAH
#define MDSPAN_DATAH

#include "mdspan_omp.h"
#include "string.h"
#include "indiceshelperfunctions.h"

#include <atomic>

class mdspan_utilities;


struct ManagedDataBlockConfig
{
    bool dprowmajor=true;
    bool dpconjugate=false;
    bool memmap=false;
    bool data_ondevice=false;
    bool default_device=true;
    int devicenum = -INT_MAX;


    inline DataBlockConfig Get_DataBlockConfig() const
    {
        return DataBlockConfig
        {
            .dprowmajor    = this->dprowmajor,
            .dpconjugate   = this->dpconjugate,
            .pmemmap=this->memmap,
            .data_is_devptr = this->data_ondevice,
            .devicenum     = this->devicenum
        };
    }

    inline static ManagedDataBlockConfig SetConfig( bool defaultdevice, const DataBlockConfig& config)
    {
        return ManagedDataBlockConfig
        {
            .dprowmajor     = config.dprowmajor,
            .dpconjugate    = config.dpconjugate,
            .memmap         =config.pmemmap,
            .data_ondevice  = config.data_is_devptr,
            .default_device = defaultdevice,
            .devicenum      = config.devicenum
        };
    }
};

template <typename T,typename Container>
class mdspan_data : public mdspan<T,Container>
{
public:

    friend class mdspan_utilities;

    mdspan_data() {};


    mdspan_data(ptrdiff_t datalength, const Container& extents, const Container& strides,ManagedDataBlockConfig config);

    mdspan_data(const Container& extents, const Container& strides, ManagedDataBlockConfig config);

    mdspan_data(const Container& extents,ManagedDataBlockConfig config);

    mdspan_data( const DataBlock<T>& view, ManagedDataBlockConfig alloc_config) ;

    mdspan_data(const mdspan_data<T, Container>& other);
    mdspan_data<T, Container>&operator=(const mdspan_data<T,Container> & other);

    mdspan_data(mdspan_data<T, Container>&& other) noexcept;
    mdspan_data<T,Container>& operator=( mdspan_data<T, Container>&& other) noexcept;



    ~mdspan_data();



    using DataBlock<T>::operator=;

    mdspan_data<T,Container> copy( ManagedDataBlockConfig config);

    void release_all_data();
protected:
    std::atomic<int>* p_ref_count = nullptr;
    void initialization_helper(const ManagedDataBlockConfig& config);
};


template<typename T, typename Tag>
using mdspan_data_t =
    mdspan_data<
        T,
        typename container_selector<Tag>::template container<ptrdiff_t>>;


template <typename T, typename Container>
void mdspan_data<T,Container>::initialization_helper(const ManagedDataBlockConfig& config)
{
    this->dpconfig.pmemmap = config.memmap;
    this->p_has_offloaded_host_data = false;

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

            if (this->p_has_offloaded_host_data)
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
        ManagedDataBlockConfig alloc_config)
{

    this->dpconfig.dprowmajor = view.dpconfig.dprowmajor;
    this->dpconfig.dpconjugate = view.dpconfig.dpconjugate;

    this->dpconfig.pmemmap = alloc_config.data_ondevice? false: alloc_config.memmap;
    this->dpconfig.data_is_devptr = alloc_config.data_ondevice;
    this->dpconfig.devicenum =alloc_config.default_device? omp_get_default_device(): alloc_config.devicenum;
    this->p_has_offloaded_host_data = false;
    this->dpdatalength = view.dpdatalength;
    this->dprank = view.dprank;

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

    std::copy(view.extents(), view.extents() + this->dprank, std::begin(this->pextents));
    std::copy(view.strides(), view.strides() + this->dprank, std::begin(this->pstrides));


    this->dpextents = this->pextents.data();
    this->dpstrides = this->pstrides.data();


    int target_device = alloc_config.devicenum;

    if (alloc_config.data_ondevice && alloc_config.default_device)
    {
        target_device = omp_get_default_device();
    }

    if (alloc_config.data_ondevice)
    {
#if defined(Unified_Shared_Memory)
        if (alloc_config.memmap)
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength]();
        this->dpconfig.data_is_devptr = false;
#else
        this->dpdata = GPU_Memory_Functions::alloc_device_ptr<T>(this->dpdatalength, target_device);
        this->dpconfig.devicenum = target_device;
        this->dpconfig.data_is_devptr = true;
        this->devptr_former_hostptr = nullptr;
#endif
    }
    else
    {
        if (alloc_config.memmap)
        {
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        }


        else
            this->dpdata = new T[this->dpdatalength]();
        this->dpconfig.data_is_devptr = false;
    }


    if (this->dpconfig.data_is_devptr)
    {
        int source_device = view.dpconfig.data_is_devptr ? view.dpconfig.devicenum : omp_get_initial_device();
        omp_target_memcpy(
            this->dpdata,
            view.data(),
            sizeof(T) * this->dpdatalength,
            0, 0,
            target_device,
            source_device
        );
    }
    else
    {
        memcpy(this->dpdata, view.data(), sizeof(T) * this->dpdatalength);
    }


    p_ref_count = new std::atomic<int>(1);
}

template <typename T, typename Container>
mdspan_data<T,Container> mdspan_data<T,Container>::copy(  ManagedDataBlockConfig alloc_config)
{

    int targetdev, sourcedev;
    bool useomptargetmemcpy = false;

    if(alloc_config.data_ondevice && this->dpconfig.data_is_devptr)
    {
        targetdev = alloc_config.default_device?omp_get_default_device(): alloc_config.devicenum;
        sourcedev = this->dpconfig.devicenum;
        useomptargetmemcpy = true;
    }
    else if(alloc_config.data_ondevice && !this->dpconfig.data_is_devptr)
    {
        targetdev =  alloc_config.default_device?omp_get_default_device(): alloc_config.devicenum;
        sourcedev = omp_get_initial_device();
        useomptargetmemcpy = true;
    }
    else if(!alloc_config.data_ondevice && this->dpconfig.data_is_devptr)
    {
        targetdev = omp_get_initial_device();
        sourcedev = this->dpconfig.devicenum;
        useomptargetmemcpy = true;
    }

    alloc_config.dpconjugate=this->dpconfig.dpconjugate;
    alloc_config.dprowmajor=this->dpconfig.dprowmajor;

    mdspan_data<T,Container> result(this->pextents, this->pstrides, alloc_config);

    if(useomptargetmemcpy)
        omp_target_memcpy(result.dpdata, this->dpdata, sizeof(T) * this->dpdatalength, 0, 0, targetdev, sourcedev);
    else
        memcpy(result.dpdata, this->dpdata, sizeof(T) * this->dpdatalength);

    return result;
}



template <typename T, typename Container>
mdspan_data<T,Container>& mdspan_data<T, Container>::operator=(const mdspan_data<T,Container>& other)
{
    if(this->dpdata != other.dpdata)
    {
        release_all_data();
        this->p_has_offloaded_host_data = false;
    }

    this->mapping_manager = other.mapping_manager;
    this->pextents = other.pextents;
    this->pstrides = other.pstrides;

    this->dpextents        = this->pextents.data();
    this->dpstrides        = this->pstrides.data();
    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;

    this->dprank           = other.dprank;
    this->dpconfig          =other.dpconfig;
    this->devptr_former_hostptr= other.devptr_former_hostptr;

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


    this->memmap = other.memmap;
    this->p_has_offloaded_host_data = false;


    if (p_ref_count)
    {
        p_ref_count->fetch_add(1, std::memory_order_relaxed);
    }
}



template<typename T, typename Container>
mdspan_data<T,Container>& mdspan_data<T, Container>::operator=(mdspan_data<T,Container>&& other) noexcept
{
    if (this != &other)
    {

        release_all_data();

        this->dpdata = other.dpdata;
        this->dpdatalength = other.dpdatalength;
        this->dprank = other.dprank;
        this->dpconfig = other.dpconfig;
        this->devptr_former_hostptr = other.devptr_former_hostptr;

        if constexpr (DynamicContainer<Container>)
        {
            this->pextents = std::move(other.pextents);
        }
        if constexpr (DynamicContainer<Container>)
        {
            this->pstrides = std::move(other.pstrides);
        }

        if constexpr (StaticContainer<Container>)
        {
            this->pextents = other.pextents;
        }
        if constexpr (StaticContainer<Container>)
        {
            this->pstrides = other.pstrides;
        }


        this->dpextents = this->pextents.data();
        this->dpstrides = this->pstrides.data();


        this->p_has_offloaded_host_data = other.p_has_offloaded_host_data;
        this->mapping_manager = std::move(other.mapping_manager);
        this->dpconfig.pmemmap = other.dpconfig.pmemmap;
        this->p_ref_count = other.p_ref_count;


        other.dpdata = nullptr;
        other.dpdatalength = 0;
        other.dprank = 0;
        other.devptr_former_hostptr = nullptr;

        other.dpconfig = DataBlockConfig{};
        other.p_has_offloaded_host_data = false;
        other.dpconfig.pmemmap = false;
        other.p_ref_count = nullptr;
    }
    return *this;
}


#endif

