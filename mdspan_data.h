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
    bool memmap=false;
    bool data_ondevice=false;
    bool default_device=false;
    int devicenum = -INT_MAX;
    bool dpconjugate=false;

inline DataBlockConfig Get_DataBlockConfig() const {
        return DataBlockConfig{
            .dprowmajor    = this->dprowmajor,
            .data_ondevice = this->data_ondevice,
            .devicenum     = this->devicenum,
            .dpconjugate   = this->dpconjugate
        };
}

 inline static ManagedDataBlockConfig SetConfig(bool pmemmap, bool defaultdevice, const DataBlockConfig& config) {
        return ManagedDataBlockConfig{
            .dprowmajor     = config.dprowmajor,
            .memmap         = pmemmap,
            .data_ondevice  = config.data_ondevice,
            .default_device = defaultdevice,
            .devicenum      = config.devicenum,
            .dpconjugate    = config.dpconjugate
        };
    }
};

template <typename T, typename Container>
class mdspan_data: public mdspan<T,Container>
{
public:

    friend class mdspan_utilities;

    mdspan_data() {};


    mdspan_data(size_t datalength, const Container& extents, const Container& strides,ManagedDataBlockConfig config);

    mdspan_data(const Container& extents, const Container& strides, ManagedDataBlockConfig config);

    mdspan_data(const Container& extents,ManagedDataBlockConfig config);

    mdspan_data( const DataBlock<T>& view, ManagedDataBlockConfig alloc_config) ;

    mdspan_data(const mdspan_data<T, Container>& other);
   mdspan_data<T, Container>&operator=(const mdspan_data<T,Container> & other);

    mdspan_data(mdspan_data<T, Container>&& other) noexcept;
      mdspan_data<T, Container>& operator=( mdspan_data<T, Container>&& other) noexcept;



    ~mdspan_data();



    using DataBlock<T>::operator=;

    mdspan_data<T, Container> copy( bool memmap=false, bool ondevice=false,bool defaultdevice=true,int devicenum=0);

    void release_all_data();
protected:
    std::atomic<int>* p_ref_count = nullptr;
    bool pmemmap=false;
    void initialization_helper(const ManagedDataBlockConfig& config);
};

template<typename T, typename Tag>
using mdspan_data_t = mdspan_data<T, typename container_for_tag<Tag>::type>;

template <typename T, typename Container>
void mdspan_data<T,Container>::initialization_helper(const ManagedDataBlockConfig& config)
{
    this->pmemmap = config.memmap;
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

        this->dpconfig.data_ondevice = false;
#else
        if (config.default_device)
        {
            target_device = omp_get_default_device();
        }
        this->dpdata = GPU_Memory_Functions::alloc_device_ptr<T>(this->dpdatalength, target_device);
        this->dpconfig.devicenum = target_device;
        this->dpconfig.data_ondevice = true;
        this->devptr_former_hostptr = nullptr;
#endif
    }
    else
    {
        if (config.memmap)
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength]();

        this->dpconfig.data_ondevice = false;
    }

    p_ref_count = new std::atomic<int>(1);
}


template <typename T, typename Container>
mdspan_data<T,Container>::mdspan_data(size_t datalength, const Container& extents, const Container& strides, ManagedDataBlockConfig config)
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



template <typename T, typename Container>
void mdspan_data<T, Container>::release_all_data()
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

            else if (this->dpconfig.data_ondevice)
            {
                if (this->dpdata != nullptr)
                {
                    GPU_Memory_Functions::free_device_ptr(this->dpdata, this->dpconfig.devicenum);
                    this->dpdata = nullptr;
                }
            }
            if (this->dpdata != nullptr)
            {
                if (this->pmemmap)
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

template <typename T, typename Container>
mdspan_data<T, Container>::~mdspan_data()
{
    release_all_data();
}

template <typename T, typename Container>
mdspan_data<T, Container>::mdspan_data( const DataBlock<T>& view,
                                        ManagedDataBlockConfig alloc_config)
{
    // 1. Synchronize basic memory configuration states
    this->dpconfig = alloc_config.Get_DataBlockConfig();
    this->pmemmap = alloc_config.memmap;
    this->p_has_offloaded_host_data = false;
    this->dpdatalength = view.datalength();
    this->dprank = view.rank();

    if constexpr (StaticContainer<Container>)
    {
        this->pextents = {};
        this->pstrides = {};
    }

    if constexpr (DynamicContainer<Container>) {
        this->pextents.resize(this->dprank);
        this->pstrides.resize(this->dprank);
    }

    std::copy(view.extents(), view.extents() + this->dprank, std::begin(this->pextents));
    std::copy(view.strides(), view.strides() + this->dprank, std::begin(this->pstrides));


    this->dpextents = this->pextents.data();
    this->dpstrides = this->pstrides.data();


    int target_device = alloc_config.devicenum;
    if (alloc_config.data_ondevice && alloc_config.default_device) {
        target_device = omp_get_default_device();
    }


    if (alloc_config.data_ondevice) {
#if defined(Unified_Shared_Memory)
        if (alloc_config.memmap)
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength]();
        this->dpconfig.data_ondevice = false;
#else
        this->dpdata = GPU_Memory_Functions::alloc_device_ptr<T>(this->dpdatalength, target_device);
        this->dpconfig.devicenum = target_device;
        this->dpconfig.data_ondevice = true;
        this->devptr_former_hostptr = nullptr;
#endif
    } else {
        if (alloc_config.memmap)
            this->dpdata = Host_Memory_Functions::create_temp_mmap<T>(this->dpdatalength);
        else
            this->dpdata = new T[this->dpdatalength]();
        this->dpconfig.data_ondevice = false;
    }


    if (this->dpconfig.data_ondevice) {
        int source_device = view.config().data_ondevice ? view.config().devicenum : omp_get_initial_device();
        omp_target_memcpy(
            this->dpdata,
            view.data(),
            sizeof(T) * this->dpdatalength,
            0, 0,
            target_device,
            source_device
        );
    } else {
        memcpy(this->dpdata, view.data(), sizeof(T) * this->dpdatalength);
    }


    p_ref_count = new std::atomic<int>(1);
}

template <typename T, typename Container>
mdspan_data<T, Container> mdspan_data<T, Container>::copy(bool memmap, bool ondevice, bool defaultdevice, int devicenum)
{
    if(defaultdevice)
        devicenum = omp_get_default_device();

    ManagedDataBlockConfig cfg = ManagedDataBlockConfig{
        .dprowmajor=this->dpconfig.dprowmajor,
        .memmap = memmap ,
                                                        .data_ondevice=ondevice,
                                                        .devicenum = devicenum,
                                                        .dpconjugate=this->dpconfig.dpconjugate};



    mdspan_data<T, Container> result(this->pextents, this->pstrides, cfg);

    int targetdev, sourcedev;
    bool useomptargetmemcpy = false;

    if(ondevice && this->dpconfig.data_ondevice)
    {
        targetdev = devicenum;
        sourcedev = this->dpconfig.devicenum;
        useomptargetmemcpy = true;
    }
    else if(ondevice && !this->dpconfig.data_ondevice)
    {
        targetdev = devicenum;
        sourcedev = omp_get_initial_device();
        useomptargetmemcpy = true;
    }
    else if(!ondevice && this->dpconfig.data_ondevice)
    {
        targetdev = omp_get_initial_device();
        sourcedev = this->dpconfig.devicenum;
        useomptargetmemcpy = true;
    }

    if(useomptargetmemcpy)
        omp_target_memcpy(result.dpdata, this->dpdata, sizeof(T) * this->dpdatalength, 0, 0, targetdev, sourcedev);
    else
        memcpy(result.dpdata, this->dpdata, sizeof(T) * this->dpdatalength);

    return result;
}



template <typename T, typename Container>
mdspan_data<T, Container>& mdspan_data<T, Container>::operator=(const mdspan_data<T,Container>& other)
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
mdspan_data<T, Container>::mdspan_data(const mdspan_data<T, Container>& other)
    : mdspan<T, Container>(other),
      p_ref_count(other.p_ref_count)
{

    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprank = other.dprank;
    this->dpconfig = other.dpconfig;
    this->devptr_former_hostptr = other.devptr_former_hostptr;


    this->pmemmap = other.pmemmap;
    this->p_has_offloaded_host_data = false;


    if (p_ref_count)
    {
        p_ref_count->fetch_add(1, std::memory_order_relaxed);
    }
}



template<typename T, typename Container>
mdspan_data<T, Container>& mdspan_data<T, Container>::operator=(mdspan_data<T, Container>&& other) noexcept
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
            this->pstrides = std::move(other.pstrides);
        }
        else
        {
            this->pextents = other.pextents;
            this->pstrides = other.pstrides;
        }


        this->dpextents = this->pextents.data();
        this->dpstrides = this->pstrides.data();


        this->p_has_offloaded_host_data = other.p_has_offloaded_host_data;
        this->mapping_manager = std::move(other.mapping_manager);
        this->pmemmap = other.pmemmap;
        this->p_ref_count = other.p_ref_count;


        other.dpdata = nullptr;
        other.dpdatalength = 0;
        other.dprank = 0;
        other.devptr_former_hostptr = nullptr;

        other.dpconfig = DataBlockConfig{};
        other.p_has_offloaded_host_data = false;
        other.pmemmap = false;
        other.p_ref_count = nullptr;
    }
    return *this;
}


#endif

