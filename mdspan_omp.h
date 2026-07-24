#ifndef MDSPANH
#define MDSPANH

#include <iostream>
#include <array>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <numbers>
#include <memory>

#include <cassert>

#include "datablock.h"

#include <array>
#include <vector>
#include <cstddef>

#include <unordered_map>
#include <set>

#include "datablock.h"
#include "gpu_memory_functions.h"

using namespace std;


// Concept definitions
template <typename Container>
concept StaticContainer =
    requires(Container c, size_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<size_t>;
    {
        c[i]
    }
    -> std::convertible_to<typename Container::value_type>;
    (!requires(Container c, size_t i)
    {
        c.reserve(i);
    });
};

template <typename Container>
concept DynamicContainer =
    requires(Container c, size_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<size_t>;
    {
        c[i]
    }
    -> std::convertible_to<typename Container::value_type>;
    c.reserve(i);  // Require reserve() for dynamic containers
};




// Concept to check if two containers are of the same type and have matching size
template <typename ExtentsContainer>
concept Container =
    (StaticContainer<ExtentsContainer>   ||  // Same size for static containers
     (DynamicContainer<ExtentsContainer>));  // Same size for dynamic containers
// Class template for mdspan


class mdspan_utilities;

template <typename T, typename Container>
class mdspan:public DataBlock<T>
{

protected:

    friend class mdspan_utilities;


    class DevicemappingManager
    {
    protected:
        struct Interval
        {
            intptr_t start;
            intptr_t end;

            bool operator<(const Interval& other) const
            {
                return start < other.start;
            }
        };
        std::unordered_map<int, std::set<Interval>> device_intervals;

        bool overlaps(const Interval& a, const Interval& b) const
        {
            return a.start < b.end && b.start < a.end;
        }

    public:
        bool insert(int device,  intptr_t start, intptr_t end)
        {
            Interval new_iv{start, end};
            auto& s = device_intervals[device];

            auto it = s.lower_bound(new_iv);

            if (it != s.end() && overlaps(new_iv, *it)) return false;


            if (it != s.begin() && overlaps(new_iv, *std::prev(it))) return false;

            s.insert(it, new_iv);
            return true;
        }

        // Remove interval
        bool remove(int device, intptr_t start, intptr_t end)
        {
            auto it = device_intervals.find(device);
            if (it != device_intervals.end())
            {
                Interval iv{start, end};
                size_t erased = it->second.erase(iv);

                if (erased == 0) return false;

                if (it->second.empty()) device_intervals.erase(it);
                return true;
            }
            else
                return false;
        }
        void showmapped() const
        {
            for (const auto& [device, intervals] : device_intervals)
            {
                std::cout << "Device " << device << ": ";
                for (const auto& iv : intervals)
                    std::cout << "[" << iv.start << "," << iv.end << ") ";
                std::cout << "\n";
            }
        }

    };

    void initialize_extents_and_strides(const Container&extents,const Container & strides);
    void initialize_extents(const Container&extents);
    void compute_initialize_strides(const Container& extents,const bool rowmajor);

    Container pextents;
    Container pstrides;
    shared_ptr<DevicemappingManager> mapping_manager=make_shared<DevicemappingManager>();

    bool p_has_offloaded_host_data=false;

public:


    mdspan() {};

    mdspan(const DataBlock<T>& ds,const shared_ptr<mdspan<T,Container>::DevicemappingManager> &dev);

    mdspan(const mdspan<T, Container>& other);
    mdspan(mdspan<T, Container>&& other)noexcept;
    mdspan<T, Container> &operator=(const mdspan<T,Container> & other);
    mdspan<T, Container> &operator=(const DataBlock<T> & other);
    mdspan<T, Container> &operator=(mdspan<T, Container>&& other)noexcept;


    mdspan(T* data, const size_t datalength, const Container& extents, const Container& strides, const DataBlockConfig  config);
    mdspan(T* data, const Container& extents, const Container& strides,const DataBlockConfig  config);
    mdspan(T* data, const Container& extents,const DataBlockConfig  config);


    virtual ~mdspan();

    using DataBlock<T>::operator();
    inline T& operator()(const Container& extents);
    inline T operator()(const Container& extents)const;

    using DataBlock<T>::operator=;

    bool  device_data_upload(bool default_device,int devicenum=0);
    bool  device_data_alloc(bool default_device,int devicenum=0);
    bool  device_data_download_release();
    bool  device_data_release();
    bool  host_data_update();
    bool  device_data_update();

    size_t extent(const size_t dim) const
    {
        return pextents[dim];
    };
    size_t rank() const
    {
        return this->dprank;
    };
    size_t stride(const size_t dim) const
    {
        return pstrides[dim];
    };

    // Member function declarations
    const Container& extents()const
    {
        return pextents;
    };
    const Container& strides()const
    {
        return pstrides;
    };

    size_t datalength() const
    {
        return this->dpdatalength;
    };


};


struct dynamic_tag {};

template<size_t Rank>
struct static_tag {};

template<typename Tag>
struct container_for_tag;

// Specialization for dynamic
template<>
struct container_for_tag<dynamic_tag>
{
    using type = std::vector<size_t>;
};

// Specialization for static
template<size_t Rank>
struct container_for_tag<static_tag<Rank>>
{
    using type = std::array<size_t, Rank>;
};

// Alias template
template<typename T, typename Tag>
using mdspan_t = mdspan<T, typename container_for_tag<Tag>::type>;





template <typename T, typename Container>
mdspan<T,Container>& mdspan<T, Container>:: operator=(const mdspan<T,Container> & other)
{
    if(this->dpdata!=other.dpdata)
    {
        if(p_has_offloaded_host_data)
            this->device_data_release();
        p_has_offloaded_host_data = false;
    }
    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprank           = other.dprank;
    this->dpconfig         = other.dpconfig;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;


    mapping_manager=other.mapping_manager;

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
        if(p_has_offloaded_host_data)
            this->device_data_release();
    }

    this->dpdata           = other.dpdata;
    this->dpdatalength      =other.dpdatalength;
    this->dprank            =other.dprank;
    this->dpconfig         =other.dpconfig;
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


template<typename T, typename Container>
mdspan<T, Container>& mdspan<T, Container>::operator=( mdspan<T, Container>&& other)noexcept
{
    if(this->dpdata!=other.dpdata)
    {
        if(p_has_offloaded_host_data)
            this->device_data_release();
    }


    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprank           = other.dprank;
    this->dpconfig         = other.dpconfig;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;


    if constexpr (DynamicContainer<Container>)
    {
        pextents  = std::move(other.pextents);
        pstrides  = std::move(other.pstrides);
    }
    if constexpr (StaticContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

        if(pstrides.data()!=other.dpstrides)
            copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));
    }
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();



    mapping_manager=std::move(other.mapping_manager);

    p_has_offloaded_host_data  = other.p_has_offloaded_host_data;
    other.p_has_offloaded_host_data = false;


    other.dpdata               = nullptr;
    other.dpstrides            = nullptr;
    other.dpextents            = nullptr;
    other.devptr_former_hostptr=nullptr;
    other.dpconfig.dprowmajor=false;
    other.dpconfig.data_ondevice=false;
    other.dpconfig.devicenum = -INT_MAX;
    other.dpconfig.dpconjugate=false;
    return *this;
}

template<typename T, typename Container>
mdspan<T, Container>::mdspan(const mdspan<T, Container>& other) {
    p_has_offloaded_host_data = false;
    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprank = other.dprank;
    this->dpconfig = other.dpconfig;
    this->devptr_former_hostptr = other.devptr_former_hostptr;
    this->mapping_manager = other.mapping_manager;


    if constexpr (DynamicContainer<Container>) {
        pextents = other.pextents;
        pstrides = other.pstrides;
    }
    if constexpr (StaticContainer<Container>) {
        if (pextents.data() != other.dpextents) std::copy(other.dpextents, other.dpextents + other.dprank, std::begin(pextents));
        if (pstrides.data() != other.dpstrides) std::copy(other.dpstrides, other.dpstrides + other.dprank, std::begin(pstrides));
    }

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(const DataBlock<T>& other, const shared_ptr<typename mdspan<T,Container>::DevicemappingManager>& m) {
    p_has_offloaded_host_data = false;
    this->dpdata = other.dpdata;
    this->dpdatalength = other.dpdatalength;
    this->dprank = other.dprank;
    this->dpconfig = other.dpconfig;
    this->devptr_former_hostptr = other.devptr_former_hostptr;
    this->mapping_manager = m;


    if constexpr (DynamicContainer<Container>) {
        if (pextents.size() != other.dprank) pextents.resize(other.dprank);
        if (pstrides.size() != other.dprank) pstrides.resize(other.dprank);
    }

    if (pextents.data() != other.dpextents) {
        std::copy(other.dpextents, other.dpextents + other.dprank, std::begin(pextents));
    }
    if (pstrides.data() != other.dpstrides) {
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

    if constexpr (DynamicContainer<Container>)
    {
        pextents  = std::move(other.pextents);
        pstrides  = std::move(other.pstrides);
    }

    if constexpr (StaticContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

        if(pstrides.data()!=other.dpstrides)
            copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));
    }

    mapping_manager=std::move(other.mapping_manager);

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


    p_has_offloaded_host_data  = other.p_has_offloaded_host_data;

    other.p_has_offloaded_host_data = false;
    other.dpdata               = nullptr;
    other.dpstrides            = nullptr;
    other.dpextents            = nullptr;
    other.devptr_former_hostptr=nullptr;
    other.dpconfig.dprowmajor=false;
    other.dpconfig.data_ondevice=false;
    other.dpconfig.devicenum = -INT_MAX;
    other.dpconfig.dpconjugate=false;


}



template <typename T, typename Container>
mdspan<T, Container>::~mdspan()
{
    if(p_has_offloaded_host_data)
        this->device_data_release();
}


// Access operator for multidimensional indices
template <typename T, typename Container>
inline T& mdspan<T, Container>::operator()(const Container& indices)
{


    size_t offset = 0;
    #pragma omp simd reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * this->dpstrides[i];
    }
    return this->dpdata[offset];
}


// Access operator for multidimensional indices
template <typename T, typename Container>
T mdspan<T, Container>::operator()(const Container& indices)const
{

    size_t offset = 0;
    #pragma omp simd reduction( + : offset)
    for (size_t i = 0; i < indices.size(); ++i)
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
    const size_t n = extents.size();
    if (n == 0) return;

    if constexpr (StaticContainer<Container>)
    {
        pstrides = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pstrides.resize(n); // Resize dynamic container
    }
    if(n==1)
    {
        pstrides[0]=1;
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
        for (size_t i = 1; i < n; ++i)
        {
            pstrides[i] = pstrides[i - 1] * extents[i - 1];
        }
    }
    this->dpstrides = pstrides.data();

}

template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents_and_strides(const Container& extents, const Container& strides)
{
    const size_t r = extents.size();

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


    #pragma omp simd
    for (size_t i = 0; i < r; ++i)
    {
        pextents[i] = extents[i];
        pstrides[i] = strides[i];
    }
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


}


template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents(const Container& extents)
{
    const size_t r = extents.size();
    if constexpr (StaticContainer<Container>)
    {
        pextents = {};
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r);

    }
    #pragma omp simd
    for (size_t i = 0; i < r; ++i)
    {
        pextents[i] = extents[i];
    }
    this->dpextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  size_t datalength, const Container& extents, const Container& strides,const DataBlockConfig  config)
    :DataBlock<T>(data,datalength,extents.size(),nullptr,nullptr, config)
{
    initialize_extents_and_strides(extents,strides);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const Container& extents, const Container& strides,const DataBlockConfig  config)
    : DataBlock<T>(data, 0,extents.size(),nullptr,nullptr,config)
{
    initialize_extents_and_strides(extents,strides);
    this->dpdatalength=compute_data_length(this->dpextents,this->dpstrides,this->dprank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  Container& extents,const DataBlockConfig  config)
    :  DataBlock<T>(data,0,extents.size(),nullptr,nullptr,  config)
{
    initialize_extents(extents);
    compute_initialize_strides(pextents,config.dprowmajor);
    this->dpdatalength=compute_data_length(this->dpextents,this->dpstrides,this->dprank);
}




template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_upload(bool default_device,int devicenum)
{

    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    if(this->dpconfig.data_ondevice && devicenum==this->dpconfig.devicenum )return false;

    if(mapping_manager==nullptr)
    {
        mapping_manager = std::make_shared<DevicemappingManager>();
    }


    if(!mapping_manager->insert(devicenum, (intptr_t)this->dpdata, (intptr_t)(this->dpdata+this->dpdatalength)))return false;

    GPU_Memory_Functions::copy_data_to_device_set_devptr(*this,devicenum);

    p_has_offloaded_host_data=true;
    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_alloc(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    if(this->dpconfig.data_ondevice && devicenum==this->dpconfig.devicenum)return false;

    if(mapping_manager==nullptr)
        mapping_manager = std::make_shared<DevicemappingManager>();

    if(!mapping_manager->insert(devicenum, (intptr_t)this->dpdata, (intptr_t)(this->dpdata+this->dpdatalength)))return false;

    GPU_Memory_Functions::alloc_data_to_device_set_devptr(*this,devicenum);
    p_has_offloaded_host_data=true;

    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_download_release()
{

    if(!p_has_offloaded_host_data)return false;
    if(mapping_manager==nullptr) return false;
    if(!mapping_manager->remove(this->dpconfig.devicenum, (intptr_t)this->devptr_former_hostptr, (intptr_t)(this->devptr_former_hostptr+this->dpdatalength)))
        return false;

    GPU_Memory_Functions::copy_data_to_host_set_host_ptr(*this);
    p_has_offloaded_host_data=false;

    return true;
}





template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_release()
{
    if(!p_has_offloaded_host_data)return false;
    if(mapping_manager==nullptr) return false;

    if(!mapping_manager->remove(this->dpconfig.devicenum, (intptr_t)this->devptr_former_hostptr, (intptr_t)(this->devptr_former_hostptr+this->dpdatalength)))
        return false;

    GPU_Memory_Functions::free_device_data_set_host_ptr(*this);
    p_has_offloaded_host_data=false;
    return true;

}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: host_data_update()
{
    if(!this->dpconfig.data_ondevice)return false;
    if(this->devptr_former_hostptr==nullptr)return false;

    GPU_Memory_Functions::copy_data_to_host_ptr(*this);
    return true;

}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_update()
{
    if(!this->dpconfig.data_ondevice)return false;
    if(this->devptr_former_hostptr==nullptr)return false;

    GPU_Memory_Functions::copy_data_to_device_ptr(*this);
    return true;

}






#endif
