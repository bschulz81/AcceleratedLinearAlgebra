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
#include "datablock_gpu_memory_functions.h"

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


template <typename T, typename Container>
class mdspan:public DataBlock<T>
{

protected:




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
    void allocate_extents_and_strides(size_t r);
    void adopt_subDataBlock_helper(const DataBlock<T>& sub);

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


    mdspan(T* data, const size_t datalength, const Container& extents, const Container& strides,const bool rowm=true,bool dpdata_is_devptr=false,int devnum=0);


    mdspan(T* data,  const Container& extents, const Container& strides,const bool rowm=true, bool dpdata_is_devptr=false,int devnum=0);


    mdspan(T* data, const Container& extents, const bool rowm=true,bool dpdata_is_devptr=false,int devnum=0);


    mdspan(T* data, const size_t rows,const size_t cols,const bool rowm=true,bool dpdata_is_devptr=false,int devnum=0);
    mdspan(T* data,const size_t rows, const bool rowm=true,bool dpdata_is_devptr=false,int devnum=0);
    virtual ~mdspan();

    using DataBlock<T>::operator();
    inline T& operator()(const Container& extents);
    inline T operator()(const Container& extents)const;

    using DataBlock<T>::operator=;

    // Subspan methods
    using DataBlock<T>::subspan;
    mdspan<T, Container> subspan(const Container& offsets,  Container& sub_extents) const;
    using DataBlock<T>::subspanmatrix;
    mdspan<T, Container> subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols)const;
    using DataBlock<T>::column;
    mdspan<T, Container>column(const size_t col_index);

    using DataBlock<T>::row;
    mdspan<T, Container>row(const size_t row_index);

    using DataBlock<T>::transpose;
    mdspan<T, Container>transpose();

    using DataBlock<T>::collapsed_view;
    mdspan<T, std::vector<size_t>> collapsed_view();

    bool  device_data_upload(bool default_device,int devicenum=0);
    bool  device_data_alloc(bool default_device,int devicenum=0);
    bool  device_data_download_release();
    bool  device_data_release();
    bool  host_data_update();
    bool  device_data_update();

    size_t extent(const size_t dim) const
    {
        return this->dpextents[dim];
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

// Primary template (undefined on purpose)
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
        //does not get copied. Only set to true for upload.
        p_has_offloaded_host_data = false;
    }

    mapping_manager=other.mapping_manager;

    pextents = other.pextents;
    pstrides = other.pstrides;

    this->dpextents        = pextents.data();
    this->dpstrides        = pstrides.data();

    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;

    this->devptr_devicenum=other.devptr_devicenum;
    this->devptr_former_hostptr=other.devptr_former_hostptr;
    return *this;
}

template <typename T, typename Container>
mdspan<T, Container>&mdspan<T, Container>::operator=(const DataBlock<T> & other)
{

    if(this->dpdata!=other.dpdata)
    {
        if(p_has_offloaded_host_data)
            this->device_data_release();
        p_has_offloaded_host_data = false;
        this->devptr_devicenum=-1;
        this->devptr_former_hostptr=nullptr;
    }

    this->dpdata           = other.dpdata;
    this->dpdatalength      =other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dpdata_is_devptr = other.dpdata_is_devptr;

    this->dprank=other.dprank;


    if(pextents.size()!=other.dprank)
        if constexpr (DynamicContainer<Container>)
            pextents.resize(other.dprank);

    if(pextents.data()!=other.dpextents)
        copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

    if(pstrides.size()!=other.dprank)
        if constexpr (DynamicContainer<Container>)
            pstrides.resize(other.dprank);

    if(pextents.data()!=other.dpstrides)
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
    this->dpdatalength      =other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dpdata_is_devptr = other.dpdata_is_devptr;

    this->dprank=other.dprank;

    if constexpr (DynamicContainer<Container>)
    {
        pextents  = std::move(other.pextents);
        pstrides  = std::move(other.pstrides);
    }
    if constexpr (StaticContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

        if(pextents.data()!=other.dpstrides)
            copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));
    }
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();



    mapping_manager=std::move(other.mapping_manager);


    // Move other raw pointers and flags
    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength     = other.dpdatalength;

    p_has_offloaded_host_data  = other.p_has_offloaded_host_data;
    this->devptr_devicenum    = other.devptr_devicenum;
    this->devptr_former_hostptr = other.devptr_former_hostptr;


    other.p_has_offloaded_host_data = false;
    other.dpdata               = nullptr;
    other.dpstrides            = nullptr;
    other.dpextents            = nullptr;
    other.devptr_former_hostptr=nullptr;

    return *this;
}


template<typename T, typename Container>
mdspan<T, Container>::mdspan(const mdspan<T, Container>& other)
{
    // don't take ownership of device memory on copy
    p_has_offloaded_host_data = false;

    // shared mapping manager (shared_ptr copy)
    mapping_manager = other.mapping_manager;

    // copy extents/strides container contents
    if constexpr (DynamicContainer<Container>)
    {
        pextents = other.pextents;
        pstrides = other.pstrides;
    }
    if constexpr(StaticContainer<Container>)
    {
        // only copy actual rank elements
        if (pextents.data() != other.dpextents)
            std::copy(other.dpextents, other.dpextents + other.dprank, std::begin(pextents));
        if (pstrides.data() != other.dpstrides)
            std::copy(other.dpstrides, other.dpstrides + other.dprank, std::begin(pstrides));
    }


    // update raw pointers used by base DataBlock
    this->dpextents  = pextents.data();
    this->dpstrides  = pstrides.data();

    // copy the underlying pointer/metadata (shallow copy of host/device pointer)
    this->dpdata           = other.dpdata;
    this->dpdatalength     = other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->devptr_devicenum=other.devptr_devicenum;
    this->devptr_former_hostptr=other.devptr_former_hostptr;

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(mdspan<T, Container>&& other)noexcept
{

    this->dpdata           = other.dpdata;
    this->dpdatalength      =other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dpdata_is_devptr = other.dpdata_is_devptr;

    this->dprank=other.dprank;

    if constexpr (DynamicContainer<Container>)
    {
        pextents  = std::move(other.pextents);
        pstrides  = std::move(other.pstrides);
    }

    if constexpr (StaticContainer<Container>)
    {
        if(pextents.data()!=other.dpextents)
            copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

        if(pextents.data()!=other.dpstrides)
            copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));
    }

    mapping_manager=std::move(other.mapping_manager);

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


    // Move other raw pointers and flags
    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength     = other.dpdatalength;

    p_has_offloaded_host_data  = other.p_has_offloaded_host_data;
    this->devptr_devicenum    = other.devptr_devicenum;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;


    other.p_has_offloaded_host_data = false;
    other.dpdata               = nullptr;
    other.dpstrides            = nullptr;
    other.dpextents            = nullptr;
    other.devptr_former_hostptr=nullptr;



}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(const DataBlock<T>&other,const shared_ptr<mdspan<T,Container>::DevicemappingManager>&m )
{

    p_has_offloaded_host_data = false;

    this->dpdata           = other.dpdata;
    this->dpdatalength      =other.dpdatalength;
    this->dprowmajor       = other.dprowmajor;
    this->dpdata_is_devptr = other.dpdata_is_devptr;

    this->dprank=other.dprank;


    if(pextents.size()!=other.dprank)
        if constexpr (DynamicContainer<Container>)
            pextents.resize(other.dprank);

    if(pextents.data()!=other.dpextents)
        copy(other.dpextents,other.dpextents+other.dprank,begin(pextents));

    if(pstrides.size()!=other.dprank)
        if constexpr (DynamicContainer<Container>)
            pstrides.resize(other.dprank);

    if(pextents.data()!=other.dpstrides)
        copy(other.dpstrides,other.dpstrides+other.dprank,begin(pstrides));


    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


    mapping_manager=m;

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();


    // Move other raw pointers and flags
    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength     = other.dpdatalength;

    p_has_offloaded_host_data  = false;

    this->devptr_devicenum    = other.devptr_devicenum;
    this->devptr_former_hostptr  = other.devptr_former_hostptr;


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

    return this->dpdata[offset];
}


template <typename Container>
void compute_strides(const Container& extents, Container& strides,const bool rowmajor)
{
    const size_t n = extents.size();
    if (n == 0) return;

    if constexpr (StaticContainer<Container>)
    {
        strides = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        strides.resize(n); // Resize dynamic container
    }
    if(n==1)
    {
        strides[0]=1;
        return;
    }

    if (rowmajor)
    {

        // Row-major layout: last dimension has stride 1
        strides[n - 1] = 1;
        #pragma omp unroll
        for (int i =(int) n - 2; i > 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
        strides[0] = strides[1] * extents[1];
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
        #pragma omp unroll
        for (size_t i = 1; i < n; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
    }
}

template <typename T, typename Container>
void mdspan<T, Container>::initialize_extents_and_strides(const Container& extents, const Container& strides)
{
    const size_t r = extents.size();
    allocate_extents_and_strides(r);

    #pragma omp simd
    for (size_t i = 0; i < r; ++i)
    {
        pextents[i] = extents[i];
        pstrides[i] = strides[i];
    }
    // Assign to DataBlock
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();

}


template <typename T, typename Container>
void mdspan<T, Container>::allocate_extents_and_strides(size_t r)
{

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
    this->dpstrides=pstrides.data();
    this->dpextents=pextents.data();

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
    // Assign to DataBlock
    this->dpextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  size_t datalength, const Container& extents, const Container& strides,const  bool rowm,bool dpdata_is_devptr,int devnum)
    :DataBlock<T>(data,datalength,rowm,extents.size(),nullptr,nullptr,false,false,dpdata_is_devptr,devnum)
{
    initialize_extents_and_strides(extents,strides);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const Container& extents, const Container& strides,const bool rowm,bool dpdata_is_devptr,int devnum)
    : DataBlock<T>(data, 0,rowm,extents.size(),nullptr,nullptr,false,false,dpdata_is_devptr,devnum)

{
    initialize_extents_and_strides(extents,strides);
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  Container& extents,const bool rowm,bool dpdata_is_devptr,int devnum)
    :  DataBlock<T>(data,0,rowm,extents.size(),nullptr,nullptr,false,false,dpdata_is_devptr,devnum)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    this->dpstrides = pstrides.data();
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);
}







template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,  const size_t rows, const size_t cols,const bool rowm,bool dpdata_is_devptr,int devnum)
    :  DataBlock<T>(data,0,rowm,2,nullptr,nullptr,false,false,dpdata_is_devptr,devnum)
{

    const size_t r=2;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }
    // Resize and copy extents from container

    pextents[0]=rows;
    pextents[1]=cols;
    compute_strides(pextents,pstrides,rowm);

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,  const size_t rows,const bool rowm,bool dpdata_is_devptr,int devnum)
    :  DataBlock<T>(data,0,rowm,1,nullptr,nullptr,false,false,dpdata_is_devptr,devnum)
{
    const size_t r=1;
    if constexpr (StaticContainer<Container>)
    {
        pextents = {}; // Default-initialize static container
    }

    if constexpr (DynamicContainer<Container>)
    {
        pextents.resize(r); // Resize dynamic container
    }
    // Resize and copy extents from container

    pextents[0]=rows;
    pstrides[0]=1;


    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();
    this->dpdatalength=rows;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_upload(bool default_device,int devicenum)
{

    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    if(this->dpdata_is_devptr && devicenum==this->devptr_devicenum )return false;

    if(mapping_manager==nullptr)
    {
        mapping_manager = std::make_shared<DevicemappingManager>();
    }


    if(!mapping_manager->insert(devicenum, (intptr_t)this->dpdata, (intptr_t)(this->dpdata+this->dpdatalength)))return false;

    DataBlock_GPU_Memory_Functions<T>::copy_data_to_device_set_devptr(*this,devicenum);

    p_has_offloaded_host_data=true;
    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_alloc(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    if(this->dpdata_is_devptr && devicenum==this->devptr_devicenum )return false;

    if(mapping_manager==nullptr)
        mapping_manager = std::make_shared<DevicemappingManager>();

    if(!mapping_manager->insert(devicenum, (intptr_t)this->dpdata, (intptr_t)(this->dpdata+this->dpdatalength)))return false;

    DataBlock_GPU_Memory_Functions<T>::alloc_data_to_device_set_devptr(*this,devicenum);
    p_has_offloaded_host_data=true;

    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_download_release()
{

    if(!p_has_offloaded_host_data)return false;
    if(mapping_manager==nullptr) return false;
    if(!mapping_manager->remove(this->devptr_devicenum, (intptr_t)this->devptr_former_hostptr, (intptr_t)(this->devptr_former_hostptr+this->dpdatalength)))
        return false;

    DataBlock_GPU_Memory_Functions<T>::copy_data_to_host_set_host_ptr(*this);
    p_has_offloaded_host_data=false;

    return true;
}





template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_release()
{
    if(!p_has_offloaded_host_data)return false;
    if(mapping_manager==nullptr) return false;

    if(!mapping_manager->remove(this->devptr_devicenum, (intptr_t)this->devptr_former_hostptr, (intptr_t)(this->devptr_former_hostptr+this->dpdatalength)))
        return false;

    DataBlock_GPU_Memory_Functions<T>::free_device_data_set_host_ptr(*this);
    p_has_offloaded_host_data=false;
    return true;

}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: host_data_update()
{
    if(!this->dpdata_is_devptr)return false;
    if(this->devptr_former_hostptr==nullptr)return false;

    DataBlock_GPU_Memory_Functions<T>::copy_data_to_host_ptr(*this);
    return true;

}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_data_update()
{
    if(!this->dpdata_is_devptr)return false;
    if(this->devptr_former_hostptr==nullptr)return false;

    DataBlock_GPU_Memory_Functions<T>::copy_data_to_device_ptr(*this);
    return true;

}

template <typename T, typename Container>
mdspan<T,std::vector<size_t>>  mdspan<T, Container>::collapsed_view()
{
    size_t num_dims = this->count_noncollapsed_dims();
    size_t *tempext=new size_t[num_dims],
           *tempstr=new size_t[num_dims];
    mdspan<T, std::vector<size_t>> result(this->collapsed_view(num_dims,tempext, tempstr),mapping_manager);
    delete []tempext;
    delete []tempstr;
    return result;

}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan(const Container&offsets,  Container &sub_extents)const
{
    size_t *tempstr=new size_t[offsets.size()];
    size_t *tempext=new size_t[offsets.size()];
    mdspan<T,Container> result( this->subspan(offsets.data(),sub_extents.data(),tempext, tempstr),mapping_manager);
    delete [] tempstr;
    delete [] tempext;
    return result;
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols )const
{
    size_t tempext[2], tempstr[2];
    mdspan<T,Container> result(this->subspanmatrix(row,col,tile_rows,tile_cols, tempext, tempstr),mapping_manager);
    return result;
}

template <typename T, typename Container>
mdspan<T,Container> mdspan<T, Container>:: row(const size_t row_index)
{
    size_t tempext[1], tempstr[1];
    mdspan<T,Container> result(this->row(row_index,tempext, tempstr),mapping_manager);
    return result;
}

template <typename T, typename Container>
mdspan<T,Container> mdspan<T, Container>::column(const size_t column_index)
{
    size_t tempext[1], tempstr[1];
    mdspan<T,Container> result(this->column(column_index,tempext, tempstr),mapping_manager);
    return result;
}


template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>::transpose()
{
    size_t tempext[2], tempstr[2];
    mdspan<T,Container> result(transpose(tempext,tempstr),mapping_manager);
    return result;
}




#endif
