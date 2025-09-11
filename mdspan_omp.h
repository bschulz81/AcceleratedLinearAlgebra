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

#include "datastruct.h"

#include <array>
#include <vector>
#include <cstddef>





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

template<typename T>
class Datastruct_Functions;

template<typename T>
class Datastruct_MPI_Functions;

template <typename T, typename Container>
class mdspan:public datastruct<T>
{
public:

    friend class Datastruct_Functions<T>;
    friend class Datastruct_MPI_Functions<T>;
    mdspan() {};
    mdspan(const datastruct<T>& ds, Container&& ext, Container&& str);

    mdspan(const mdspan& other);

    mdspan(mdspan&& other) noexcept ;
    mdspan<T, Container>& operator=(mdspan<T, Container>&& other) noexcept;
    mdspan<T,Container>& operator=(const mdspan<T,Container> & other);
    mdspan<T, Container>&operator=(const datastruct<T> & other);

    mdspan(T* __restrict data, const size_t datalength,const bool rowm, const Container& extents, const Container& strides, bool p_is_devicedata_owner=true);
    mdspan(T* __restrict data, const bool rowm, const Container& extents, const Container& strides,bool p_is_devicedata_owner=true);
    mdspan(T* __restrict data, const bool rowm, const Container& extents,bool p_is_devicedata_owner=true);
    mdspan(T* __restrict data, const bool rowm,const size_t rows,const size_t cols,bool p_is_devicedata_owner=true);
    mdspan(size_t r,bool p_is_devicedata_owner=true);
    mdspan(size_t rank,Container ext,bool p_is_devicedata_owner=true);
    ~mdspan();
    // Access operators
    using datastruct<T>::operator();
    inline T& operator()(const Container& extents);
    inline T operator()(const Container& extents)const;




    // Subspan methods
    mdspan<T, Container> subspan(const Container& offsets,  Container& sub_extents) const;
    mdspan<T, Container> subspan(const Container& offsets,  Container& sub_extents, T* __restrict sub_data) const;

    using datastruct<T>::subspanmatrix;
    mdspan<T, Container> subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols)const;
    mdspan<T, Container> subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T* __restrict sub_data)const;

    mdspan<T, Container> copy(T*__restrict  data);


    using datastruct<T>::column;
    using datastruct<T>::row;
    using datastruct<T>::transpose;

    mdspan<T, std::vector<size_t>>column(const size_t col_index);
    mdspan<T, std::vector<size_t>>column(const size_t col_index,T* __restrict ptr);
    mdspan<T, std::vector<size_t>>row(const size_t row_index);
    mdspan<T, std::vector<size_t>>row(const size_t row_index,T* __restrict ptr);

    mdspan<T, Container>transpose();
    mdspan<T, Container>transpose(T* __restrict ptr);
    using datastruct<T>::collapsed_view;

    mdspan<T, std::vector<size_t>> collapsed_view();

    bool device_update(bool default_device=true, int devicenum=0);
    bool device_update_async(bool default_device=true, int devicenum=0);
    bool host_update(bool default_device=true, int devicenum=0);
    bool host_update_async(bool default_device=true, int devicenum=0);
    bool device_download_release(bool default_device=true, int devicenum=0);
    bool device_upload(bool default_device=true,int devicenum=0);
    bool device_alloc(bool default_device=true,int devicenum=0);
    bool device_release(bool default_device=true,int devicenum=0);
    bool device_delete(bool default_device=true,int devicenum=0);
    void release_all_owned_devices();
    void become_device_dataowner();

    // Other utility methods

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

protected:
    // Private member variables

    void initialize_extents_and_strides(const Container&extents,const Container & strides);
    void initialize_extents(const Container&extents);
    void allocate_extents_and_strides(size_t r);
    void adopt_subdatastruct_helper(const datastruct<T>& sub);

    Container pextents;  // Use the ExtentContainer type
    Container pstrides;  // Use the StrideContainer type
    unordered_map<int,bool> pis_offloaded;
private:
    bool is_device_data_owner = false;
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
mdspan<T, Container>::
mdspan(const mdspan& other) :
    datastruct<T>(other)
{
    pextents = other.pextents;
    pstrides = other.pstrides;
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();
    is_device_data_owner = false;
    pis_offloaded.clear();
}


template <typename T, typename Container>
mdspan<T,Container>& mdspan<T, Container>:: operator=(const mdspan<T,Container> & other)
{
    if(is_device_data_owner)
        this->release_all_owned_devices();

    pextents = other.pextents;
    pstrides = other.pstrides;

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();

    is_device_data_owner = false;
    pis_offloaded    = other.pis_offloaded;


    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength=other.dpdatalength;

    return *this;
}

template <typename T, typename Container>
mdspan<T, Container>&mdspan<T, Container>::operator=(const datastruct<T> & other)
{

    if(this->dpdata!=other.dpdata)
        this->release_all_owned_devices();

    pextents = other.pextents;
    pstrides = other.pstrides;

    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();

    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength=other.dpdatalength;

    return *this;
}


template<typename T, typename Container>
mdspan<T, Container>& mdspan<T, Container>::operator=(mdspan<T, Container>&& other) noexcept
{
    if (is_device_data_owner)
    {
        release_all_owned_devices();
    }

    // Move host containers
    pextents  = std::move(other.pextents);
    pstrides  = std::move(other.pstrides);

    // Update raw pointers
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();

    // Move other raw pointers and flags
    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength=other.dpdatalength;

    pis_offloaded    = std::move(other.pis_offloaded);

    // Transfer ownership
    is_device_data_owner       = other.is_device_data_owner;
    other.is_device_data_owner = false;
    other.dpdata               = nullptr;

    return *this;
}
template <typename T, typename Container>
mdspan<T, Container>::mdspan(mdspan<T, Container>&& other) noexcept{
    // Move host containers
    pextents = std::move(other.pextents);
    pstrides = std::move(other.pstrides);

    // Update raw pointers
    this->dpextents = pextents.data();
    this->dpstrides = pstrides.data();

    // Move data pointers and flags
    this->dpdata           = other.dpdata;
    this->dprowmajor       = other.dprowmajor;
    this->dprank           = other.dprank;
    this->dpdata_is_devptr = other.dpdata_is_devptr;
    this->dpdatalength=other.dpdatalength;
    pis_offloaded    = std::move(other.pis_offloaded);

    // Transfer ownership
    is_device_data_owner       = other.is_device_data_owner;
    other.is_device_data_owner = false;

    // Reset "other" so it can be destroyed safely
    other.dpdata     = nullptr;
    other.dpextents  = nullptr;
    other.dpstrides  = nullptr;
}

template <typename T, typename Container>
inline void mdspan<T, Container>::release_all_owned_devices()
{
    if(is_device_data_owner)
    {


        for(auto& p : this->pis_offloaded)
        {
            if(p.second==true)
            {
                Datastruct_GPU_Memory_Functions<T>::exit_struct(*this,p.first);
                p.second=false;
            }
        }
        is_device_data_owner=false;
    }
}

template <typename T, typename Container>
mdspan<T, Container>::~mdspan()
{
    if(is_device_data_owner)
        this->release_all_owned_devices();
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

    if (rowmajor)
    {
        // Row-major layout: last dimension has stride 1
        strides[n - 1] = 1;
        #pragma omp unroll
        for (int i = n - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
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
    // Assign to datastruct
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
    // Assign to datastruct
    this->dpextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  size_t datalength,const  bool rowm, const Container& extents, const Container& strides,bool pis_device_data_owner)
    :datastruct<T>(data,datalength,rowm,extents.size(),nullptr,nullptr,false,false,false)
{
    initialize_extents_and_strides(extents,strides);
    is_device_data_owner = pis_device_data_owner;
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm, const Container& extents, const Container& strides,bool pis_device_data_owner )
    : datastruct<T>(data, 0,rowm,extents.size(),nullptr,nullptr,false,false,false)
      // Initialize pdatastruct with placeholders
{
    initialize_extents_and_strides(extents,strides);
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);
    is_device_data_owner = pis_device_data_owner;
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const bool rowm,const  Container& extents,bool pis_device_data_owner)
    :  datastruct<T>(data,0,rowm,extents.size(),nullptr,nullptr,false,false,false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    this->dpstrides = pstrides.data();
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);
    is_device_data_owner = pis_device_data_owner;
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(size_t rank,bool pis_device_owner)
    :  datastruct<T>(nullptr,0,false,rank,nullptr,nullptr,false,false,false)
{
    allocate_extents_and_strides(rank);
    is_device_data_owner=pis_device_owner;
}

template <typename T, typename Container>
mdspan<T, Container>::mdspan(size_t rank,Container ext,bool pis_device_owner)
    :  datastruct<T>(nullptr,0,false,rank,nullptr,nullptr,false,false,false)
{
    allocate_extents_and_strides(rank);
    #pragma omp simd
    for (size_t i=0; i<rank; i++)
    {
        pextents[i]=ext[i];
    }
    is_device_data_owner=pis_device_owner;
}





template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  const size_t rows, const size_t cols,bool pis_device_data_owner)
    :  datastruct<T>(data,0,rowm,2,nullptr,nullptr,false,false,false)
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
    is_device_data_owner =  pis_device_data_owner;
}




template<typename T, typename Container>
void mdspan<T, Container>::become_device_dataowner()
{
    is_device_data_owner = true;
}




template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_upload(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;

    this->pis_offloaded.try_emplace(devicenum, true);

    Datastruct_GPU_Memory_Functions<T>::create_in_struct(*this,devicenum);
    return true;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_alloc(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();

    if(devicenum>=omp_get_num_devices()) return false;
    this->pis_offloaded.try_emplace(devicenum, true);
    Datastruct_GPU_Memory_Functions<T>::create_out_struct(*this,devicenum);
    return true;
}




template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_download_release(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    Datastruct_GPU_Memory_Functions<T>::update_host(this,devicenum);
    Datastruct_GPU_Memory_Functions<T>::release_struct(this,devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_delete(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    Datastruct_GPU_Memory_Functions<T>::exit_struct(*this,devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_release(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    Datastruct_GPU_Memory_Functions<T>::release_struct(*this,devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: host_update(bool default_device,int devicenum)
{
    if (default_device)  devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;
    if(this->pis_offloaded.contains(devicenum)==false) return false;
    Datastruct_GPU_Memory_Functions<T>::update_host(*this,devicenum);
    return true;

}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_update(bool default_device,int devicenum)
{

    if (default_device)  devicenum=omp_get_default_device();
    if(devicenum>=omp_get_num_devices()) return false;
    if(this->pis_offloaded.contains(devicenum)==false) return false;

    Datastruct_GPU_Memory_Functions<T>::update_device(*this,devicenum);
    return true;

}


template<typename T, typename Container>
void mdspan<T, Container>::adopt_subdatastruct_helper(const datastruct<T>& sub)
{
    this->dpdata =      sub.dpdata;
    this->dprowmajor = sub.dprowmajor;
    this->dprank = sub.dprank;
    this->dpdata_is_devptr = sub.dpdata_is_devptr;

    this->dpdatalength=sub.dpdatalength;
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::copy(T*__restrict  data)
{
    memcpy(data, this->dpdata,sizeof(T)*this->pdatalength);
    mdspan<T, Container>  md= mdspan(data,this->pdrowmajor,this->pextents, this->pstrides );
    md.pis_offloaded=this->pis_offloaded;
    for(auto& p : this->pis_offloaded)
    {
        if(p.second==true)
        {
            Datastruct_Functions<T>::create_in_struct(*md,p.first);
        }
    }
    return md;
}




template <typename T, typename Container>
mdspan<T,std::vector<size_t>>  mdspan<T, Container>::collapsed_view()
{
    size_t num_dims = this->count_noncollapsed_dims();
    mdspan<T, std::vector<size_t>> result(num_dims,false);
    datastruct<T> sub=this->collapsed_view(num_dims,result.dpextents, result.dpstrides);
    result.dpdata =      sub.dpdata;
    result.dprowmajor = sub.dprowmajor;
    result.dprank = sub.dprank;
    result.dpdata_is_devptr = sub.dpdata_is_devptr;
    result.dpdatalength=sub.dpdatalength;
    return result;

}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan(const Container&offsets,  Container &sub_extents)const
{
    mdspan<T,Container> result(this->dprank,sub_extents,false);

    datastruct<T> sub = this->subspan_v(offsets.data(),result.dpextents, result.dpstrides);
    result.adopt_subdatastruct_helper(sub);
    return result;
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan(const Container&offsets,  Container &sub_extents,T* __restrict sub_data )const
{
    mdspan<T,Container> result(this->dprank,sub_extents,true);
    datastruct<T> sub = this->subspan_v(offsets.data(),result.dpextents, result.dpstrides, sub_data);
    result.adopt_subdatastruct_helper(sub);
    return result;
}




template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols )const
{
    mdspan<T,Container> result(this->dprank,false);
    datastruct<T> sub = this->subspanmatrix(row,col,tile_rows,tile_cols, result.dpextents, result.dpstrides);
    result.adopt_subdatastruct_helper(sub);
    return result;
}

template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspanmatrix(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*__restrict  sub_data)const
{
    mdspan<T,Container> result(this->dprank,true);
    datastruct<T> sub = this->subspanmatrix_v(row,col,tile_rows,tile_cols,result.dpextents, result.dpstrides,sub_data);
    result.adopt_subdatastruct_helper(sub);
    return  result;
}




template <typename T, typename Container>
mdspan<T,std::vector<size_t>> mdspan<T, Container>::column(const size_t col_index)
{
    mdspan<T,std::vector<size_t>> result(1,false);
    datastruct<T> sub = column(col_index,result.pextents.data(), result.pstrides.data());
    result.dpdata =      sub.dpdata;
    result.dprowmajor = sub.dprowmajor;
    result.dprank = sub.dprank;
    result.dpdata_is_devptr = sub.dpdata_is_devptr;
    result.dpdatalength=sub.dpdatalength;
    return result;
}

template <typename T, typename Container>

mdspan<T,std::vector<size_t>> mdspan<T, Container>:: row(const size_t row_index)
{
    mdspan<T,std::vector<size_t>> result(1,false);
    datastruct<T> sub = row(row_index,result.dpextents, result.dpstrides);
    result.dpdata =      sub.dpdata;
    result.dprowmajor = sub.dprowmajor;
    result.dprank = sub.dprank;
    result.dpdata_is_devptr = sub.dpdata_is_devptr;
    result.dpdatalength=sub.dpdatalength;
    return result;
}


template <typename T, typename Container>
mdspan<T,std::vector<size_t>>  mdspan<T, Container>::column(const size_t col_index, T*__restrict ptr)
{

    mdspan<T,std::vector<size_t>> result(1,true);
    datastruct<T> sub = this->column_v(col_index,result.dpextents, result.dpstrides,ptr);
    result.dpdata =      sub.dpdata;
    result.dprowmajor = sub.dprowmajor;
    result.dprank = sub.dprank;
    result.dpdata_is_devptr = sub.dpdata_is_devptr;
    result.dpdatalength=sub.dpdatalength;
    return result;
}

template <typename T, typename Container>
mdspan<T,std::vector<size_t>> mdspan<T, Container>:: row(const size_t row_index, T* __restrict ptr)
{
    mdspan<T,std::vector<size_t>> result(1,true);
    datastruct<T> sub = row_v(row_index,result.dpextents, result.dpstrides,ptr);
    result.dpdata =      sub.dpdata;
    result.dprowmajor = sub.dprowmajor;
    result.dprank = sub.dprank;
    result.dpdata_is_devptr = sub.dpdata_is_devptr;
    result.dpdatalength=sub.dpdatalength;
    return result;
}

template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>::transpose()
{
    mdspan<T,Container> result(this->dprank,false);
    datastruct<T> sub = transpose(result.dpextents, result.dpstrides);
    result.adopt_subdatastruct_helper(sub);
    return result;
}

template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>:: transpose(T* __restrict pdata)
{
    mdspan<T,Container> result(this->dprank,true);
    datastruct<T> sub = this->transpose_v(result.dpextents, result.dpstrides,pdata);
    result.adopt_subdatastruct_helper(sub);
    return result;

}



#endif
