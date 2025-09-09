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


#include "datastruct.h"






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

    mdspan(){};
    mdspan(T* __restrict data, const size_t datalength,const bool rowm, const Container& extents, const Container& strides);
    mdspan(T* __restrict data, const bool rowm, const Container& extents, const Container& strides);
    mdspan(T* __restrict data, const bool rowm, const Container& extents);
    mdspan(T* __restrict data, const bool rowm,const size_t rows,const size_t cols);
    ~mdspan();
    // Access operators
    using datastruct<T>::operator();

    inline T& operator()(const Container& extents);
    inline T operator()(const Container& extents)const;


    inline mdspan<T, Container>&operator=(const datastruct<T> & other);

    // Subspan methods
    mdspan<T, Container> subspan_view(const Container& offsets, const Container& sub_extents) const;
    mdspan<T, Container> subspan_view(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols)const;
    mdspan<T, Container> subspan_copy(const Container& offsets, const Container& sub_extents, T* __restrict sub_data) const;
    mdspan<T, Container> subspan_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T* __restrict sub_data)const;
    mdspan<T, Container> copy(T*__restrict  data,int destdevice,int srcdevice);

    mdspan<T, Container>column_view(const size_t col_index);
    mdspan<T, Container>row_view(const size_t row_index);
    mdspan<T, Container>column_copy(const size_t col_index,T* __restrict ptr);
    mdspan<T, Container>row_copy(const size_t row_index,T* __restrict ptr);
    mdspan<T, Container>transpose_view();
    mdspan<T, Container>transpose_copy(T* __restrict ptr);

    void glue_matrices(mdspan<T, Container> target, const vector<mdspan<T, Container>>& spans,  const vector<pair<size_t, size_t>>& offsets);
    void printvector();
    void printmatrix();

    bool device_update(bool default_device=true, int devicenum=0);
    bool device_update_async(bool default_device=true, int devicenum=0);
    bool host_update(bool default_device=true, int devicenum=0);
    bool host_update_async(bool default_device=true, int devicenum=0);
    bool device_download_release(bool default_device=true, int devicenum=0);
    bool device_upload(bool default_device=true,int devicenum=0);
    bool device_alloc(bool default_device=true,int devicenum=0);
    bool device_release(bool default_device=true,int devicenum=0);
    bool device_delete(bool default_device=true,int devicenum=0);
    void release_all_devices();
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
    Container& extents()const
    {
        return pextents;
    };
    Container& strides()const
    {
        return pstrides;
    };

    size_t datalength() const
    {
        return this->dpdatalength;
    };

protected:
    // Private member variables
    unordered_map<int,bool> pis_offloaded;

    void initialize_extents_and_strides(const Container&extents,const Container & strides);
    void initialize_extents(const Container&extents);
    void transfer_extents_and_strides(datastruct<T> &other);


    Container pextents;  // Use the ExtentContainer type
    Container pstrides;  // Use the StrideContainer type
};

#include <array>
#include <vector>
#include <cstddef>

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
inline void mdspan<T, Container>::transfer_extents_and_strides(datastruct<T> &other)
{
    if (this->pextents.data()!=other.dpextents)
    {
        if (other.dprank!=this->dprank)
        {
            if constexpr (DynamicContainer<Container>)
            {
                this->pextents.resize(other.dprank);
            }
        }
        this->pextents.data()=other.dpextents;
    }
    if (this->pstrides.data()!=other.dpstrides)
    {
        if (other.pdrank!=this->dprank)
        {
            if constexpr (DynamicContainer<Container>)
            {
                this->pstrides.resize(other.dprank);
            }
            this->pextents.data()=other.dpstrides;
        }
    }
}


template <typename T, typename Container>
inline void mdspan<T, Container>::release_all_devices()
{
    for(auto& p : this->pis_offloaded)
    {
        if(p.second==true)
        {
            Datastruct_GPU_Memory_Functions<T>::exit_struct(*this,p.first);
            p.second=false;
        }
    }
}

template <typename T, typename Container>
mdspan<T, Container>::~mdspan()
{
   mdspan<T, Container>::release_all_devices();
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
        for (int i = n - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        // Column-major layout: first dimension has stride 1
        strides[0] = 1;
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
    // Assign to datastruct
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
    // Assign to datastruct
    this->dpextents = pextents.data();
}


template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const  size_t datalength,const  bool rowm, const Container& extents, const Container& strides)
    :datastruct<T>(data,datalength,rowm,extents.size(),nullptr,nullptr,false,false,false)
{
    initialize_extents_and_strides(extents,strides);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm, const Container& extents, const Container& strides )
    : datastruct<T>(data, 0,rowm,extents.size(),nullptr,nullptr,false,false,false)
      // Initialize pdatastruct with placeholders
{
    initialize_extents_and_strides(extents,strides);
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);

}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data, const bool rowm,const  Container& extents)
    :  datastruct<T>(data,0,rowm,extents.size(),nullptr,nullptr,false,false,false)
{
    initialize_extents(extents);
    compute_strides(pextents,pstrides,rowm);
    this->dpstrides = pstrides.data();
    this->dpdatalength=compute_data_length_w(this->dpextents,this->dpstrides,this->dprank);
}



template <typename T, typename Container>
mdspan<T, Container>::mdspan(T* data,const bool rowm,  const size_t rows, const size_t cols)
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
}

template <typename T, typename Container>
mdspan<T, Container>&mdspan<T, Container>::operator=(const datastruct<T> & other)
{
    transfer_extents_and_strides(other);
    if(this->pdata!=other.pdata)
    {
        Datastruct_Functions<T>::release_all_devices();
    }
    datastruct<T>::operator=(other);

    return *this;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_upload(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices()) return false;

    this->pis_offloaded.try_emplace(devicenum, true);

    Datastruct_Functions<T>::create_in_struct(this,devicenum);
    return true;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_alloc(bool default_device,int devicenum)
{
    if (default_device)
        devicenum=omp_get_default_device();

    if(devicenum>omp_get_num_devices()) return false;
    this->pis_offloaded.try_emplace(devicenum, true);
    Datastruct_Functions<T>::create_out_struct(this,devicenum);
    return true;
}




template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_download_release(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    Datastruct_Functions<T>::update_host(this,devicenum);
    Datastruct_Functions<T>::release_struct(this,devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}


template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_delete(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    Datastruct_Functions<T>::exit_struct(this,devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_release(bool default_device,int devicenum)
{

    if (default_device) devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices())   return false;
    if(this->pis_offloaded.contains(devicenum)==false)  return false;

    Datastruct_Functions<T>::release_struct(this,devicenum);
    this->pis_offloaded.at(devicenum)=false;
    return true;
}

template <typename T, typename Container>inline
bool mdspan<T, Container>:: host_update(bool default_device,int devicenum)
{
    if (default_device)  devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices()) return false;
    if(this->pis_offloaded.contains(devicenum)==false) return false;
    Datastruct_Functions<T>::update_host(this,devicenum);
    return true;

}
template <typename T, typename Container>inline
bool mdspan<T, Container>:: device_update(bool default_device,int devicenum)
{

    if (default_device)  devicenum=omp_get_default_device();
    if(devicenum>omp_get_num_devices()) return false;
    if(this->pis_offloaded.contains(devicenum)==false) return false;

    Datastruct_Functions<T>::update_device(this,devicenum);
    return true;

}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan_view(const Container&offsets, const Container &sub_extents)const
{
    const size_t r=this->dprank;

    // Compute the offset to the starting point
    size_t offset_index = 0;

    #pragma omp simd reduction( + : offset_index )
    for (size_t i = 0; i < r; ++i)
    {
        offset_index += offsets[i] * this->dpstrides[i];
    }

    // Create a new mdspan_dynamic with the updated pointer, extents, and the same strides
    return mdspan(this->dpdata + offset_index,this->dprowmajor, sub_extents, pstrides);


}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::subspan_copy(const Container&offsets, const Container &sub_extents,T* __restrict sub_data )const
{
    const size_t r=this->dprank;

    // Compute the new strides for the subspan
    Container sub_strides;
    compute_strides(sub_extents, sub_strides, this->dprowmajor);
    vector<size_t> indices(r,0);
    vector<size_t> global_indices(r,0);
    while (true)
    {
        // Compute the current global indices
        #pragma omp simd
        for (size_t i = 0; i < r; ++i)
        {
            global_indices[i] = offsets[i] + indices[i];
        }

        // Compute the offsets for the original data and the new buffer
        size_t original_index = compute_offset_v(global_indices.data(), this->dpstrides,global_indices.size(), this->dprowmajor);
        size_t buffer_index = compute_offset_v(indices.data(),sub_strides.data(),indices.size(), this->dprowmajor);

        // Copy the data from the original tensor to the sub-buffer
        sub_data[buffer_index] = this->dpdata[original_index];

        // Increment the indices for the Cartesian product
        size_t dim = r;
        while (dim-- > 0)
        {
            if (++indices[dim] < sub_extents[dim])
            {
                break; // If no overflow, stop carrying
            }
            indices[dim] = 0; // Reset the current dimension and carry to the next
        }

        // If all dimensions have overflowed, we're done
        if (dim == size_t(-1))
        {
            break;
        }
    }
    // Create and return a new mdspan with the updated pointer, extents, and strides
    return mdspan(sub_data, this->dprowmajor, sub_extents, sub_strides );

}


template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspan_view(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols)const
{
    const size_t offset=row * this->pdstrides[0]+col * this->pdstrides[1];
    const Container ext= {tile_rows,tile_cols};
    return mdspan(this->pdata +offset,this->pdrowmajor,ext,this->pdstrides);
}



template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::copy(T*__restrict  data,int destdevice,int srcdevice)
{

    if (this->pd_data_is_devptr)
        omp_target_memcpy(data,this->dpdata,this->pdatalength,0,0,destdevice,srcdevice);
    else
        memcpy(data,this->dpdata,this->pdatalength);


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



template <typename T, typename Container>inline
mdspan<T, Container> mdspan<T, Container>::subspan_copy(const size_t row, const size_t col,const  size_t tile_rows,const  size_t tile_cols,T*__restrict  sub_data)const
{
    const size_t str0=this->pdstrides[0];
    const size_t str1=this->pdstrides[1];
    T *pd=this->pdata;
    #pragma omp parallel for simd collapse (2) shared(sub_data,pd,tile_cols,row,col,str0,str1)
    for (size_t i = 0; i < tile_rows; ++i)
    {
        for (size_t j = 0; j < tile_cols; ++j)
        {
            sub_data[i * tile_cols + j] =pd[ compute_offset(row + i, col + j, str0,str1) ];
        }
    }

    const Container sub_extents = {tile_rows, tile_cols};

    const Container sub_strides = (this->pdrowmajor==true)? Container{tile_cols, 1} :
                                  Container{1,tile_rows};

    return mdspan(sub_data,this->pdrowmajor, sub_extents, sub_strides );

}



template <typename T, typename Container>
void  mdspan<T, Container>::printmatrix()
{
    const size_t rows= this->dpextents[0];
    const size_t cols= this->dpextents[1];
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
    cout <<endl;
}

template <typename T, typename Container>
void  mdspan<T, Container>::printvector()
{
    const size_t rows=this->dpextents[0];
    for (size_t i = 0; i <rows; ++i)
    {
        std::cout << (*this)(i) << " ";
    }
    cout <<endl;
}



template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::column_view(const size_t col_index)
{
    const size_t num_rows = this->pdextents[0];
    const Container column_extents = {num_rows};
    const Container column_strides = {this->pdstrides[0]};
    return mdspan<T, Container>( this->pdata + col_index * this->dpstrides[1], this->pdrowmajor, column_extents, column_strides );
}

template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>:: row_view(const size_t row_index)
{

    const size_t num_cols = this->pdextents[1];
    const Container row_extents = {num_cols};
    const Container row_strides = {this->pdstrides[1]};
    return mdspan<T, Container>(this->pdata + row_index * this->pdstrides[0], this->pdrowmajor, row_extents, row_strides );
}


template <typename T, typename Container>
mdspan<T, Container> mdspan<T, Container>::column_copy(const size_t col_index, T*__restrict ptr)
{
    const size_t num_rows = this->pdextents[0];
    const size_t str0=this->pdstrides[0];
    const size_t str1=this->pdstrides[1];
    T *pd=this->pdata;
    #pragma omp parallel for simd shared(ptr,pd,num_rows,col_index, str0,str1)
    for (size_t i = 0; i < num_rows; ++i)
    {
        ptr[i] =pd[ compute_offset(i, col_index, str0,str1) ];
    }

    const Container sub_extents = {num_rows};

    const Container sub_strides = Container{1};
    return mdspan(ptr,this->pdrowmajor, sub_extents, sub_strides );
}

template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>:: row_copy(const size_t row_index, T* __restrict ptr)
{

    const size_t num_cols = this->pdextents[1];

    const size_t str0=this->pdstrides[0];
    const size_t str1=this->pdstrides[1];
    T *pd=this->pdata;
    #pragma omp parallel for simd shared(ptr,pd,row_index,num_cols,str0,str1)
    for (size_t j = 0; j < num_cols; ++j)
    {
        ptr[j] =pd[ compute_offset(row_index, j, str0,str1) ];
    }

    const Container sub_extents = {num_cols};

    const Container sub_strides = Container {1};

    return mdspan(ptr,this->pdrowmajor, sub_extents, sub_strides );
}

template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>::  transpose_view()
{
    Container transposed_extents = {this->extent(1), this->extent(0)};
    Container transposed_strides = {this->stride(1), this->stride(0)};
    mdspan md=mdspan(this->data(),this->datalength(), this->prowmajor(),  transposed_extents,   transposed_strides);
    return md;
}

template <typename T, typename Container>
mdspan<T, Container>mdspan<T, Container>:: transpose_copy(T* __restrict pdata)
{
    Container transposed_extents = {this->extent(1), this->extent(0)};
    Container transposed_strides = {this->stride(1), this->stride(0)};
    size_t s=this->extent(1)* this->extent(0);
    for (size_t i=0; i<s; i++)
    {
        pdata[i]=this->pdata[i];
    }
    return mdspan(pdata,this->datalength(), this->rowmajor()(),  transposed_extents,   transposed_strides);
}


template <typename T, typename Container>
void mdspan<T, Container>::glue_matrices(mdspan<T, Container> target, const vector<mdspan<T, Container>>& spans,
        const vector<pair<size_t, size_t>>& offsets)
{

    #pragma omp parallel for
    for (size_t idx = 0; idx < spans.size(); ++idx)
    {
        const mdspan<T,Container>& span = spans[idx];
        const size_t row_offset = offsets[idx].first;
        const size_t col_offset = offsets[idx].second;
        const size_t ext0=span.extent(0);
        const size_t ext1=span.extent(1);
        // Copy the current span into the target at the given offset
        #pragma omp simd collapse(2)
        for (size_t i = 0; i < ext0; ++i)    // Rows of the span
        {
            for (size_t j = 0; j < ext1; ++j)    // Columns of the span
            {
                target(row_offset + i, col_offset + j) = span(i, j);
            }
        }
    }
}




#endif
