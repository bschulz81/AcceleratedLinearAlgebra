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
#include <array>
#include <vector>
#include <cstddef>

#include <unordered_map>
#include <set>

#include "datablock.h"


template<typename T, typename Container>class mdspan;
template<typename T, typename Container>
class mdspan;

template<typename T>
class DataBlock;

using namespace std;


// Concept definitions
template <typename Container>
concept StaticContainer =
    requires(Container c, ptrdiff_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<typename Container::value_type>;
    (!requires(Container c, ptrdiff_t i)
    {
        c.reserve(i);
    });
};

template <typename Container>
concept DynamicContainer =
    requires(Container c, ptrdiff_t i)
{
    {
        c.size()
    }
    -> std::convertible_to<typename Container::value_type>;
    c.reserve(i);  // Require reserve() for dynamic containers
};

struct dynamic_tag {};

template <std::ptrdiff_t N>
struct static_tag
{
    static constexpr std::ptrdiff_t rank = N;
};

template<typename Tag>
struct container_selector;

template<>
struct container_selector<dynamic_tag>
{
    template<typename T>
    using container = std::vector<T>;
};

template<std::ptrdiff_t N>
struct container_selector<static_tag<N>>
{
    template<typename T>
    using container = std::array<T, N>;
};


struct LocationCheckContext
{
    bool check_started = false;

    bool data_is_device = false;
    int device_number = -INT_MAX;

    template<typename T>
    bool check(const DataBlock<T>& d)
    {
#if defined(Unified_Shared_Memory)
        return d.dpdata != nullptr;
#endif

        if (d.data() == nullptr)
            return false;

        bool this_is_device =
            d.data_is_devptr();

        if (!check_started)
        {
            check_started = true;

            data_is_device = this_is_device;

            if (this_is_device)
                device_number = d.devptr_num();

            return true;
        }

        if (data_is_device != this_is_device)
            return false;

        if (data_is_device &&
                device_number != d.devptr_num())
            return false;

        return true;
    }
};



class mdspan_utilities;


template<typename Container>
struct Layout
{

    ptrdiff_t rank = 0;
    bool rowmajor=true;
    Container extents;
    Container strides;
    template<typename OtherContainer>
    bool operator==(const Layout<OtherContainer>& other) const
    {
        if (rank != other.rank)
            return false;
        if(rowmajor!=other.rowmajor)
            return false;

        if (extents.size() != other.extents.size())
            return false;

        for (size_t i = 0; i < extents.size(); ++i)
        {
            if (extents[i] != other.extents[i])
                return false;

            if (strides[i] != other.strides[i])
                return false;
        }

        return true;
    }
template<typename OtherContainer>
    bool operator!=(const Layout<OtherContainer>& other) const
    {
        return !(*this == other);
    }
};

template <typename T, typename Container>
class mdspan:public DataBlock<T>
{

protected:

    friend class mdspan_utilities;


    class DeviceOffloadRegistry
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
                ptrdiff_t erased = it->second.erase(iv);

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


    void initialize_extents_and_strides(const Container & extents,const Container & strides);
    void initialize_extents(const Container&extents);
    void compute_initialize_strides(const Container& extents,const bool rowmajor);

    Container pextents;
    Container pstrides;
    shared_ptr<DeviceOffloadRegistry> offload_registry=make_shared<DeviceOffloadRegistry>();

    bool p_owns_device_offload=false;

public:


    mdspan() {};

    mdspan(const DataBlock<T>& ds,const shared_ptr<mdspan<T,Container>::DeviceOffloadRegistry> &dev);

    mdspan(const mdspan<T, Container>& other);
    mdspan(mdspan<T, Container>&& other)noexcept;


    mdspan<T, Container> &operator=(const mdspan<T,Container> & other);
    mdspan<T, Container> &operator=(const DataBlock<T> & other);
    mdspan<T, Container> &operator=(mdspan<T, Container>&& other)noexcept;


    mdspan(T* data, const ptrdiff_t datalength, const Container& extents, const Container& strides, const DataBlockConfig  config);
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

    ptrdiff_t extent(const ptrdiff_t dim) const;
    ptrdiff_t rank() const;
    ptrdiff_t stride(const ptrdiff_t dim) const;

    const Container& extents()const;
    const Container& strides()const;



    ptrdiff_t datalength() const;
    bool location_check(LocationCheckContext& ctx) const;



};


template<typename T, typename Tag>
using mdspan_t =
    mdspan<
    T,
    typename container_selector<Tag>::template container<ptrdiff_t>>;

#include "mdspan.hpp"

#endif
