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
#include "datablock.hpp"
#include "gpu_memory_functions.h"
#include "expression_templates.h"

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
struct static_tag {
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


class mdspan_utilities;

template <typename T, typename Container>
class mdspan:public DataBlock<T>,
            public  expr::ExpressionInterface<mdspan_data<T,Container>>
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

    ptrdiff_t extent(const ptrdiff_t dim) const
    {
        return pextents[dim];
    };
    ptrdiff_t rank() const
    {
        return this->dprank;
    };
    ptrdiff_t stride(const ptrdiff_t dim) const
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

    ptrdiff_t datalength() const
    {
        return this->dpdatalength;
    };

bool location_check(expr::LocationCheckContext& ctx) const
{
    return ctx.check(*this);
}


};


template<typename T, typename Tag>
using mdspan_t =
    mdspan<
        T,
        typename container_selector<Tag>::template container<ptrdiff_t>>;



#endif
