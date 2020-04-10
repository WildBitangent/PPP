/**
 * @file    base.h
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          This file contains basic helpers to simplify rest of the application.
 *
 * @date    2018-03-28
 */

#ifndef BASE_H
#define BASE_H

#include <mpi.h>
#include <immintrin.h>
#include <iostream>
#include <memory>

/**
 * @brief AutoHandle class enables RAII for object represented by handle (for
 * example HDF5 handles).
 *
 * This can be used to ensure that the object is released as soon as it's handle
 * leaves it's scope.
 *
 * For example:
 * AutoHandle<hid_t> hPropList(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
 * where H5Pcreate creates object and returns handle "hid_t", which is closed by
 * function H5Pclose(hid_t handle) (in AutoHandle's destructor).
 */
template<typename T>
class AutoHandle
{
public:
    class BaseDeleter
    {
    public:
        virtual ~BaseDeleter() { }

        virtual void Delete(T handle) = 0;
    };

    template<typename D>
    class Deleter : public BaseDeleter
    {
    public:
        Deleter(D deleter)
            : m_deleter(deleter)
        {
        }

        virtual ~Deleter() { }

        virtual void Delete(T handle) {
            if(m_deleter)
                m_deleter(handle);
        }

    protected:
        D m_deleter;
    };

    template<typename D>
    AutoHandle(T handle, D deleter)
        : m_handle(handle), m_pDeleter(new Deleter<D>(deleter))
    {
    }

    AutoHandle(const AutoHandle<T> &) = delete;
    AutoHandle &operator=(const AutoHandle<T> &) = delete;

    ~AutoHandle()
    {
        m_pDeleter->Delete(m_handle);
    }

    template<typename D>
    void Set(T handle, D deleter) {
        m_pDeleter->Delete(m_handle);
        m_handle = handle;
        m_pDeleter = std::unique_ptr<BaseDeleter>(new Deleter<D>(deleter));
    }

    operator T&() { return m_handle; }

protected:
    T m_handle;                              ///< Internal handle to be released
    std::unique_ptr<BaseDeleter> m_pDeleter; ///< Functor to release handle by calling specified function
};

/**
 * @brief Simple implementation of "allocator concept" from STL which
 * can be used to automatically allocate aligned memory in STL types such as
 * std::vector.
 *
 * For example:
 * "std::vector<float, AlignedAllocator<float> >" will create array of allocated
 * as 64-byte boundary aligned memory.
 */
template<class T>
struct AlignedAllocator {
    typedef T value_type;
    AlignedAllocator() = default;

    template<class U>
    constexpr AlignedAllocator(const AlignedAllocator<U> &) noexcept
    {
    }

    T *allocate(std::size_t n) {
        if(n > std::size_t(-1) / sizeof(T))
            throw std::bad_alloc();
        if(auto p = static_cast<T *>(_mm_malloc(n * sizeof(T), 64)))
            return p;

        throw std::bad_alloc();
    }

    void deallocate(T *p, std::size_t) noexcept {
        _mm_free(p);
    }
};

template<class T, class U>
bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) {
    return true;
}

template<class T, class U>
bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) {
    return false;
}

#endif // BASE_H
