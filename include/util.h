//
// Created by Eric Darve on 9/16/18.
//

#ifndef GTFXX_UTIL_H
#define GTFXX_UTIL_H

#define LOGGER_OUTPUT_ON

#include <iostream>
#include <sstream>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include <exception>
#include <stdexcept>
#include <random>

#include <array>
#include <vector>
#include <list>
#include <queue>
#include <map>

#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <upcxx/upcxx.hpp>

// --------------
// Logging utility

// Builds a logging message and writes it out when calling the destructor
class Log_message {
public:
    Log_message(const char *file, const char *function, int line);
    ~Log_message();

    // output operator
    template<typename T>
    Log_message &operator<<(const T &t) {
        os << t;
        return *this;
    }

private:
    std::ostringstream os;
};

#ifdef LOGGER_OUTPUT_ON

#define LOG(out) do { \
Log_message(__FILE__,__func__,__LINE__) << out; \
} while (0)

#else
#define LOG(out)
#endif

// ----------------------------
// Assert with exception throws

class Assertion_failure_exception : public std::exception {
private:
    const char *m_expression;
    const char *m_file;
    int m_line;
    std::string m_message;
    std::string m_report;

public:
    // Helper class for formatting assertion message
    class StreamFormatter {
    private:
        std::ostringstream stream;

    public:
        operator std::string() const;

        template<typename T>
        StreamFormatter &operator<<(const T &value) {
            stream << value;
            return *this;
        }
    };

    // Construct an assertion failure exception
    Assertion_failure_exception(const char *expression, const char *file, int line, const std::string &message);

    ~Assertion_failure_exception() noexcept override = default;

    // Log error before throwing
    void log_error();

    // The assertion message
    const char *what() const noexcept override;

    // The expression which was asserted to be true
    const char *expression() const noexcept;

    // Source file
    const char *file() const noexcept;

    // Source line
    int line() const noexcept;

    // Description of failure
    const char *message() const noexcept;
};

// Assert that EXPRESSION evaluates to true, otherwise raise Assertion_failure_exception
// with associated MESSAGE (which may use C++ stream-style message formatting).
#define throw_assert(EXPRESSION, MESSAGE) if(!(EXPRESSION)) { throw Assertion_failure_exception(#EXPRESSION, __FILE__, __LINE__, (Assertion_failure_exception::StreamFormatter() << MESSAGE)); }

// ---------------------------------------------
// Vector class with different memory allocators

#ifdef VECTOR_ALLOCATE_LOG
template<class T>
struct custom_allocator {
    typedef T value_type;

    custom_allocator() noexcept = default;

    template<class U>
    explicit custom_allocator(const custom_allocator<U> &) noexcept {};

    T *allocate(std::size_t n) {
        printf("Allocating %ld bytes\n", n * sizeof(T));
        return static_cast<T *>(::operator new(n * sizeof(T)));
    }

    void deallocate(T *p, std::size_t n) {
        printf("Freeing %ld bytes\n", n * sizeof(T));
        ::delete (p);
    }
};

template<class T, class U>
constexpr bool operator==(const custom_allocator<T> &, const custom_allocator<U> &) noexcept {
    return true;
}

template<class T, class U>
constexpr bool operator!=(const custom_allocator<T> &, const custom_allocator<U> &) noexcept {
    return false;
}

typedef std::vector<int64_t, custom_allocator<int64_t> > Vector;

#else
typedef std::vector<int64_t> Vector;
#endif

// --------
// Profiler

namespace gtfxx {
    struct Thread_pool;
}

struct Profiling_event {
    enum event {
        uninitialized, duration, timepoint
    };

    std::string name;
    std::string thread_id;
    std::chrono::high_resolution_clock::time_point m_start, m_stop;
    event ev_type;

    Profiling_event(std::string s, std::string t_id);

    void timestamp();

    void start();

    void stop();
};

std::string id_to_string(std::thread::id id);

std::string get_thread_id();

struct Profiler {
    std::string file_name{"prof.out"};
    std::list<Profiling_event> events;
    std::map<std::string, int> thread_id_map;

    std::mutex mtx;

    void record_thread_ids(gtfxx::Thread_pool &);

    void open(std::string s);

    void timestamp(std::string s);

    Profiling_event *start(std::string s);

    void stop(Profiling_event *prof_ev);

    void dump();
};

static Profiler profiler;

#endif //GTFXX_UTIL_H
