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
using namespace upcxx;
using namespace std;

// --------------
// Logging utility

// Builds a logging message and writes it out when calling the destructor
class LogMessage {
public:
    LogMessage(const char *file, const char *function, int line);
    ~LogMessage();

    // output operator
    template<typename T>
    LogMessage &operator<<(const T &t) {
        os << t;
        return *this;
    }

private:
    std::ostringstream os;
};

#ifdef LOGGER_OUTPUT_ON

#define LOG(out) do { \
LogMessage(__FILE__,__func__,__LINE__) << out; \
} while (0)

#else
#define LOG(out)
#endif

// ----------------------------
// Assert with exception throws

class AssertionFailureException : public std::exception {
private:
    const char *expression;
    const char *file;
    int line;
    std::string message;
    std::string report;

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
    AssertionFailureException(const char *expression, const char *file, int line, const std::string &message);

    ~AssertionFailureException() noexcept override = default;

    // Log error before throwing
    void LogError();

    // The assertion message
    const char *what() const noexcept override;

    // The expression which was asserted to be true
    const char *Expression() const noexcept;

    // Source file
    const char *File() const noexcept;

    // Source line
    int Line() const noexcept;

    // Description of failure
    const char *Message() const noexcept;
};

// Assert that EXPRESSION evaluates to true, otherwise raise AssertionFailureException
// with associated MESSAGE (which may use C++ stream-style message formatting).
#define throw_assert(EXPRESSION, MESSAGE) if(!(EXPRESSION)) { throw AssertionFailureException(#EXPRESSION, __FILE__, __LINE__, (AssertionFailureException::StreamFormatter() << MESSAGE)); }

// --------
// Profiler

struct Threadpool;

struct Profiling_event {
    enum event {
        uninitialized, duration, timepoint
    };

    std::string name;
    std::string thread_id;
    std::chrono::high_resolution_clock::time_point m_start, m_stop;
    event ev_type;

    Profiling_event(string s, string t_id);

    void timestamp();

    void start();

    void stop();
};

std::string id_to_string(std::thread::id id);

std::string get_thread_id();

struct Profiler {
    std::string file_name{"prof.out"};
    list<Profiling_event> events;
    std::map<std::string, int> thread_id_map;

    std::mutex mtx;

    void map_team_threads(Threadpool &);

    void open(string s);

    void timestamp(string s);

    Profiling_event *start(string s);

    void stop(Profiling_event *prof_ev);

    void dump();
};

static Profiler profiler;

#endif //GTFXX_UTIL_H
