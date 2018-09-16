#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <cassert>
#include <exception>
#include <stdexcept>

#include <random>

#include <list>
#include <queue>
#include <map>
#include <array>
#include <vector>

#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>

using namespace std;

#include "gtest/gtest.h"

#include <upcxx/upcxx.hpp>

using namespace upcxx;

using std::min;
using std::max;

using std::list;
using std::vector;
using std::unordered_map;
using std::array;

using std::thread;
using std::mutex;
using std::bind;

typedef std::function<void()> Base_task;

// Builds a logging message and writes it out when calling the destructor
class LogMessage {
public:
    LogMessage(const char *file, const char *function, int line) {
        os << file << ":" << line << " (" << function << ") ";
    }

    // output operator
    template<typename T>
    LogMessage &operator<<(const T &t) {
        os << t;
        return *this;
    }

    ~LogMessage() {
        os << "\n";
        std::cout << os.str();
        std::cout.flush();
    }

private:
    std::ostringstream os;
};

#if 1
#define LOG(out) do { \
LogMessage(__FILE__,__func__,__LINE__) << out; \
} while (0)
#else
#define LOG(out)
#endif

/// Exception type for assertion failures
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
        operator std::string() const {
            return stream.str();
        }

        template<typename T>
        StreamFormatter &operator<<(const T &value) {
            stream << value;
            return *this;
        }
    };

    // Log error before throwing
    void LogError() {
        std::cerr << report << std::endl;
    }

    // Construct an assertion failure exception
    AssertionFailureException(const char *expression, const char *file, int line, const std::string &message)
            : expression(expression), file(file), line(line), message(message) {
        std::ostringstream outputStream;

        if (!message.empty()) {
            outputStream << message << ": ";
        }

        std::string expressionString = expression;

        if (expressionString == "false" || expressionString == "0" || expressionString == "FALSE") {
            outputStream << "Unreachable code assertion";
            /* We asserted false to abort at a line that code code
             * should not be able to reach. */
        } else {
            outputStream << "Assertion '" << expression << "'";
        }

        outputStream << " failed in file '" << file << "' line " << line;
        report = outputStream.str();

        LogError();
    }

    // The assertion message
    virtual const char *what() const throw() {
        return report.c_str();
    }

    // The expression which was asserted to be true
    const char *Expression() const throw() {
        return expression;
    }

    // Source file
    const char *File() const throw() {
        return file;
    }

    // Source line
    int Line() const throw() {
        return line;
    }

    // Description of failure
    const char *Message() const throw() {
        return message.c_str();
    }

    ~AssertionFailureException() throw() {
    }
};


/// Assert that EXPRESSION evaluates to true, otherwise raise AssertionFailureException with associated MESSAGE (which may use C++ stream-style message formatting)
#define throw_assert(EXPRESSION, MESSAGE) if(!(EXPRESSION)) { throw AssertionFailureException(#EXPRESSION, __FILE__, __LINE__, (AssertionFailureException::StreamFormatter() << MESSAGE)); }

struct Profiling_Event {
    enum event {
        uninitialized, duration, timepoint
    };

    std::string name;
    std::string thread_id;
    std::chrono::high_resolution_clock::time_point m_start, m_stop;
    event ev_type;

    Profiling_Event(string s, string t_id) :
            name(std::move(s)), thread_id(std::move(t_id)), ev_type(uninitialized) {}

    void timestamp() {
        ev_type = timepoint;
        m_start = std::chrono::high_resolution_clock::now();
    }

    void start() {
        ev_type = duration;
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        m_stop = std::chrono::high_resolution_clock::now();
    }
};

struct Thread_team;

std::string id_to_string(std::thread::id id) {
    std::stringstream s_id;
    s_id << id;
    return s_id.str();
}

std::string get_thread_id() {
    return id_to_string(std::this_thread::get_id());
}

struct Profiler {
    std::string file_name{"prof.out"};
    list<Profiling_Event> events;
    std::map<std::string, int> thread_id_map;

    std::mutex mtx;

    void map_team_threads(Thread_team &);

    void open(string s) {
        // Open a new file and clear up the list of events
        file_name = std::move(s);
        events.clear();
        thread_id_map.clear();
    }

    void timestamp(string s) {
        auto thread_id = get_thread_id();
        Profiling_Event prof_ev(std::move(s), thread_id);
        prof_ev.timestamp();
        {
            std::lock_guard<std::mutex> lck(mtx);
            events.push_back(prof_ev);
        }
    }

    Profiling_Event *start(string s) {
        auto thread_id = get_thread_id();
        Profiling_Event *prof_ev = new Profiling_Event(std::move(s), thread_id);
        prof_ev->start();
        return prof_ev;
    }

    void stop(Profiling_Event *prof_ev) {
        prof_ev->stop();
        {
            std::lock_guard<std::mutex> lck(mtx);
            events.push_back(*prof_ev);
        }
        delete prof_ev;
    }

    void dump() {
        FILE *fp = std::fopen(file_name.c_str(), "w");
        if (!fp) {
            std::perror("File opening failed");
            return;
        }

        fprintf(fp, "nthreads %ld\n", thread_id_map.size());
        for (auto &it : thread_id_map) {
            fprintf(fp, "tidmap %s %d\n", it.first.c_str(), it.second);
        }

        for (auto &it : events) {
            auto d_start = std::chrono::duration<long long, std::nano>(it.m_start.time_since_epoch());
            switch (it.ev_type) {
                case Profiling_Event::duration : {
                    auto d_end = std::chrono::duration<long long, std::nano>(it.m_stop.time_since_epoch());
                    fprintf(fp, "tid %s start %lld end %lld name %s\n",
                            it.thread_id.c_str(),
                            d_start.count(), d_end.count(), it.name.c_str());
                }
                    break;
                case Profiling_Event::timepoint :
                    fprintf(fp, "tid %s timestamp %lld name %s\n", it.thread_id.c_str(),
                            d_start.count(), it.name.c_str());
                    break;
                case Profiling_Event::uninitialized :
                    printf("Fatal error; we found an uninitialized profiling event");
                    exit(1);
                    break;
            }
        }

        std::fclose(fp);
    }
} profiler;

struct Task : public Base_task {
    float m_priority = 0.0; // task priority

    Task(const Base_task &&a_f, float a_priority = 0.0) :
            Base_task(a_f), m_priority(a_priority) {}
};

// Task comparison based on their priorities
struct Task_comparison {
public:
    bool operator()(const Task *a_lhs, const Task *a_rhs) const {
        return (a_lhs->m_priority < a_rhs->m_priority);
    };
};

// Thread with priority queue management
struct Thread_prio {
    Thread_team *team;
    unsigned short m_id; // team id

    std::priority_queue<Task *, vector<Task *>, Task_comparison>
            ready_queue;
    thread th;
    mutex mtx;

    std::atomic_bool m_empty;
    // For optimization to avoid testing ready_queue.empty() in some cases

    // Thread starts executing the function spin()
    void start();

    // Add new task to queue; thread safe
    void spawn(Task *a_t) {
        std::lock_guard<std::mutex> lck(mtx);
        m_empty.store(false);
        ready_queue.push(a_t); // Add task to queue
    }

    // Not thread safe
    Task *pop_unsafe() {
        Task *tsk = ready_queue.top();
        ready_queue.pop();
        if (ready_queue.empty())
            m_empty.store(true);
        return tsk;
    }

    // join() the thread
    void join() {
        if (th.joinable()) {
            th.join();
        }
    }

    ~Thread_prio() {
        join();
    }
};

struct Thread_team : public vector<Thread_prio *> {
    vector<Thread_prio> v_thread;
    unsigned long n_query_spawn = 4;  // Optimization parameter
    unsigned long n_query_steal = 16; // Optimization parameter
    std::atomic_int ntasks; // Number of ready tasks in any thread queue
    std::atomic_bool m_stop;

    explicit Thread_team(const int n_thread) :
            v_thread(n_thread), ntasks(0), m_stop(false) {
        for (int i = 0; i < n_thread; ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = static_cast<unsigned short>(i);
        }
    }

    void start() {
        ntasks.store(0);
        m_stop.store(false);
        for (auto &th : v_thread) th.start();
    }

    void join();

    void spawn(const int a_id, Task *a_task) {
        assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());

        ++ntasks;

        int id_ = a_id;

        // Check if queue is empty
        if (!v_thread[a_id].m_empty.load()) {
            // Thread is already busy
            // Are there other threads that are idle?
            const unsigned long n_query = min(1 + n_query_spawn, v_thread.size());
            for (unsigned long i = a_id + 1; i < a_id + n_query; ++i) {
                auto j = i % v_thread.size();
                if (v_thread[j].m_empty.load()) {
                    // We have found an idle thread
                    id_ = static_cast<int>(j);
                    break;
                }
            }
        }

        {
            char timestamp_message[80];
            if (id_ == a_id) {
                sprintf(timestamp_message, "spawn_to %d", a_id);
            } else {
                sprintf(timestamp_message, "spawn_other [%d->]%d", a_id, id_);
            }
            profiler.timestamp(string(timestamp_message));
        }

        v_thread[id_].spawn(a_task);
    }

    void steal(unsigned short a_id) {
        const unsigned long n_query = min(n_query_steal, v_thread.size());
        for (unsigned long i = a_id + 1; i < a_id + n_query; ++i) {
            auto j = i % v_thread.size();
            Thread_prio &thread_j = v_thread[j];
            if (!thread_j.m_empty.load()) {
                std::unique_lock<std::mutex> lck(thread_j.mtx);
                if (!thread_j.ready_queue.empty()) {
                    // We have found a non empty task queue
                    Task *tsk = thread_j.pop_unsafe();
                    lck.unlock();
                    {
                        char timestamp_message[80];
                        sprintf(timestamp_message, "steal %d[<-%ld]", a_id, j);
                        profiler.timestamp(timestamp_message);
                    }
                    v_thread[a_id].spawn(tsk);
                    break;
                }
            }
        }
    }
};

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin_task(Thread_prio *a_thread) {
    auto pe = profiler.start("overhead");

    std::unique_lock<std::mutex> lck(a_thread->mtx);

    while (true) {

        while (!a_thread->ready_queue.empty()) {
            Task *tsk = a_thread->pop_unsafe();
            lck.unlock();
            profiler.stop(pe);

            (*tsk)();

            pe = profiler.start("overhead");
            --(a_thread->team->ntasks);

            // Free memory
            delete tsk;

            lck.lock();
        }

        lck.unlock();
        // Try to steal a task
        a_thread->team->steal(a_thread->m_id);

        while (a_thread->m_empty.load()) {

            // Return if stop=true and no tasks are left
            if (a_thread->team->m_stop.load() && a_thread->team->ntasks.load() <= 0) {
                profiler.stop(pe);
                return;
            }

            profiler.stop(pe);

            // When queue is empty, yield
            pe = profiler.start("wait");
            //  std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::microseconds(40));
            profiler.stop(pe);

            pe = profiler.start("overhead");

            if (a_thread->m_empty.load()) {
                // Try to steal a task
                a_thread->team->steal(a_thread->m_id);
            }
        }

        lck.lock();
    }
}

void Thread_prio::start() {
    m_empty.store(true);
    th = thread(spin_task, this); // Execute tasks in queue
}

typedef std::atomic_int Promise;

struct Matrix3_promise : vector<Promise> {
    int n0, n1, n2;

    Matrix3_promise() : n0(0), n1(0), n2(0) {};

    // Initialize with -1
    Matrix3_promise(int a_n0, int a_n1, int a_n2) :
            vector<Promise>(a_n0 * a_n1 * a_n2),
            n0(a_n0), n1(a_n1), n2(a_n2) {
        for (int i = 0; i < n0 * n1 * n2; ++i) {
            operator[](i).store(-1);
        }
    };

    std::atomic_int &operator()(int i, int j, int k) {
        assert(i >= 0 && i < n0);
        assert(j >= 0 && j < n1);
        assert(k >= 0 && k < n2);
        return operator[](i + n0 * (j + n1 * k));
    }
};

typedef array<int, 3> int3;

struct Dependency_flow {
    typedef std::function<int(const int3)> Dependency_count;
    typedef std::function<void(const int3)> Build_task;

    // Thread team
    Thread_team *team = nullptr;

    explicit Dependency_flow(Thread_team *a_team) : team(a_team) {}

    // How to calculate the number of dependencies for promises
    Dependency_count m_dep_count = nullptr;

    // How to construct a task
    Build_task m_build_task = nullptr;

    virtual ~Dependency_flow() = default;

    // Decrement the dependency counter and spawn task if ready
    virtual void fulfill_promise(int3) = 0;
};

struct Task_flow : public Dependency_flow {

    typedef std::function<int(int3 &)> Map_task;

    // Tasks are indexed using a 3D grid of atomic dependencies
    Matrix3_promise promise_grid;

    // Mapping from task index to thread id
    Map_task m_map = nullptr;

    Task_flow &dependency_count(Dependency_count f) {
        m_dep_count = std::move(f);
        return *this;
    }

    Task_flow &define_task(Build_task f) {
        m_build_task = std::move(f);
        return *this;
    }

    void compute_on(Map_task a_map) {
        m_map = std::move(a_map);
    }

    Task_flow(Thread_team *a_team, int n0, int n1, int n2) :
            Dependency_flow(a_team), promise_grid(n0, n1, n2) {}

    // Spawn a task from index
    void async(int3);

    // Spawn a task that is already initialized
    void async(int3, Task *);

    // Decrement the dependency counter and spawn task if ready
    void fulfill_promise(int3) override;
};

// Spawn task
void Task_flow::async(int3 idx, Task *a_tsk) {
    // Basic sanity check
    assert(team != nullptr);
    assert(a_tsk != nullptr);
    assert(m_map != nullptr);
    assert(m_map(idx) >= 0);

    team->spawn(/*task map*/ m_map(idx), a_tsk);
}

// Enqueue task immediately
void Task_flow::async(int3 idx) {
    throw_assert(this->m_build_task != nullptr,
                 "define_task() was not called; the task cannot be defined");
    Build_task f = this->m_build_task;
    async(idx, new Task([f, idx]() { f(idx); }));
}

void Task_flow::fulfill_promise(int3 idx) {
    assert(0 <= idx[0] && idx[0] < promise_grid.n0);
    assert(0 <= idx[1] && idx[1] < promise_grid.n1);
    assert(0 <= idx[2] && idx[2] < promise_grid.n2);

    auto atomic_count = &(promise_grid(idx[0], idx[1], idx[2]));

    // Decrement counter
    int wait_count = std::atomic_fetch_sub(atomic_count, 1);

    if (wait_count == -1) { // Uninitialized task
        throw_assert(m_dep_count != nullptr,
                "dependency_count() was not called; the promise cannot be initialized");
        int wait_count_init = m_dep_count(idx);
        wait_count = std::atomic_fetch_add(atomic_count, wait_count_init);
        // wait_count = value before add operation
        wait_count += wait_count_init + 1;
    }

    if (wait_count == 0) { // task is ready to run
        throw_assert(this->m_build_task != nullptr,
                     "define_task() was not called; the task cannot be defined");
        Build_task f = this->m_build_task;
        async(idx, new Task([f, idx]() { f(idx); }));
    }
}

void capture_master_thread() {
    if (upcxx::backend::initial_master_scope == nullptr) {
        upcxx::backend::initial_master_scope = new persona_scope{backend::master};
    }
}

void release_master_thread() {
    if (upcxx::backend::initial_master_scope != nullptr) {
        upcxx::liberate_master_persona();
    }
}

// Progress thread for communications
struct Thread_comm {
    Thread_team *team = nullptr;
    queue<Task *> ready_queue;
    thread th;
    mutex mtx;
    std::atomic_bool m_empty;

    void start();

    // Add new task to queue; thread safe
    void spawn(Task *a_t) {
        std::lock_guard<std::mutex> lck(mtx);
        m_empty.store(false);
        ready_queue.push(a_t); // Add task to queue
    }

    // Not thread safe
    Task *pop_unsafe() {
        Task *tsk = ready_queue.front();
        ready_queue.pop();
        if (ready_queue.empty())
            m_empty.store(true);
        return tsk;
    }

    // join() the thread
    void join() {
        if (th.joinable()) {
            th.join();
        }
    }

    ~Thread_comm() {
        join();
    }
};

void active_message_progress() {
    upcxx::progress();
}

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin_comm(Thread_comm *a_thread) {

    // Acquiring master persona
    upcxx::persona_scope scope(upcxx::master_persona());

    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {

        while (!a_thread->ready_queue.empty()) {
            Task *tsk = a_thread->pop_unsafe();
            lck.unlock();

            (*tsk)(); // Issue active message

            --(a_thread->team->ntasks);

            // Free memory
            delete tsk;

            // Make progress on active messages
            active_message_progress();

            lck.lock();
        }

        lck.unlock();

        while (a_thread->m_empty.load()) {
            // Return if stop=true and no tasks are left
            if (a_thread->team->m_stop.load() && a_thread->team->ntasks.load() <= 0) {
                return;
            }

            // When queue is empty, sleep
            std::this_thread::sleep_for(std::chrono::microseconds(40));

            // Make progress on active messages
            active_message_progress();
        }

        lck.lock();
    }
}

void Thread_comm::start() {
    m_empty.store(true);
    th = thread(spin_comm, this);
}

namespace hash_array {

// Another option is to specify the maximum sizes in each dimension
// for the sparse grid.
    inline void hash_combine(std::size_t &seed, int const &v) {
        seed ^= std::hash<int>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    struct hash {
        size_t
        operator()(array<int, 3> const &key) const {
            size_t seed = 0;
            hash_combine(seed, key[0]);
            hash_combine(seed, key[1]);
            hash_combine(seed, key[2]);
            return seed;
        }
    };
}

struct Channel : public Dependency_flow {

    typedef std::function<void(int3)> Finalize_task;

    // How to unpack after a data transfer is complete
    Finalize_task m_finalize = nullptr;

    unordered_map<int3, std::atomic_int, hash_array::hash> promise_map;
    mutex mtx_graph; // Used to concurrently make changes to task_map

    Channel &dependency_count(Dependency_count f) {
        m_dep_count = std::move(f);
        return *this;
    }

    void define_task(Build_task f) {
        m_build_task = std::move(f);
    }

    void set_finalize(Finalize_task a_finalize) {
        m_finalize = std::move(a_finalize);
    }

    void finalize(int3 idx) {
        throw_assert(m_finalize != nullptr,
                     "set_finalize() was not called; the finalize() task cannot be called");
        m_finalize(idx);
    }

    explicit Channel(Thread_team *a_team) : Dependency_flow(a_team) {}

    // Find a promise in promise_map and return a pointer
    Promise *find_promise(int3 &);

    // Enqueue a communication from index
    void async(int3);

    // Enqueue a communication that is already initialized
    void async(int3, Task *);

    // Decrement the dependency counter and enqueue communication if ready
    void fulfill_promise(int3) override;
};

Promise *Channel::find_promise(int3 &idx) {
    Promise *t_ = nullptr;

    std::unique_lock<std::mutex> lck(mtx_graph);

    auto prom_it = promise_map.find(idx);

    if (prom_it == promise_map.end()) {
        // Promise does not exist; create it
        promise_map.emplace(std::piecewise_construct,
                            std::forward_as_tuple(idx),
                            std::forward_as_tuple(-1)); // Insert in promise_map
        prom_it = promise_map.find(idx); // Find location
//        prom_it->load(-1);
    }

    t_ = &(prom_it->second); // Get promise

    lck.unlock();

    assert(t_ != nullptr);

    return t_;
}

// Initialize task and spawn it
void Channel::async(int3 idx) {
    throw_assert(this->m_build_task != nullptr,
                 "define_task() was not called; the task cannot be defined");
    Build_task f = this->m_build_task;
    async(idx, new Task([f, idx]() { f(idx); }));
}

/* TODO: fulfill_promise is nearly the same as Task_flow.
 * Need to make this part of the base class.
 * The only difference is find_task. */
void Channel::fulfill_promise(int3 idx) {
    auto atomic_count = find_promise(idx);

    // Decrement counter
    int wait_count = std::atomic_fetch_sub(atomic_count, 1);

    if (wait_count == -1) { // Uninitialized task
        throw_assert(m_dep_count != nullptr,
                     "dependency_count() was not called, the promise cannot be initialized");
        int wait_count_init = m_dep_count(idx) + 1;
        wait_count = std::atomic_fetch_add(atomic_count, wait_count_init);
        // wait_count = value before add operation
        wait_count += wait_count_init;
    }

    if (wait_count == 0) { // task is ready to run
        throw_assert(this->m_build_task != nullptr,
                     "define_task() was not called; the task cannot be defined");
        Build_task f = this->m_build_task;
        async(idx, new Task([f, idx]() { f(idx); }));
    }
}

// Task flow context
struct Context {
    map<string, Task_flow> m_map_task;
    map<string, Channel> m_map_comm;

    Thread_comm th_comm;

    Task_flow empty_task{nullptr, 0, 0, 0};
    Channel empty_comm{nullptr};

} gtfxx_context;


void Profiler::map_team_threads(Thread_team &team) {
    // IDs of team threads
    for (auto &th_prio : team.v_thread) {
        thread_id_map[id_to_string(th_prio.th.get_id())] = th_prio.m_id;
    }

    // Adding the ID of the comm thread
    thread_id_map[id_to_string(gtfxx_context.th_comm.th.get_id())] = -1;

    // ID of the main() thread that is running this function
    thread_id_map[id_to_string(std::this_thread::get_id())] = -2;
}

void Thread_team::join() {
    m_stop.store(true);
    for (auto &th : v_thread) th.join();
    // Joining comm thread
    gtfxx_context.th_comm.th.join();
}

void task_emplace(string s, int n0, int n1, int n2, Thread_team *team) {
    gtfxx_context.m_map_task.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(s),
                                     std::forward_as_tuple(team, n0, n1, n2)
    );
}

void comm_emplace(string s, Thread_team *team) {
    gtfxx_context.m_map_comm.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(s),
                                     std::forward_as_tuple(team)
    );
}

void start_comm_thread(Thread_team *team) {
    assert(team != nullptr);
    gtfxx_context.th_comm.team = team;

    // Master thread will not be responsible for making progress on communications
    release_master_thread();

    gtfxx_context.th_comm.start();
}

Task_flow &map_task(const string &s) {
    auto search = gtfxx_context.m_map_task.find(s);
    if (search != gtfxx_context.m_map_task.end()) {
        return search->second;
    } else {
        assert(false);
    }
    // This line should not be reached
    return gtfxx_context.empty_task;
}

Channel &map_comm(const string &s) {
    auto search = gtfxx_context.m_map_comm.find(s);
    if (search != gtfxx_context.m_map_comm.end()) {
        return search->second;
    } else {
        assert(false);
    }
    return gtfxx_context.empty_comm;
}

// Spawn task
void Channel::async(int3 idx, Task *a_tsk) {

    // Increment the team task counter
    ++(team->ntasks);

    // Spawn the comm task
    gtfxx_context.th_comm.spawn(a_tsk);

    // Delete entry in promise_map
    std::unique_lock<std::mutex> lck(mtx_graph);
    assert(promise_map.find(idx) != promise_map.end());
    promise_map.erase(idx);
}

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

// TODO: move previous classes inside the gtfxx namespace
namespace gtfxx {

    template<typename... T>
    struct comm {

        std::tuple<T...> tuple_;
        upcxx::intrank_t dest;

        explicit comm(std::tuple<T...> a_tuple) : tuple_(std::move(a_tuple)), dest(-1) {}

        comm &to_rank(upcxx::intrank_t a_dest) {
            dest = a_dest;
            return *this;
        }

        template<typename Fn>
        void on_receive(Fn &&a_tsk_) {
            rpc_ff_tuple(a_tsk_,
                         upcxx::make_index_sequence<std::tuple_size<std::tuple<T...> >::value>()
            );
        }

        template<typename Fn, int... i>
        void rpc_ff_tuple(Fn &a_tsk_, upcxx::index_sequence<i...>) {
            upcxx::rpc_ff(dest, std::forward<Fn>(a_tsk_), std::forward<T>(std::get<i>(tuple_))...);
        }
    };

    template<typename T, typename Iter = T *>
    struct memblock {
        typedef upcxx::view<T, Iter> iterator;
    };

    template<typename T, typename Iter = T *>
    upcxx::view<T, Iter> memblock_view(std::size_t n, Iter begin) {
        Iter end = begin + n;
        return {static_cast<Iter &&>(begin), static_cast<Iter &&>(end), n};
    }

    template<typename... T>
    comm<T...> send(T... a_msg) {
        return comm<T...>(std::tuple<T...>(a_msg...));
    }
}

int64_t ans = -1;

TEST(UPCXX, Basic) {
    // -----------------
    // Simple upc++ test

    const int64_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();
    const upcxx::intrank_t dest = n_rank - 1 - my_rank;

    Vector msg(1000);
    int64_t dummy = 1 + my_rank;

    int64_t expected = 0;
    for (int i = 0; i < msg.size(); ++i) {
        if (i % 2) {
            msg[i] = i;
        } else {
            msg[i] = (647 * i) % 1000;
        }
        expected += msg[i];
    }

    expected += 1 + dest;

    ans = -1;

    gtfxx::send(gtfxx::memblock_view<int64_t>(msg.size(), &msg[0]), dummy)
            .to_rank(dest)
            .on_receive([](
                    gtfxx::memblock<int64_t>::iterator msg_, int64_t dummy) {
                assert(dummy == upcxx::rank_n() - upcxx::rank_me());
                int64_t sum = 0;
                auto it = msg_.begin();
                for (; it != msg_.end(); ++it) {
                    sum += *it;
                }
                ans = sum + dummy;
            });

    capture_master_thread();

    while (ans != expected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        upcxx::progress();
    }

    ASSERT_EQ(ans, expected);

    upcxx::barrier();
}

TEST(UPCXX, ThreadCommScalar) {
    // -----------------
    // Using a comm thread and upc++ to send a scalar

    const int64_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();
    const upcxx::intrank_t dest = n_rank - 1 - my_rank;

    const int64_t payload = my_rank;
    const int64_t expected = dest;

    ans = -1;

    auto task_thread_comm = [my_rank, dest, payload, expected]() {
        // Acquiring master persona
        upcxx::persona_scope scope(upcxx::master_persona());

        upcxx::rpc_ff(dest,
                      [](int64_t payload) {
                          assert(payload == upcxx::rank_n() - 1 - upcxx::rank_me());
                          ans = payload;
                      }, payload);

        while (ans != expected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            upcxx::progress();
        }
    };

    release_master_thread();

    auto th_ = thread(task_thread_comm);

    th_.join();

    ASSERT_EQ(ans, expected);

    upcxx::barrier();
}

TEST(UPCXX, ThreadCommVector) {
    // -----------------
    // Using a comm thread and upc++ to send a vector

    const int64_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();
    const upcxx::intrank_t dest = n_rank - 1 - my_rank;

    Vector msg(10000);

    int64_t my_rank_send = my_rank;

    int64_t expected = 0;
    for (int i = 0; i < msg.size(); ++i) {
        if (i % 2) {
            msg[i] = i;
        } else {
            msg[i] = 2 * i;
        }
        expected += msg[i];
    }

    msg[0] += my_rank;

    expected += dest + dest;

    ans = -1;

    auto task_thread_comm = [=]() {
        // Acquiring master persona
        upcxx::persona_scope scope(upcxx::master_persona());

        upcxx::rpc_ff(dest,
                      [=](upcxx::view<int64_t> msg_, int64_t my_rank_send) {
                          assert(my_rank_send == upcxx::rank_n() - 1 - upcxx::rank_me());
                          int sum = 0;
                          for (auto it : msg_) {
                              sum += it;
                          }
                          ans = sum + my_rank_send;
                      }, upcxx::make_view(msg.begin(), msg.end()), my_rank_send);

        while (ans != expected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            upcxx::progress();
        }
    };

    release_master_thread();

    auto th_ = thread(task_thread_comm);

    th_.join();

    ASSERT_EQ(ans, expected);

    upcxx::barrier();
}

TEST(UPCXX, GTFVector) {
    // -----------------
    // Using a comm thread and gtf++ to send a vector

    const int64_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();
    const upcxx::intrank_t dest = n_rank - 1 - my_rank;

    Vector msg(1000000);

    int64_t my_rank_send = my_rank;

    int64_t expected = 0;
    for (int i = 0; i < msg.size(); ++i) {
        msg[i] = 1 + i % 3;
        expected += msg[i];
    }

    msg[0] += my_rank;

    expected += dest + dest;

    int local_ans = -1;

    auto task_thread_comm = [=, &local_ans]() {
        // Acquiring master persona
        upcxx::persona_scope scope(upcxx::master_persona());

        gtfxx::send(gtfxx::memblock_view<int64_t>(msg.size(), &msg[0]), my_rank_send)
                .to_rank(dest)
                .on_receive([=, &local_ans](gtfxx::memblock<int64_t>::iterator msg_, int64_t my_rank_send) {
                    assert(my_rank_send == upcxx::rank_n() - 1 - upcxx::rank_me());
                    int sum = 0;
                    for (auto it : msg_) {
                        sum += it;
                    }
                    local_ans = sum + my_rank_send;
                });

        while (local_ans != expected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            upcxx::progress();
        }
    };

    release_master_thread();

    auto th_ = thread(task_thread_comm);

    th_.join();

    ASSERT_EQ(local_ans, expected);

    upcxx::barrier();
}

TEST(gtfxx, UPCXX) {

    const int64_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();
    const int n_thread = 4; // number of threads to use

    LOG("[main] n_rank " << n_rank << "; my_rank " << my_rank <<
                         "; n_thread " << n_thread);

    profiler.open("prof.out");

    // Create thread team
    Thread_team team(n_thread);

    task_emplace("map", n_thread, 1, 1, &team);
    comm_emplace("send", &team);
    task_emplace("reduce", 1, 1, 1, &team);

    vector<int> data(n_thread * n_rank, -1);

    auto compute_on_i = [=](int3 &idx) {
        return idx[0] % n_thread;
    };

    map_task("map")
            .dependency_count([](const int3 idx) { return 0; })
            .define_task(
                    [n_rank, n_thread, my_rank, &data](const int3 idx) {
                        const int i = idx[0];
                        assert(i >= 0 && i < n_thread);

                        const int offset = my_rank * n_thread;
                        const int global_comm_idx = i + offset;
                        assert(global_comm_idx >= 0 && global_comm_idx < n_thread * n_rank);

                        data[global_comm_idx] = 1;

                        map_comm("send").fulfill_promise({global_comm_idx, 0, 0});
                    })
            .compute_on(compute_on_i);

    // Need to be defined on the sending rank only
    map_comm("send")
            .dependency_count([](const int3 idx) { return 1; })
            .define_task([n_rank, n_thread, my_rank, &data](const int3 global_comm_idx) {
                assert(data[global_comm_idx[0]] == 1);
                assert(my_rank >= 0 && my_rank < n_rank);

                // Comms executed by the comm thread
                for (int i = 0; i < n_rank; ++i) {
                    if (i != my_rank) {
                        gtfxx::send(global_comm_idx, data[global_comm_idx[0]])
                                .to_rank(i)
                                .on_receive([n_rank, n_thread, &data](int3 global_comm_idx, int d) {
                                    assert(global_comm_idx[0] >= 0 && global_comm_idx[0] < n_thread * n_rank);
                                    assert(d == 1);
                                    assert(data[global_comm_idx[0]] == -1);

                                    data[global_comm_idx[0]] = d;
                                    map_comm("send").finalize(global_comm_idx);
                                    //map_task("reduce").fulfill_promise({0, 0, 0});
                                });
                    } else {
                        map_task("reduce").fulfill_promise({0, 0, 0});
                    }

                }
            });

    /* Will run on the receiving rank but needs to be defined on the sending rank only.
     * All captured variables correspond to memory locations on the receiving rank.
     * Arguments correspond to the payload defined on the sending rank.
     * The payload is communicated over the network. */
    map_comm("send").set_finalize([](int3 global_comm_idx) {
        map_task("reduce").fulfill_promise({0, 0, 0});
    });

    int local_sum;
    atomic_bool done(false);

    map_task("reduce")
            .dependency_count([n_thread, n_rank](const int3 idx) { return n_thread * n_rank; })
            .define_task([&data, &local_sum, &done](const int3 idx) {
                const int i = idx[0];
                assert(i == 0);

                done.store(true); // This is the last task that we need to run
                local_sum = 0;
                for (auto d : data) {
                    local_sum += d;
                    assert(d == 1);
                }
            })
            .compute_on(compute_on_i);

    // Start team of threads
    team.start();

    // TODO: we need a safer way to make sure the comm thread has been started
    start_comm_thread(&team); // Start communication thread

    profiler.map_team_threads(team);

    // Create seed tasks and start
    for (int i = 0; i < n_thread; ++i) {
        map_task("map").async({i, 0, 0});
    }

    // Because of the communications, detecting quiescence is not easy
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Wait for end of task queue execution
    team.join();

    upcxx::barrier();

    ASSERT_EQ(local_sum, n_thread * n_rank);

    profiler.dump();
}

int main(int argc, char **argv) {

    upcxx::init();

    ::testing::InitGoogleTest(&argc, argv);

    const int return_flag = RUN_ALL_TESTS();

    upcxx::finalize();

    return return_flag;
}
