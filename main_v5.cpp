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

    // TODO: this should be made a template parameter
    bool m_delete = false;
    // whether the object should be deleted after running the function
    // to completion.

    // TODO: move to Task_flow
    std::atomic_int m_wait_count; // number of dependencies

    Task() : m_wait_count(-1) {}
    // m_wait_count must be initialized at -1

    void set_function(const Base_task &a_f, float a_priority = 0.0, bool a_del = false) {
        Base_task::operator=(a_f);
        m_priority = a_priority;
        m_delete = a_del;
    }

    Task &operator=(const Base_task &a_f) {
        Base_task::operator=(a_f);
        return *this;
    }
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

    void join() {
        m_stop.store(true);
        for (auto &th : v_thread) th.join();
    }

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
            if (tsk->m_delete) delete tsk;

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

void Profiler::map_team_threads(Thread_team &team) {
    for (auto &th_prio : team.v_thread) {
        thread_id_map[id_to_string(th_prio.th.get_id())] = th_prio.m_id;
    }
    thread_id_map[id_to_string(std::this_thread::get_id())] = -1;
}

struct Matrix3_task : vector<Task> {
    int n0, n1, n2;

    Matrix3_task() : n0(0), n1(0), n2(0) {};

    Matrix3_task(int a_n0, int a_n1, int a_n2) : vector<Task>(a_n0 * a_n1 * a_n2),
                                                 n0(a_n0), n1(a_n1), n2(a_n2) {};

    Task &operator()(int i, int j, int k) {
        assert(i >= 0 && i < n0);
        assert(j >= 0 && j < n1);
        assert(k >= 0 && k < n2);
        return operator[](i + n0 * (j + n1 * k));
    }
};

typedef array<int, 3> int3;

struct Dependency_flow {
    typedef std::function<int(int3 &, Task *)> Init_task;

    // Thread team
    Thread_team *team = nullptr;

    explicit Dependency_flow(Thread_team *a_team) : team(a_team) {}

    // How to initialize a task
    Init_task m_init;

    virtual ~Dependency_flow() = default;

    // Decrement the dependency counter and spawn task if ready
    virtual void decrement_wait_count(int3) = 0;
};

struct Task_flow : public Dependency_flow {

    /* TODO: we don't need to store the function
     * We just need to know what function to call for a given int3.
     * Then this function can be enqueued directly to a thread.
     * Only one atomic for dependencies needs to be stored. */
    typedef std::function<int(int3 &)> Map_task;

    // Task are indexed using a sparse 3D grid
    Matrix3_task task_grid;

    // Mapping from task index to thread id
    Map_task m_map;

    Task_flow &set_task_init(Init_task a_init) {
        m_init = std::move(a_init);
        return *this;
    }

    void compute_on(Map_task a_map) {
        m_map = std::move(a_map);
    }

    Task_flow(Thread_team *a_team, int n0, int n1, int n2) :
            Dependency_flow(a_team), task_grid(n0, n1, n2) {}

    // spawn a task
    void async(int3);

    void async(int3, Task *);

    // Decrement the dependency counter and spawn task if ready
    void decrement_wait_count(int3) override;
};

// Spawn task
void Task_flow::async(int3 idx, Task *a_tsk) {
    team->spawn(/*task map*/ m_map(idx), a_tsk);
}

// Initialize task and spawn it
void Task_flow::async(int3 idx) {
    auto t_ = &task_grid(idx[0], idx[1], idx[2]);
    m_init(idx, t_);
    async(idx, t_);
}

void Task_flow::decrement_wait_count(int3 idx) {
    /* It's a bug to call this function more times than
     * wait_count returned by m_init(). */

    assert(0 <= idx[0] && idx[0] < task_grid.n0);
    assert(0 <= idx[1] && idx[1] < task_grid.n1);
    assert(0 <= idx[2] && idx[2] < task_grid.n2);

    auto t_ = &task_grid(idx[0], idx[1], idx[2]);

    // Decrement counter
    int wait_count = std::atomic_fetch_sub(&(t_->m_wait_count), 1);

    if (wait_count == -1) { // Uninitialized task
        int wait_count_init = m_init(idx, t_);
        wait_count = std::atomic_fetch_add(&(t_->m_wait_count), wait_count_init);
        // wait_count = value before add operation
        wait_count += wait_count_init + 1;
    }

    if (wait_count == 0) { // task is ready to run
        async(idx, t_);
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
    std::unique_lock<std::mutex> lck(a_thread->mtx);

    // Acquiring master persona
    upcxx::persona_scope scope(upcxx::master_persona());

    LOG("entered while loop in spin_comm on rank " << upcxx::rank_me());

    while (true) {

        while (!a_thread->ready_queue.empty()) {
            Task *tsk = a_thread->pop_unsafe();
            lck.unlock();

            LOG("running comm task on rank " << upcxx::rank_me());
            (*tsk)(); // Issue active message

            --(a_thread->team->ntasks);
            if (tsk->m_delete) delete tsk;

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
    // TODO: check error reports by CLion
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

struct Comm_flow : public Dependency_flow {

    typedef std::function<void(int3 &)> Finalize_task;

    // How to unpack after a data transfer is complete
    Finalize_task m_finalize;

    unordered_map<int3, Task *, hash_array::hash> task_map;
    mutex mtx_graph; // Used to concurrently make changes to task_map

    void set_pack(Init_task a_init) {
        m_init = std::move(a_init);
    }

    void set_finalize(Finalize_task a_finalize) {
        LOG("[set_finalize]");
        m_finalize = std::move(a_finalize);
    }

    void finalize(int3 &idx) {
        LOG("[finalize] calling m_finalize with idx " << idx[0]);
        m_finalize(idx);
    }

    explicit Comm_flow(Thread_team *a_team) : Dependency_flow(a_team) {}

    // Find a task in task_map and return a pointer
    Task *find_task(int3 &);

    // Enqueue a communication
    void async(int3);

    void async(int3, Task *);

    // Decrement the dependency counter and spawn task if ready
    void decrement_wait_count(int3) override;
};

Task *Comm_flow::find_task(int3 &idx) {
    Task *t_ = nullptr;

    std::unique_lock<std::mutex> lck(mtx_graph);

    auto tsk = task_map.find(idx);

    // Task exists
    if (tsk != task_map.end()) {
        t_ = tsk->second;
    } else {
        // Task does not exist; create it
        t_ = new Task;
        task_map[idx] = t_; // Insert in task_map
    }

    lck.unlock();

    assert(t_ != nullptr);

    return t_;
}

// Initialize task and spawn it
void Comm_flow::async(int3 idx) {
    auto t_ = find_task(idx);
    m_init(idx, t_);
    async(idx, t_);
}

/* TODO: decrement_wait_count is nearly the same as Task_flow.
 * Need to make this part of the base class.
 * The only difference is find_task. */
void Comm_flow::decrement_wait_count(int3 idx) {
    auto t_ = find_task(idx);

    // Decrement counter
    int wait_count = std::atomic_fetch_sub(&(t_->m_wait_count), 1);

    if (wait_count == -1) { // Uninitialized task
        int wait_count_init = m_init(idx, t_) + 1;
        wait_count = std::atomic_fetch_add(&(t_->m_wait_count), wait_count_init);
        // wait_count = value before add operation
        wait_count += wait_count_init;
    }

    if (wait_count == 0) { // task is ready to run
        async(idx, t_);
    }
}

// Task flow context
struct Context {
    map<string, Task_flow> m_map_task;
    map<string, Comm_flow> m_map_comm;

    Thread_comm th_comm;

    Task_flow empty_task{nullptr, 0, 0, 0};
    Comm_flow empty_comm{nullptr};

} gtfxx_context;

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

void start_comm_thread(Thread_team * team) {
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

Comm_flow &map_comm(const string &s) {
    auto search = gtfxx_context.m_map_comm.find(s);
    if (search != gtfxx_context.m_map_comm.end()) {
        return search->second;
    } else {
        assert(false);
    }
    return gtfxx_context.empty_comm;
}

// Spawn task
void Comm_flow::async(int3 idx, Task *a_tsk) {
    ++(team->ntasks);
    LOG("[async] spawn comm");
    gtfxx_context.th_comm.spawn(a_tsk);

    LOG("[async] DONE");

    // Delete entry in task_map
    std::unique_lock<std::mutex> lck(mtx_graph);
    assert(task_map.find(idx) != task_map.end());
    task_map.erase(idx);

    LOG("[async] task comm deleted");
}

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

//typedef std::vector<int64_t, custom_allocator<int64_t> > Vector;
typedef std::vector<int64_t> Vector;

// TODO: move previous classes inside the gtfxx namespace
namespace gtfxx {

    template<typename... T>
    struct comm {

        std::tuple<T...> tuple_;
        upcxx::intrank_t dest;

        comm(std::tuple<T...> a_tuple) : tuple_(std::move(a_tuple)), dest(-1) {}

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
            msg[i] = (647 * i)%1000;
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

    assert(ans == expected);

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
    for (int i=0; i<msg.size(); ++i) {
        if (i%2) {
            msg[i] = i;
        } else {
            msg[i] = 2*i;
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
                      }, upcxx::make_view(msg.begin(),msg.end()), my_rank_send);

        while (ans != expected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            upcxx::progress();
        }
    };

    release_master_thread();

    auto th_ = thread(task_thread_comm);

    th_.join();

    assert(ans == expected);

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
    for (int i=0; i<msg.size(); ++i) {
        msg[i] = 1 + i%3;
        expected += msg[i];
    }

    msg[0] += my_rank;

    expected += dest + dest;

    int local_ans = -1;

    auto task_thread_comm = [=,&local_ans]() {
        // Acquiring master persona
        upcxx::persona_scope scope(upcxx::master_persona());

        gtfxx::send(gtfxx::memblock_view<int64_t>(msg.size(), &msg[0]), my_rank_send)
                .to_rank(dest)
                .on_receive([=,&local_ans](gtfxx::memblock<int64_t>::iterator msg_, int64_t my_rank_send) {
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

    assert(local_ans == expected);

    upcxx::barrier();
}

TEST(gtfxx, UPCXX) {

    const int64_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();

    LOG("[main] n_rank " << n_rank << " my_rank " << my_rank);

    const int n_thread = 8; // number of threads to use

    profiler.open("prof.out");

    LOG("[main] Initializing");

    // Create thread team
    Thread_team team(n_thread);

    task_emplace("map", n_thread, 1, 1, &team);
    comm_emplace("send", &team);
    task_emplace("reduce", 1, 1, 1, &team);

    vector<int> data(n_thread * n_rank, -1);

    auto compute_on_i = [=](int3 &idx) {
        return idx[0] % n_thread;
    };

    map_task("map").set_task_init([=, &data](int3 &idx, Task *tsk) -> int {
                const int i = idx[0];
                assert(i >= 0 && i < n_thread);
                tsk->set_function([=, &data]() {
                    const int offset = my_rank * n_thread;
                    const int global_comm_idx = i + offset;
                    assert(global_comm_idx >= 0 && global_comm_idx < n_thread * n_rank);
                    LOG("[map init] global_comm_idx " << global_comm_idx << " i " <<
                                                      i << " my_rank " << my_rank);
                    data[global_comm_idx] = 1;
                    map_comm("send").decrement_wait_count({global_comm_idx, 0, 0});
                });
                return 0; // wait_count
            })
            .compute_on(compute_on_i);

    // Need to be defined on the sending rank only
    map_comm("send").set_pack([=, &data](int3 &global_comm_idx, Task *tsk) -> int {
        const int i = global_comm_idx[0] % n_thread;
        LOG("[send] from thread " << i << " rank " << my_rank <<
            " global index " << global_comm_idx[0]);

        assert(data[global_comm_idx[0]] == 1);

        tsk->set_function( [=, &data] () {
            // Comm function that is executed on the comm thread

//            upcxx::rpc_ff(dest,
//                    [=,&data](int3 global_comm_idx, int d_) {
//                    LOG("[on_receive] with " << global_comm_idx[0]);
//                    assert(global_comm_idx[0] >= 0 && global_comm_idx[0] < n_thread * n_rank);
//                    assert(d_ == 1);
//                    assert(data[global_comm_idx[0]] == -1);
//                    data[global_comm_idx[0]] = d_;
//                    map_comm("send").finalize(global_comm_idx);
//                    }, global_comm_idx, data[global_comm_idx[0]]);

            for (int i=0; i<n_rank; ++i) {
                if (i != my_rank) {
                    gtfxx::send(global_comm_idx, data[global_comm_idx[0]])
                            .to_rank(i)
                            .on_receive([=, &data](int3 global_comm_idx, int d) {
                                assert(global_comm_idx[0] >= 0 && global_comm_idx[0] < n_thread * n_rank);
                                assert(d == 1);
                                assert(data[global_comm_idx[0]] == -1);

                                data[global_comm_idx[0]] = d;
                                map_comm("send").finalize(global_comm_idx);
                            });
                }
            }
        });

        return 1; // wait_count
    });

    /* Will run on the receiving rank but needs to be defined on the sending rank only.
     * All captured variables correspond to memory locations on the receiving rank.
     * Arguments correspond to the payload defined on the sending rank.
     * The payload is communicated over the network. */
    map_comm("send").set_finalize([](int3 &global_comm_idx) {
        map_task("reduce").decrement_wait_count({0, 0, 0});
    });

    int local_sum;

    atomic_bool done(false);

    map_task("reduce").set_task_init([=, &data, &local_sum, &done](int3 &idx, Task *tsk) -> int {
                const int i = idx[0];
                assert(i == 0);
                LOG("[reduce] init with idx " << idx[0]);
                tsk->set_function([i, &data, &local_sum, &done]() {
                    assert(i == 0);
                    LOG("[REDUCE] " << i);
                    done.store(true); // This is the last task that we need to run
                    local_sum = 0;
                    for (auto d : data) {
                        local_sum += d;
                        assert(d == 1);
                    }
                });
                return n_thread * (n_rank-1); // wait_count
            })
            .compute_on(compute_on_i);

    LOG("Start");

    // Start team of threads
    team.start();

    // TODO: we need a safer way to make sure the comm thread has been started
    start_comm_thread(&team); // Start communication thread

    profiler.map_team_threads(team);

    LOG("Create seed tasks");

    // Create seed tasks and start
    for (int i = 0; i < n_thread; ++i) {
        map_task("map").async({i, 0, 0});
    }

    // Because of the communications detecting quiescence is not easy
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Wait for end of task queue execution
    team.join();

    upcxx::barrier();

    LOG("Done with calculation");

    printf("On rank %d, sum = %d\n", my_rank, local_sum);

    assert(local_sum == n_thread * n_rank);

    profiler.dump();
}

int main(int argc, char **argv) {

    upcxx::init();

    ::testing::InitGoogleTest(&argc, argv);

    const int return_flag = RUN_ALL_TESTS();

    upcxx::finalize();

    return return_flag;
}
