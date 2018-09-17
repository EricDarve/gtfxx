//
// Created by Eric Darve on 9/16/18.
//

#ifndef GTFXX_GTFXX_H
#define GTFXX_GTFXX_H

#include "util.h"

// ---------------------
// UPC++ function calls

void active_message_progress();

void capture_master_thread();

void release_master_thread();

// Base_task is a simple function with no argument
typedef std::function<void()> Base_task;

// Tasks are functions executed by threads
struct Task : public Base_task {
    float m_priority = 0.0; // task priority
    explicit Task(const Base_task &&a_f, float a_priority = 0.0);
};

// Thread with priority queue

// Task comparison based on their priorities
struct Task_comparison {
public:
    // Comparison () operator
    bool operator()(const Task *a_lhs, const Task *a_rhs) const;
};

// Thread with priority queue
struct Thread_prio {
    Threadpool *th_pool;
    unsigned short m_id; // thread id in a thread pool

    std::priority_queue<Task *, vector<Task *>, Task_comparison> ready_queue;
    mutex mtx; // Used to manage access to the priority queue

    thread th; // thread that will execute the tasks

    // For optimization to avoid testing ready_queue.empty() in some cases
    std::atomic_bool m_empty;

    Thread_prio();
    ~Thread_prio();

    // Thread starts executing the function spin()
    void start();

    // Add new task to queue; thread safe
    void spawn(Task *);

    // Not thread safe
    Task *pop_unsafe();

    // join() the thread
    void join();
};

// ----------------------------------
// Progress thread for communications

struct Thread_comm {
    Threadpool *th_pool = nullptr;
    queue<Task *> ready_queue;
    thread th;
    mutex mtx;
    std::atomic_bool m_empty;

    Thread_comm();
    ~Thread_comm();

    void start();

    // Add new task to queue; thread safe
    void spawn(Task *a_t);

    // Not thread safe
    Task *pop_unsafe();

    // join() the thread
    void join();
};

// -----------
// Thread pool

struct Threadpool : public vector<Thread_prio *> {
    vector<Thread_prio> v_thread;
    unsigned long n_query_spawn = 4;  // Optimization parameter
    unsigned long n_query_steal = 16; // Optimization parameter
    std::atomic_int n_tasks; // Number of ready tasks in any thread queue
    std::atomic_bool m_stop;

    explicit Threadpool(const int n_thread);

    void start();

    void join();

    void spawn(const int a_id, Task *a_task);

    void steal(unsigned short a_id);
};

// ------------------------
// Base class for task flow

typedef array<int, 3> int3;

struct Dependency_flow {
    typedef std::function<int(const int3)> Dependency_count;
    typedef std::function<void(const int3)> Build_task;

    // Thread pool
    Threadpool *th_pool;

    // How to calculate the number of dependencies for promises
    Dependency_count m_dep_count;

    // How to construct a task
    Build_task m_build_task;

    Dependency_flow();
    explicit Dependency_flow(Threadpool *a_pool);

    virtual ~Dependency_flow() = default;

    // Decrement the dependency counter and spawn task if ready
    virtual void fulfill_promise(int3) = 0;
};

// ------------
// Promise grid

typedef std::atomic_int Promise;

struct Matrix3_promise : vector<Promise> {
    int n0, n1, n2;

    Matrix3_promise();

    // Initialize all promises with -1
    Matrix3_promise(int a_n0, int a_n1, int a_n2);

    std::atomic_int &operator()(int i, int j, int k);
};

// -----------------------------------
// Task flow for computational kernels

struct Task_flow : public Dependency_flow {

    typedef std::function<int(int3 &)> Map_task;

    // Tasks are indexed using a 3D grid of atomic dependencies
    Matrix3_promise promise_grid;

    // Mapping from task index to thread id
    Map_task m_map;

    Task_flow();
    Task_flow(Threadpool *a_pool, int n0, int n1, int n2);

    // Defines the number of dependencies
    Task_flow &dependency_count(Dependency_count f);

    // Defines the task to be run asynchronously
    Task_flow &define_task(Build_task f);

    // Which thread should execute the task
    void compute_on(Map_task a_map);

    // Spawn a task from index
    void async(int3);

    // Spawn a task that is already initialized
    void async(int3, Task *);

    // Decrement the dependency counter and spawn task if ready
    void fulfill_promise(int3) override;
};

// -----------------
// Channel class: used to communicate data between nodes
// using active messages

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
    Finalize_task m_finalize;

    unordered_map<int3, std::atomic_int, hash_array::hash> promise_map;
    mutex mtx_graph; // Used to concurrently make changes to task_map

    Channel();
    explicit Channel(Threadpool *a_team);

    // Defines the number of dependencies
    Channel &dependency_count(Dependency_count f);

    // Defines the communication channel to run
    void define_task(Build_task f);

    // Defines the callback function.
    // The callback function may be called after communication completes.
    void set_finalize(Finalize_task a_finalize);

    // Run the m_finalize() callback function defined in set_finalize()
    // This function may be called in the active message.
    void finalize(int3 idx);

    // Find a promise in promise_map and return a pointer
    Promise *find_promise(int3 &);

    // Enqueue a communication from index
    void async(int3);

    // Enqueue a communication that is already initialized
    void async(int3, Task *);

    // Decrement the dependency counter and enqueue communication if ready
    void fulfill_promise(int3) override;
};

// -----------------
// Task flow context

struct Context {
    map<string, Task_flow> m_map_task;
    map<string, Channel> m_map_comm;

    Thread_comm th_comm;

    Task_flow empty_task;
    Channel empty_comm;

    Context();
};

#endif //GTFXX_GTFXX_H
