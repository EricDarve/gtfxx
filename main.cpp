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
#include <array>
#include <unordered_map>
#include <vector>

#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>

#include "gtest/gtest.h"

using namespace std::chrono;

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;

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

// Builds a logging message and outputs it in the destructor
class LogMessage {
public:
    LogMessage(const char * file, const char * function, int line)
    {
        os << file << ":" << line << " (" << function << ") ";
    }

    // output operator
    template<typename T>
    LogMessage & operator<<(const T & t)
    {
        os << t;
        return *this;
    }

    ~LogMessage()
    {
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
    enum event {duration,timepoint};

    std::string name;
    std::string thread_id;
    std::chrono::high_resolution_clock::time_point m_start, m_stop;
    event ev_type;

    Profiling_Event(char* c, std::string& t_id) : name(c), thread_id(t_id)
    {
    }

    void timestamp()
    {
        ev_type = timepoint;
        m_start = std::chrono::high_resolution_clock::now();
    }

    void start()
    {
        ev_type = duration;
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        m_stop = std::chrono::high_resolution_clock::now();
    }
};

struct Thread_team;

std::string id_to_string(std::thread::id id)
{
    std::stringstream s_id;
    s_id << id;
    return s_id.str();
}

std::string get_thread_id()
{
    return id_to_string( std::this_thread::get_id() );
}

struct Profiler {
    list<Profiling_Event> events;
    std::mutex mtx;
    std::map<std::string, int> thread_id_map;
    std::string file_name{"prof.out"};

    void map_team_threads(Thread_team &);
    
    void open(char * c) {
        // Open a new file and clear up the list of events
        file_name = std::string(c);
        events.clear();
    }

    void timestamp(char* c)
    {
        auto thread_id = get_thread_id();
        Profiling_Event prof_ev(c, thread_id);
        prof_ev.timestamp();
        {
            std::lock_guard<std::mutex> lck(mtx);
            events.push_back(prof_ev);
        }
    }

    Profiling_Event* start(char *c)
    {
        auto thread_id = get_thread_id();
        Profiling_Event* prof_ev = new Profiling_Event(c, thread_id);
        prof_ev->start();
        return prof_ev;
    }

    void stop(Profiling_Event* prof_ev)
    {
        prof_ev->stop();
        {
            std::lock_guard<std::mutex> lck(mtx);
            events.push_back(*prof_ev);
        }
        delete prof_ev;
    }

    void dump()
    {
        FILE* fp = std::fopen(file_name.c_str(), "w");
        if(!fp) {
            std::perror("File opening failed");
            return;
        }

        fprintf(fp,"nthreads %d\n",thread_id_map.size());
        for (auto it = thread_id_map.begin(); it != thread_id_map.end(); ++it) {
            fprintf(fp,"tidmap %s %d\n",it->first.c_str(),it->second);
        }

        for (auto it = events.begin(); it != events.end(); ++it) {
            auto d_start = std::chrono::duration<long long, std::nano>(it->m_start.time_since_epoch());
            switch (it->ev_type) {
            case Profiling_Event::duration :
                auto d_end   = std::chrono::duration<long long, std::nano>(it->m_stop.time_since_epoch());
                fprintf(fp,"tid %s start %ld end %ld name %s\n",it->thread_id.c_str(),
                        d_start.count(),d_end.count(),it->name.c_str());
                break;
            case Profiling_Event::timepoint :
                fprintf(fp,"tid %s timestamp %ld name %s\n",it->thread_id.c_str(),
                        d_start.count(),it->name.c_str());
                break;
            }
        }

        std::fclose(fp);
    }
} profiler;

struct Task : public Base_task {
    float m_priority = 0.0; // task priority

    bool m_delete = false;
    // whether the object should be deleted after running the function
    // to completion.

    std::atomic_int m_wait_count; // number of dependencies

    Task()
    {
        m_wait_count.store(-1); // Must be initialized at -1
    }

    ~Task() {}

    void set_function(Base_task a_f, float a_priority = 0.0, bool a_del = false)
    {
        Base_task::operator=(a_f);
        m_priority = a_priority;
        m_delete = a_del;
    }

    void operator=(Base_task a_f)
    {
        Base_task::operator=(a_f);
    }
};

// Task comparison based on their priorities
struct Task_comparison {
public:
    bool operator() (const Task* a_lhs, const Task* a_rhs) const
    {
        return (a_lhs->m_priority < a_rhs->m_priority);
    };
};

struct Thread_prio;
void spin(Thread_prio * a_thread);

// Thread with priority queue management
struct Thread_prio {
    Thread_team * team;
    unsigned short m_id;

    std::priority_queue<Task*, vector<Task*>, Task_comparison> ready_queue;
    thread th;
    mutex mtx;
    std::condition_variable cv;
    std::atomic_bool m_empty;
    // For optimization to avoid testing ready_queue.empty() in some cases

    // Thread starts executing the function spin()
    void start()
    {
        m_empty.store(true);
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    void spawn(Task * a_t)
    {
        std::lock_guard<std::mutex> lck(mtx);
        ready_queue.push(a_t); // Add task to queue
        m_empty.store(false);
        cv.notify_one(); // Wake up thread
    };

    Task* pop()
    {
        if (ready_queue.size() == 1) m_empty.store(true);
        Task* tsk = ready_queue.top();
        ready_queue.pop();
        return tsk;
    };

    void notify()
    {
        std::lock_guard<std::mutex> lck(mtx);
        cv.notify_one(); // Wake up thread
    }

    // join() the thread
    void join()
    {
        if (th.joinable()) {
            th.join();
        }
    }

    ~Thread_prio()
    {
        join();
    }
};

struct Thread_team : public vector<Thread_prio*> {
    vector<Thread_prio> v_thread;
    unsigned long n_query_spawn = 4;  // Optimization parameter
    unsigned long n_query_steal = 32; // Optimization parameter
    std::atomic_int ntasks; // number of ready tasks in any thread queue
    std::atomic_bool m_stop;

    Thread_team(const int n_thread) : v_thread(n_thread)
    {
        for (int i=0; i<n_thread; ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = static_cast<unsigned short>(i);
        }
        ntasks.store(0);
        m_stop.store(false);
    }

    void start()
    {
        ntasks.store(0);
        m_stop.store(false);
        for (auto& th : v_thread) th.start();
    }

    void join()
    {
        m_stop.store(true);
        if (ntasks.load() == 0) {
            wake_all();
        }
        for (auto& th : v_thread) th.join();
    }

    void wake_all()
    {
        if (atomic_fetch_sub(&ntasks,1) == 0) {
            for (auto& th : v_thread) th.notify();
        }
    }

    void spawn(const int a_id, Task * a_task)
    {
        assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());
        int id_ = a_id;
        // Check if queue is empty
        if (!v_thread[a_id].m_empty.load()) {
            // Check whether other threads have empty queues
            const unsigned long n_query = min(1+n_query_spawn, v_thread.size());
            for (unsigned long i=a_id+1; i<a_id+n_query; ++i) {
                auto j = i%v_thread.size();
                if (v_thread[j].m_empty.load()) {
                    id_ = static_cast<int>(j);
                    break;
                }
            }
        }
        char timestamp_message[80];
        if (id_ == a_id) {
            sprintf(timestamp_message,"spawn_self %d",a_id);
        }
        else {
            sprintf(timestamp_message,"spawn_other %d to %d",a_id,id_);
        }
        profiler.timestamp(timestamp_message);
        // Note that because we are not using locks, the queue may no longer be empty.
        // But this implementation is more efficient than using locks.
        ++ntasks;
        v_thread[id_].spawn(a_task);
    }

    void steal(unsigned short a_id)
    {
        const unsigned long n_query = min(n_query_steal, v_thread.size());
        for (unsigned long i=a_id+1; i<a_id+n_query; ++i) {
            auto j = i%v_thread.size();
            Thread_prio & thread_ = v_thread[j];
            if (!thread_.m_empty.load()) {
                std::unique_lock<std::mutex> lck(thread_.mtx);
                if (!thread_.ready_queue.empty()) {
                    profiler.timestamp("steal");
                    Task * tsk = thread_.pop();
                    lck.unlock();
                    v_thread[a_id].spawn(tsk);
                    break;
                }
            }
        }
    }
};

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin(Thread_prio * a_thread)
{
    auto pe = profiler.start("overhead");
    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {
        while (!a_thread->ready_queue.empty()) {
            Task * tsk = a_thread->pop();
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

        lck.lock();
        while (a_thread->ready_queue.empty()) {
            // Return if stop=true and no tasks are left
            if (a_thread->team->m_stop.load() && a_thread->team->ntasks.load() <= 0) {
                lck.unlock();
                profiler.stop(pe);
                // Wake up all threads and exits
                a_thread->team->wake_all();
                return;
            }
            // Wait if queue is empty
            profiler.stop(pe);
            pe = profiler.start("wait");
            
            a_thread->cv.wait(lck);
            
//            lck.unlock();
//            std::this_thread::yield();
//            lck.lock();
            
            profiler.stop(pe);
            pe = profiler.start("overhead");
        }
    }
}

void Profiler::map_team_threads(Thread_team & team)
{
    for (auto& th_prio : team.v_thread) {
        thread_id_map[ id_to_string(th_prio.th.get_id()) ] = th_prio.m_id;
    }
    thread_id_map[id_to_string(std::this_thread::get_id()) ] = -1;
}

struct Matrix3_task : vector<Task> {
    int n0, n1, n2;
    Matrix3_task() {};
    Matrix3_task(int a_n0, int a_n1, int a_n2) : vector<Task>(a_n0*a_n1*a_n2),
        n0(a_n0), n1(a_n1), n2(a_n2) {};

    Task& operator()(int i, int j, int k)
    {
        assert(i>=0 && i<n0);
        assert(j>=0 && j<n1);
        assert(k>=0 && k<n2);
        return operator[](i + n0*(j + n1*k));
    }
};

typedef array<int,3> int3;

struct Task_flow {

    typedef std::function<int(int3&,Task*)> Init_task;
    typedef std::function<int(int3&)> Map_task;

    Thread_team * team = nullptr;
    Matrix3_task task_grid;

    struct Init {
        // How to initialize a task
        Init_task init;
        // Mapping from task index to thread id
        Map_task map;

        Init& task_init(Init_task a_init)
        {
            init = a_init;
            return *this;
        }

        Init& compute_on(Map_task a_map)
        {
            map = a_map;
            return *this;
        }
    } m_task;

    Task_flow& operator=(Init & a_setup)
    {
        m_task.init = a_setup.init;
        m_task.map = a_setup.map;
        return *this;
    }

    Task_flow(Thread_team * a_team, int n0, int n1, int n2) : team(a_team), task_grid(n0,n1,n2) {}

    virtual ~Task_flow() {}

    // Initialize the task
    void init_task(int3&);

    // spawn a task
    void async(int3);
    void async(int3, Task*);

    // Decrement the dependency counter and spawn task if ready
    void decrement_wait_count(int3);
};

Task_flow::Init task_flow_init()
{
    return Task_flow::Init();
}

void Task_flow::init_task(int3& idx)
{
    Task& tsk = task_grid(idx[0],idx[1],idx[2]);
    int wait_count = m_task.init(idx,&tsk);
    std::atomic_fetch_add(&(tsk.m_wait_count), wait_count+1);
}

// Spawn task
void Task_flow::async(int3 idx, Task* a_tsk)
{
    team->spawn(/*task map*/ m_task.map(idx), a_tsk);
}

// Initialize task and spawn it
void Task_flow::async(int3 idx)
{
    init_task(idx);
    async(idx, &task_grid(idx[0],idx[1],idx[2]));
}

void Task_flow::decrement_wait_count(int3 idx)
{
    assert(0 <= idx[0] && idx[0] < task_grid.n0);
    assert(0 <= idx[1] && idx[1] < task_grid.n1);
    assert(0 <= idx[2] && idx[2] < task_grid.n2);

    Task& tsk = task_grid(idx[0],idx[1],idx[2]);
    // Decrement counter
    int wait_count = std::atomic_fetch_sub(&(tsk.m_wait_count),1);

    if (wait_count == -1) { // Uninitialized task
        init_task(idx);
        wait_count = std::atomic_fetch_sub(&(tsk.m_wait_count),1);
    }

    if (wait_count == 0) { // task is ready to run
        async(idx, &tsk);
    }
}

struct Block_matrix : vector<MatrixXd*> {
    int row, col;
    Block_matrix() {};
    Block_matrix(int a_row, int a_col) : vector<MatrixXd*>(a_row*a_col),
        row(a_row), col(a_col) {};

    void resize(int a_row, int a_col)
    {
        vector<MatrixXd*>::resize(a_row*a_col);
        row = a_row;
        col = a_col;
    }

    MatrixXd* & operator()(int i, int j)
    {
        assert(i>=0 && i<row);
        assert(j>=0 && j<col);
        return operator[](i + j*row);
    }
};

void f_1()
{
}

void f_x(int * x)
{
    --(*x);
};

struct Fun_mv {
    MatrixXd *A;
    VectorXd *x, *y;
    void operator()()
    {
        (*y) = (*A) * (*x);
    }
};

TEST(TaskFlow, BasicTask)
{
    Task t1;
    t1.set_function( f_1 );
    ASSERT_EQ(t1.m_priority, 0.0);
    ASSERT_EQ(t1.m_delete, false);
    ASSERT_EQ(t1.m_wait_count.load(), -1);
    t1();

    Task t2;
    t2.set_function( f_1, 1.0 );
    ASSERT_EQ(t2.m_priority, 1.0);
    ASSERT_EQ(t2.m_delete, false);
    ASSERT_EQ(t2.m_wait_count.load(), -1);
    t2();

    Task t3;
    t3.set_function( f_1, 1.0, true );
    ASSERT_EQ(t3.m_priority, 1.0);
    ASSERT_EQ(t3.m_delete, true);
    ASSERT_EQ(t3.m_wait_count.load(), -1);
    t3();

    // Task using function with bind()
    int x = 2;
    Task t4;
    t4.set_function(bind(f_x, &x));
    t4();
    ASSERT_EQ(x, 1);

    MatrixXd A(2,2);
    A << 1,1,2,-2;
    ASSERT_EQ(A(0,0), 1);
    ASSERT_EQ(A(0,1), 1);
    ASSERT_EQ(A(1,0), 2);
    ASSERT_EQ(A(1,1), -2);

    VectorXd vec_x(2);
    vec_x << 3,2;
    ASSERT_EQ(vec_x(0), 3);
    ASSERT_EQ(vec_x(1), 2);

    VectorXd vec_y(2);

    // Demonstrating a function with arguments without bind()
    Fun_mv f_mv;
    f_mv.A = &A;
    f_mv.x = &vec_x;
    f_mv.y = &vec_y;

    Task t5;
    t5.set_function(f_mv);
    t5();

    ASSERT_EQ(vec_y(0), 5);
    ASSERT_EQ(vec_y(1), 2);
}

void f_count(std::atomic_int* c)
{
    ++(*c);
    std::this_thread::sleep_for(microseconds(20));
};

TEST(TaskFlow, Team)
{
    const int n_thread = 1024;
    const int max_count = 10;

    // Create thread team
    Thread_team team(n_thread);

    ASSERT_EQ(team.v_thread.size(), n_thread);
    ASSERT_EQ(team.ntasks.load(), 0);
    ASSERT_EQ(team.m_stop.load(), false);

    vector<std::atomic_int> counter(n_thread);

    for(auto & c : counter) {
        c.store(0);
    }

    vector<Task> tsk(n_thread);
    for(int nt=0; nt<n_thread; ++nt) {
        tsk[nt].set_function(bind(f_count, &counter[nt]));
    }

    {
        team.start();
        auto start = high_resolution_clock::now();
        for(int it=0; it < max_count; ++it) {
            for(int nt=0; nt<n_thread; ++nt) {
                team.spawn(0, &tsk[nt]);
                // spawn @ 0
            }
        }
        team.join();
        auto end = high_resolution_clock::now();
        auto duration_ = duration_cast<milliseconds>(end - start);
        ASSERT_GE(0, team.ntasks.load());

        std::cout << "Elapsed time for spawn @0: " << duration_.count() << "\n";
    }

    for(int nt=0; nt<n_thread; ++nt) {
        ASSERT_EQ(counter[nt].load(), max_count);
    }

    for(auto & c : counter) {
        c.store(0);
    }

    {
        team.start();
        auto start = high_resolution_clock::now();
        for(int it=0; it < max_count; ++it) {
            for(int nt=0; nt<n_thread; ++nt) {
                team.spawn(nt, &tsk[nt]);
                // spawn @ nt
            }
        }
        team.join();
        auto end = high_resolution_clock::now();
        auto duration_ = duration_cast<milliseconds>(end - start);
        std::cout << "Elapsed time for spawn @nt: " << duration_.count() << "\n";
    }

    for(int nt=0; nt<n_thread; ++nt) {
        ASSERT_EQ(counter[nt].load(), max_count);
    }
}

void pass_the_bucket(vector<Task>* tsk, vector<std::atomic_int>* counter,
                     Thread_team* team,
                     const int size, const int n_thread, int th)
{
    auto pe = profiler.start("task");
    ++(*counter)[th%size];
    std::this_thread::sleep_for(microseconds(10));
    profiler.stop(pe);
    pe = profiler.start("spawn");
    if (th%size!= size-1) {
        team->spawn((th+1)%n_thread, &(*tsk)[th+1]);
    }
    profiler.stop(pe);
}

TEST(TaskFlow, Ring)
{
    const int n_thread = 2;
    const int ring_size = 32;
    const int max_count = 16;

    // Create thread team
    Thread_team team(n_thread);

    ASSERT_EQ(team.v_thread.size(), n_thread);
    ASSERT_EQ(team.ntasks.load(), 0);
    ASSERT_EQ(team.m_stop.load(), false);

    vector<std::atomic_int> counter(ring_size);

    for(auto & c : counter) {
        c.store(0);
    }

    vector<Task> tsk(ring_size*max_count);
    int nt = 0;
    for(int ic=0; ic<max_count; ++ic) {
        for(int i=0; i<ring_size; ++i, ++nt) {
            tsk[nt].set_function(bind(pass_the_bucket,
                                      &tsk,&counter,&team,ring_size,n_thread,nt));
        }
    }

    team.start();
    for(int ic=0; ic<ring_size*max_count; ++ic) {
        team.spawn(ic%n_thread, &tsk[ic]);
    }
    team.join();

    ASSERT_GE(0, team.ntasks.load());

    for(int i=0; i<ring_size; ++i) {
        ASSERT_EQ(counter[i].load(), max_count*(i+1));
    }

    profiler.dump();
}

void test_ring_flow(const int n_thread,const int ring_size, const int max_count, const int stride)
{
    const int n_task = ring_size * max_count;

    vector<std::atomic_int> counter(ring_size);

    for(auto & c : counter) {
        c.store(0);
    }

    // Create thread team
    Thread_team team(n_thread);

    // Ring task flow
    Task_flow ring_flow(&team,n_task,1,1);

    ASSERT_EQ(ring_flow.team, &team);
    ASSERT_EQ(ring_flow.task_grid.n0, n_task);
    ASSERT_EQ(ring_flow.task_grid.n1, 1);
    ASSERT_EQ(ring_flow.task_grid.n2, 1);

    // Init task flow
    ring_flow = task_flow_init()
    .task_init([=,&ring_flow,&counter] (int3& idx, Task* a_tsk) -> int {
        int i = idx[0];
        a_tsk->set_function([=,&ring_flow,&counter] ()
        {
            ++counter[i%ring_size];
            if (i < n_task-1 && i%stride != stride-1) {
                ring_flow.decrement_wait_count({i+1,0,0});
            }
        });
        return (i%stride == 0 ? 0 : 1); // wait_count
    })
    .compute_on([=] (int3& idx) {
        return idx[0] % n_thread;
    });

    // Start team of threads
    team.start();

    // Create seed tasks and start
    for (int i=0; i<n_task; i += stride) {
        ring_flow.async({i,0,0});
    }

    // Wait for end of task queue execution
    team.join();

    ASSERT_GE(0, team.ntasks.load());

    for(int i=0; i<ring_size; ++i) {
        ASSERT_EQ(counter[i].load(), max_count);
    }
}

TEST(TaskFlow, RingFlow1)
{
    const int n_thread = 1024;
    const int ring_size = 1<<8;
    const int max_count = 1<<4;
    const int stride    = 1<<12;
    test_ring_flow(n_thread,ring_size,max_count,stride);
}

TEST(TaskFlow, RingFlow2)
{
    const int n_thread = 1024;
    const int ring_size = 1<<8;
    const int max_count = 1<<4;
    const int stride    = 1<<6;
    test_ring_flow(n_thread,ring_size,max_count,stride);
}

TEST(TaskFlow, RingFlow3)
{
    const int n_thread = 1024;
    const int ring_size = 1<<8;
    const int max_count = 1<<4;
    const int stride    = 1<<1;
    test_ring_flow(n_thread,ring_size,max_count,stride);
}

TEST(TaskFlow, RingFlow4)
{
    const int n_thread = 1024;
    const int ring_size = 1<<8;
    const int max_count = 1<<4;
    const int stride    = 1<<0;
    test_ring_flow(n_thread,ring_size,max_count,stride);
}

TEST(TaskFlow, Eigen)
{
    const int nb = 8; // number of blocks
    const int b = 8;   // size of blocks

    const int n = b*nb; // matrix size

    MatrixXd A(n,n), B(n,n);

    // Initialize GEMM matrices
    std::mt19937 mers_rand;
    // Seed the random engine
    mers_rand.seed(2018);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            A(i,j) = mers_rand()%3;
            B(i,j) = mers_rand()%3;
        }
    }

    MatrixXd C = A*B; // Reference result for testing

    // Doing calculation using matrix of blocks
    {
        Block_matrix Ab(nb,nb), Bb(nb,nb), Cb(nb,nb);

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                Ab(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
                Bb(i,j) = new MatrixXd(B.block(i*b,j*b,b,b));
                Cb(i,j) = new MatrixXd(MatrixXd::Zero(b,b));
            }
        }

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                ASSERT_EQ(*(Ab(i,j)), A.block(i*b,j*b,b,b));
                ASSERT_EQ(*(Bb(i,j)), B.block(i*b,j*b,b,b));
                ASSERT_EQ(*(Cb(i,j)), MatrixXd::Zero(b,b));
            }
        }

        // Calculate matrix product using blocks
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                for (int k=0; k <nb; ++k) {
                    *(Cb(i,j)) += *(Ab(i,k)) * *(Bb(k,j));
                }
            }
        }

        // First test
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                ASSERT_EQ(*(Cb(i,j)), C.block(i*b,j*b,b,b));
            }
        }

        // Copy back for testing purposes
        MatrixXd C0(n,n);
        for (int i=0; i<nb; ++i) {
            for (int j=0; j<nb; ++j) {
                MatrixXd & M = *(Cb(i,j));
                for (int i0=0; i0<b; ++i0) {
                    for (int j0=0; j0<b; ++j0) {
                        C0(i0+b*i,j0+b*j) = M(i0,j0);
                    }
                }
            }
        }

        // Second test
        ASSERT_EQ(C, C0);

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                delete Ab(i,j);
                delete Bb(i,j);
                delete Cb(i,j);
            }
        }
    }
}

TEST(TaskFlow, GEMM)
{
    const int nb = 4; // number of blocks
    const int b = 32;  // size of blocks

    const int n_thread = 2; // number of threads to use

    const int n = b*nb; // matrix size

    MatrixXd A(n,n), B(n,n);

    // Initialize GEMM matrices
    std::mt19937 mers_rand;
    // Seed the engine
    mers_rand.seed(2018);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            A(i,j) = mers_rand()%3;
            B(i,j) = mers_rand()%3;
        }
    }

    MatrixXd C = A*B;

    // Calculation using task flow
    Block_matrix Ab(nb,nb), Bb(nb,nb), Cb(nb,nb);

    for (int i=0; i<nb; ++i) {
        for (int j=0; j<nb; ++j) {
            Ab(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
            Bb(i,j) = new MatrixXd(B.block(i*b,j*b,b,b));
            Cb(i,j) = new MatrixXd(MatrixXd::Zero(b,b));
        }
    }

    // Create thread team
    Thread_team team(n_thread);

    // GEMM task flow
    Task_flow gemm_g(&team,nb,nb,nb);

    auto compute_on_ij = [=] (int3& idx) {
        return ( ( idx[0] + nb * idx[1] ) % n_thread );
    };

    // GEMM task flow
    gemm_g = task_flow_init()
    .task_init([=,&gemm_g,&Ab,&Bb,&Cb] (int3& idx, Task* a_tsk) -> int {
        int i = idx[0];
        int j = idx[1];
        int k = idx[2];

        a_tsk->set_function([=,&gemm_g,&Ab,&Bb,&Cb] ()
        {
            auto pe = profiler.start("gemm");
            *(Cb(i,j)) += *(Ab(i,k)) * *(Bb(k,j)); // GEMM
            profiler.stop(pe);
            pe = profiler.start("dependency");
            if (k < nb - 1) {
                gemm_g.decrement_wait_count({i,j,k+1});
            }
            profiler.stop(pe);
        });
        return (/*k*/ idx[2] == 0 ? 0 : 1); // wait_count
    })
    .compute_on(compute_on_ij);
    
    profiler.open("gemm_v2.out");

    // Start team of threads
    team.start();

    profiler.map_team_threads(team);

    auto start = high_resolution_clock::now();

    // Create seed tasks and start
    for (int i=0; i<nb; ++i) {
        for (int j=0; j<nb; ++j) {
            gemm_g.async({i,j,0});
        }
    }

    // Wait for end of task queue execution
    team.join();

    auto end = high_resolution_clock::now();
    auto duration_ = duration_cast<milliseconds>(end - start);
    std::cout << "Elapsed time Task Flow: " << duration_.count() << "ms\n";

    // Test output
    for (int i=0; i<nb; ++i) {
        for (int j=0; j <nb; ++j) {
            ASSERT_EQ(*(Cb(i,j)), C.block(i*b,j*b,b,b));
        }
    }

    for (int i=0; i<nb; ++i) {
        for (int j=0; j <nb; ++j) {
            delete Ab(i,j);
            delete Bb(i,j);
            delete Cb(i,j);
        }
    }

    profiler.dump();
}

TEST(TaskFlow, GEMMInit)
{
    const int nb = 4; // number of blocks
    const int b = 32;  // size of blocks

    const int n_thread = 8; // number of threads to use

    const int n = b*nb; // matrix size

    MatrixXd A(n,n), B(n,n);

    // Initialize GEMM matrices
    std::mt19937 mers_rand;
    // Seed the engine
    mers_rand.seed(2018);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            A(i,j) = mers_rand()%3;
            B(i,j) = mers_rand()%3;
        }
    }

    auto start = high_resolution_clock::now();
    MatrixXd C = A*B;
    auto end = high_resolution_clock::now();
    auto duration_ = duration_cast<milliseconds>(end - start);
    std::cout << "Elapsed time Eigen: " << duration_.count() << "ms\n";

    // Calculation using task flow
    Block_matrix Ab(nb,nb), Bb(nb,nb), Cb(nb,nb);

    // Create thread team
    Thread_team team(n_thread);

    // Task flow context
    struct Context {
        Task_flow init_mat;
        Task_flow gemm_g;
        Context(Thread_team* a_tt, int a_nb) :
            init_mat(a_tt,a_nb,a_nb,1),
            gemm_g(a_tt,a_nb,a_nb,a_nb) {}
    } ctx(&team, nb);

    ASSERT_EQ(ctx.init_mat.team, &team);
    ASSERT_EQ(ctx.init_mat.task_grid.n0, nb);
    ASSERT_EQ(ctx.init_mat.task_grid.n1, nb);
    ASSERT_EQ(ctx.init_mat.task_grid.n2, 1);
    ASSERT_EQ(ctx.gemm_g.task_grid.n0, nb);
    ASSERT_EQ(ctx.gemm_g.task_grid.n1, nb);
    ASSERT_EQ(ctx.gemm_g.task_grid.n2, nb);

    auto compute_on_ij = [=] (int3& idx) {
        return ( ( idx[0] + nb * idx[1] ) % n_thread );
    };

    // Init task flow
    ctx.init_mat = task_flow_init()
    .task_init([=,&ctx,&Ab,&Bb,&Cb] (int3& idx, Task* a_tsk) -> int {
        int i = idx[0];
        int j = idx[1];
        a_tsk->set_function([=,&ctx,&Ab,&Bb,&Cb] ()
        {
            auto pe = profiler.start("init mat");
            Ab(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
            Bb(i,j) = new MatrixXd(B.block(i*b,j*b,b,b));
            Cb(i,j) = new MatrixXd(MatrixXd::Zero(b,b));
            for (int k=0; k<nb; ++k) {
                ctx.gemm_g.decrement_wait_count({i,k,0});
                ctx.gemm_g.decrement_wait_count({k,j,0});
            }
            profiler.stop(pe);
        });
        return 0; // wait_count
    })
    .compute_on(compute_on_ij);

    // GEMM task flow
    ctx.gemm_g = task_flow_init()
    .task_init([=,&ctx,&Ab,&Bb,&Cb] (int3& idx, Task* a_tsk) -> int {
        int i = idx[0];
        int j = idx[1];
        int k = idx[2];

        a_tsk->set_function([=,&ctx,&Ab,&Bb,&Cb] ()
        {
            auto pe = profiler.start("gemm");
            *(Cb(i,j)) += *(Ab(i,k)) * *(Bb(k,j)); // GEMM
            if (k < nb - 1) {
                ctx.gemm_g.decrement_wait_count({i,j,k+1});
            }
            profiler.stop(pe);
        });
        return (/*k*/ idx[2] == 0 ? 2*nb : 1); // wait_count
    })
    .compute_on(compute_on_ij);

    // Start team of threads
    team.start();

    start = high_resolution_clock::now();

    // Create seed tasks and start
    for (int i=0; i<nb; ++i) {
        for (int j=0; j<nb; ++j) {
            ctx.init_mat.async({i,j,0});
        }
    }

    // Wait for end of task queue execution
    team.join();

    end = high_resolution_clock::now();
    duration_ = duration_cast<milliseconds>(end - start);
    std::cout << "Elapsed time Task Flow: " << duration_.count() << "ms\n";

    // Test output
    for (int i=0; i<nb; ++i) {
        for (int j=0; j <nb; ++j) {
            if ( *(Cb(i,j)) != C.block(i*b,j*b,b,b) ) {
                LOG(i << " " << j << " = " << *(Cb(i,j)) << " " << C.block(i*b,j*b,b,b));
            }
        }
        for (int j=0; j <nb; ++j) {
            ASSERT_EQ(*(Cb(i,j)), C.block(i*b,j*b,b,b));
        }
    }

    for (int i=0; i<nb; ++i) {
        for (int j=0; j <nb; ++j) {
            delete Ab(i,j);
            delete Bb(i,j);
            delete Cb(i,j);
        }
    }

    profiler.dump();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
