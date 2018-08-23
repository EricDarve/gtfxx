#include <cstdio>
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

#include "deque.hpp"

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

void f_1()
{
}

void f_x(int * x)
{
    --(*x);
};

void f_count(std::atomic_int * c)
{
    ++(*c);
    std::this_thread::sleep_for(microseconds(20));
};

struct Fun_mv {
    MatrixXd *A;
    VectorXd *x, *y;
    void operator()()
    {
        (*y) = (*A) * (*x);
    }
};

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
struct Thread_team;

#define BUFFER_SIZE 1<<8

struct thread_task_pair {
    int th_id; // thread id
    Task* tsk; // Task pointer
    thread_task_pair(int a_th_id, Task* a_tsk) : th_id(a_th_id), tsk(a_tsk) {}
};

// Thread with priority queue management
struct Thread_prio {
    Thread_team * team;
    unsigned short m_id;

    // lockless and waitfree queue
    Deque<Task,BUFFER_SIZE> ready_queue;
    // private queue
    std::priority_queue<Task*, vector<Task*>, Task_comparison> private_queue;
    // tasks assigned to other threads
    std::queue<thread_task_pair> alien_queue;

    thread th;
    mutex mtx;
    std::condition_variable cv;

    // Thread starts executing the function spin()
    void start()
    {
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    bool spawn(Task * a_t)
    {
        std::lock_guard<std::mutex> lck(mtx);
        bool pushed = ready_queue.push(a_t);
        if (pushed) {
            cv.notify_one(); // Wake up thread
        }
        return pushed;
    };

    bool pop(Task** tsk)
    {
        return ready_queue.pop(tsk);
    }

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
    std::map<std::thread::id,int> thread_map; // Associate a thread ID to an integer
    std::atomic_int ntasks; // number of ready tasks in any thread queue
    std::atomic_bool m_stop;

    Deque<unsigned short,BUFFER_SIZE> idle_queue;

    Thread_team(const int n_thread) : v_thread(n_thread)
    {
        for (unsigned short i=0; i < static_cast<unsigned short>(n_thread); ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = i;
            thread_map[v_thread[i].th.get_id()] = i;
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
        ++ntasks;
        if (!v_thread[a_id].spawn(a_task)) {
            // Failed to push into queue
            //LOG("Failed spawn");
            std::thread::id this_id = std::this_thread::get_id();
            auto search = thread_map.find(this_id);
            if (search != thread_map.end()) {
                // A thread in the team
                LOG("Pushing into alien queue " << this_id);
                v_thread[ thread_map[this_id] ].alien_queue.push(thread_task_pair(a_id,a_task));
            }
            else {
                // This thread is not part of the team
                //LOG("Failed spawn from alien thread");
                while (!v_thread[a_id].spawn(a_task));
            }
        }
    }
};

// Keep executing tasks until m_stop = true && there are no tasks left anywhere
void spin(Thread_prio * a_thread)
{
    while (true) {
        // First empty the ready queue
        while (!a_thread->ready_queue.empty()) {
            Task * tsk;
            while (!a_thread->ready_queue.pop(&tsk));
            a_thread->private_queue.push(tsk);
        }
        // Then go through all our private tasks
        while (!a_thread->private_queue.empty()) {
            Task * tsk = a_thread->private_queue.top();
            a_thread->private_queue.pop();
            (*tsk)();
            --(a_thread->team->ntasks);
            if (tsk->m_delete) delete tsk;
        }
        // Go through tasks assigned to other threads
        while (!a_thread->alien_queue.empty()) {
            thread_task_pair th_task = a_thread->alien_queue.front();
            // Try to insert into rightful queue
            auto & th_prio = a_thread->team->v_thread[th_task.th_id];
            if (!th_prio.spawn(th_task.tsk)) {
                // Failed; then execute
                Task * tsk = th_task.tsk;
                a_thread->alien_queue.pop();
                (*tsk)();
                --(a_thread->team->ntasks);
                if (tsk->m_delete) delete tsk;
            }
        }
        {
            std::unique_lock<std::mutex> lck(a_thread->mtx);
            while (a_thread->ready_queue.empty()) {
                // Return if stop=true and no tasks are left
                if (a_thread->team->m_stop.load() && a_thread->team->ntasks.load() <= 0) {
                    lck.unlock();
                    // Wake up all threads and exits
                    a_thread->team->wake_all();
                    return;
                }
                // while (!a_thread->team->idle_queue.push(&a_thread->m_id));
                // Now wait
                a_thread->cv.wait(lck);
            }
        }
    }
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
    const int nb = 32; // number of blocks
    const int b = 1;  // size of blocks

    const int n_thread = 1024; // number of threads to use

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
    std::cout << "Elapsed time Eigen: " << duration_.count() << "\n";

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
            Ab(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
            Bb(i,j) = new MatrixXd(B.block(i*b,j*b,b,b));
            Cb(i,j) = new MatrixXd(MatrixXd::Zero(b,b));
            for (int k=0; k<nb; ++k) {
                ctx.gemm_g.decrement_wait_count({i,k,0});
                ctx.gemm_g.decrement_wait_count({k,j,0});
            }
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
            *(Cb(i,j)) += *(Ab(i,k)) * *(Bb(k,j)); // GEMM
            if (k < nb - 1) {
                ctx.gemm_g.decrement_wait_count({i,j,k+1});
            }
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
    std::cout << "Elapsed time Task Flow: " << duration_.count() << "\n";

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
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
