#include <cstdio>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::list;
using std::vector;
using std::thread;
using std::mutex;
using std::bind;
typedef std::function<void()> Base_task;

void f_1()
{
}

void f_2()
{
}

void f_x(int * x)
{
    --(*x);
};

void f_count(int * c)
{
    ++(*c);
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
    int indegree = 0; // Number of incoming edges
    std::queue<Task*> outgoing; // List of outgoing edges
    float priority = 0; // task priority

    void operator()()
    {
        Base_task::operator()(); // Execute task
        decrement_outgoing(); // Decrement counters for outgoing edges
    }

    Task() {}

    Task( std::function<void()> a_f, int a_indegree = 0, float a_priority = 0.) :
        Base_task(a_f), indegree(a_indegree), priority(a_priority) {}
    ~Task() {};

    void decrement_outgoing()
    {
        while (!outgoing.empty()) {
            --outgoing.front()->indegree; // TODO race condition!
            outgoing.pop();
        }
    }
};

// Task comparison based on their priorities
struct Task_comparison {
public:
    bool operator() (const Task* a_lhs, const Task* a_rhs) const
    {
        return (a_lhs->priority < a_rhs->priority);
    };
};

struct Thread_team;

struct Thread_prio;
void spin(Thread_prio * a_prio_t);

// Thread with priority queue management
struct Thread_prio {
    std::priority_queue<Task*, vector<Task*>, Task_comparison> ready_queue;
    thread th;
    mutex mtx;
    std::condition_variable cv;
    std::atomic_bool m_stop;
    Thread_team * team;
    Task tsk;

    static void final_task() {};

    // Thread starts executing the function spin()
    void start()
    {
        m_stop.store(false); // Used to return from spin()
        tsk = Task(final_task); // Task is inserted in queue at the end
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    void run(Task * a_t)
    {
        std::lock_guard<std::mutex> lck(mtx);
        ready_queue.push(a_t); // Add task to queue
        cv.notify_all(); // Wake up thread if needed
    };

    // Set stop boolean to true so spin() can return
    void stop()
    {
        m_stop.store(true); // Set boolean to true
        run(&tsk); // Run a final (empty) task to wake up the thread if needed
    };

    // join() the thread
    void join()
    {
        stop();
        if(th.joinable()) {
            th.join();
        }
    }

    ~Thread_prio()
    {
        join();
    }
};

// Keep executing tasks until m_continue = false
void spin(Thread_prio * a_prio_t)
{
    std::unique_lock<std::mutex> lck(a_prio_t->mtx);
    while (true) {
        while (!a_prio_t->ready_queue.empty()) {
            Task * t = a_prio_t->ready_queue.top();
            a_prio_t->ready_queue.pop();
            lck.unlock();
            (*t)();
            lck.lock();
        }
        // queue must be empty
        if (a_prio_t->m_stop.load()) { // if stop=true then return
            return;
        }
        // Wait if queue is empty
        while (a_prio_t->ready_queue.empty()) a_prio_t->cv.wait(lck);
    }
}

struct Thread_team : public vector<Thread_prio*> {
    vector<Thread_prio> v_thread;

    Thread_team(const int n_thread) : v_thread(n_thread)
    {
        for (auto& th : v_thread) th.team = this;
    }

    void start()
    {
        for (auto& th : v_thread) th.start();
    }
    void stop()
    {
        for (auto& th : v_thread) th.stop();
    }
    void join()
    {
        for (auto& th : v_thread) th.join();
    }
    void run(const int a_thread, Task * a_task)
    {
        v_thread[a_thread].run(a_task);
    }
};

void test();

int main(void)
{

    try {
        test();
    }
    catch (std::exception& a_e) {
        std::cout << a_e.what() << '\n';
    }

    return 0;
}

void test()
{

    {
        Task t1( f_1 );
        assert(t1.indegree == 0);

        Task t2( f_2, 4 );
        assert(t2.indegree == 4);

        t1.outgoing.push(&t2);

        t1();
    }

    {
        int x = 2;
        assert(x == 2);

        Task t(bind(f_x, &x));
        t();

        assert(x == 1);
    }

    {
        MatrixXd A(2,2);
        A << 1,1,2,-2;
        assert(A(0,0) == 1);
        assert(A(0,1) == 1);
        assert(A(1,0) == 2);
        assert(A(1,1) == -2);
        VectorXd x(2);
        x << 3,2;
        assert(x(0) == 3);
        assert(x(1) == 2);
        VectorXd y(2);

        // Demonstrating a function with arguments without bind()
        Fun_mv f_mv;
        f_mv.A = &A;
        f_mv.x = &x;
        f_mv.y = &y;

        Task t(f_mv);
        t();

        assert(y(0) == 5);
        assert(y(1) == 2);
    }

    {
        const int n_thread = 2;

        // Create thread team
        Thread_team team(n_thread);
        vector<int> counter(n_thread, 0);

        vector<Task> tsk(n_thread);
        for(int nt=0; nt<n_thread; ++nt) {
            tsk[nt] = static_cast<Task>(bind(f_count, &counter[nt]));
        }

        team.start();

        const int max_count = 100;
        for(int it=0; it < max_count; ++it) {
            for(int nt=0; nt<n_thread; ++nt) {
                team.run(nt, &tsk[nt]);
            }
        }

        team.join();

        for(int nt=0; nt<n_thread; ++nt) {
            assert(counter[nt] == max_count);
        }
    }
}
