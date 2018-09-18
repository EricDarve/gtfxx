#include "gtest/gtest.h"

// My header files
#include "gtfxx.h"

using namespace std;
using namespace upcxx;
using namespace gtfxx;

int64_t ans = -1;

TEST(UPCXX, Basic) {
    // -----------------
    // Simple upc++ test

    const upcxx::intrank_t n_rank = upcxx::rank_n();
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
            .then_on_receiving_rank([](
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
    // Using a Active_message thread and upc++ to send a scalar

    const upcxx::intrank_t n_rank = upcxx::rank_n();
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
    // Using a Active_message thread and upc++ to send a vector

    const upcxx::intrank_t n_rank = upcxx::rank_n();
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
    // Using a Active_message thread and gtf++ to send a vector

    const upcxx::intrank_t n_rank = upcxx::rank_n();
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

    int64_t local_ans = -1;

    auto task_thread_comm = [=, &local_ans]() {
        // Acquiring master persona
        upcxx::persona_scope scope(upcxx::master_persona());

        gtfxx::send(gtfxx::memblock_view<int64_t>(msg.size(), &msg[0]), my_rank_send)
                .to_rank(dest)
                .then_on_receiving_rank([=, &local_ans](gtfxx::memblock<int64_t>::iterator msg_, int64_t my_rank_send) {
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

// -----------------
// Task flow context

struct Context {
    std::map<std::string, gtfxx::Task_flow> m_map_task;
    std::map<std::string, gtfxx::Channel> m_map_comm;

    void task_emplace(const std::string s, int n0, int n1, int n2, gtfxx::Thread_pool *th_pool) {
        m_map_task.emplace(std::piecewise_construct,
                           std::forward_as_tuple(s),
                           std::forward_as_tuple(th_pool, n0, n1, n2)
        );
    }

    void comm_emplace(const std::string s, gtfxx::Thread_pool *th_pool) {
        m_map_comm.emplace(std::piecewise_construct,
                           std::forward_as_tuple(s),
                           std::forward_as_tuple(th_pool)
        );
    }

    gtfxx::Task_flow &map_task(const std::string s) {
        auto search = m_map_task.find(s);
        if (search != m_map_task.end()) {
            return search->second;
        } else {
            throw_assert(false, "invalid argument for map_task " << s);
        }
    };

    gtfxx::Channel &map_comm(const std::string s) {
        auto search = m_map_comm.find(s);
        if (search != m_map_comm.end()) {
            return search->second;
        } else {
            throw_assert(false, "invalid argument for map_comm " << s);
        }
    };
};

TEST(gtfxx, UPCXX) {

    // TODO: replace by gtfxx implementations
    const upcxx::intrank_t n_rank = upcxx::rank_n();
    const upcxx::intrank_t my_rank = upcxx::rank_me();
    const int n_thread = 4; // number of threads to use

    LOG("[main] n_rank " << n_rank << "; my_rank " << my_rank <<
                         "; n_thread " << n_thread);

    profiler.open("prof.out");

    // GTF++ context; thread pool
    Context ctx;

    Thread_pool th_pool(n_thread);

    // Useful shortcut lambda functions
    auto ctx_task = [&ctx](const string s) -> Task_flow & {
        return ctx.map_task(s);
    };

    auto ctx_comm = [&ctx](const string s) -> Channel & {
        return ctx.map_comm(s);
    };

    // Setting up task flow grids that are needed later
    ctx.task_emplace("map", n_thread, 1, 1, &th_pool);
    ctx.comm_emplace("send", &th_pool);
    ctx.task_emplace("reduce", 1, 1, 1, &th_pool);

    vector<int> data(n_thread * n_rank, -1);

    auto compute_on_i = [=](int3 &idx) {
        return idx[0] % n_thread;
    };

    // Define TF "map"
    ctx_task("map")
            .wait_on_promises([](const int3 idx) { return 0; })
            .then_run(
                    [=, &data](const int3 idx) {
                        assert(idx[0] >= 0 && idx[0] < n_thread);
                        assert(my_rank >= 0 && my_rank < n_rank);

                        const int global_comm_idx = idx[0] + my_rank * n_thread;
                        data[global_comm_idx] = 1;

                        ctx_comm("send").fulfill_promise({global_comm_idx, 0, 0});
                    })
            .on_thread(compute_on_i);

    // Define comm TF "send"
    // Needs to be defined on the sending rank only
    ctx_comm("send")
            .wait_on_promises([](const int3 idx) { return 1; })
            .then_send([=, &data](const int3 global_comm_idx) {
                assert(global_comm_idx[0] >= 0 && global_comm_idx[0] < n_rank * n_thread);
                assert(data[global_comm_idx[0]] == 1);
                assert(my_rank >= 0 && my_rank < n_rank);

                // Executed by the active message thread on the sending rank
                for (int i = 0; i < n_rank; ++i) {
                    if (i != my_rank) {
                        gtfxx::send(global_comm_idx, data[global_comm_idx[0]])
                                .to_rank(i)
                                .then_on_receiving_rank([=, &data](int3 global_comm_idx, int d) {
                                    /* Executed by the active message thread on the receiving rank
                                     * The arguments must match the arguments in send(...) above. */
                                    assert(global_comm_idx[0] >= 0 && global_comm_idx[0] < n_thread * n_rank);
                                    assert(data[global_comm_idx[0]] == -1);
                                    assert(d == 1);

                                    data[global_comm_idx[0]] = d;
                                    ctx_comm("send").finalize(global_comm_idx);
                                    //ctx_task("reduce").fulfill_promise({0, 0, 0});
                                });
                    } else {
                        ctx_task("reduce").fulfill_promise({0, 0, 0}); // Required in case n_rank==1
                    }
                }
            });

    /* Will run on the receiving rank, and needs to be defined on the receiving rank only.
     * Captured variables correspond to memory locations on the receiving rank. */
    ctx_comm("send").set_finalize([=](int3 global_comm_idx) {
        ctx_task("reduce").fulfill_promise({0, 0, 0});
    });

    int local_sum;
    atomic_bool done(false);

    // Define TF "reduce"
    ctx_task("reduce")
            .wait_on_promises([=](const int3 idx) { return n_thread * n_rank; })
            .then_run([&data, &local_sum, &done](const int3 idx) {
                assert(idx[0] == 0);
                assert(!done.load());
                for (auto d : data) {
                    assert(d == 1);
                }

                done.store(true); // This is the last task that we need to run
                local_sum = 0;
                for (auto d : data) {
                    local_sum += d;
                }
            })
            .on_thread(compute_on_i);

    // Start pool of threads
    th_pool.start();

    profiler.record_thread_ids(th_pool);

    // Create seed tasks and start
    for (int i = 0; i < n_thread; ++i) {
        ctx_task("map").seed_task({i, 0, 0});
    }

    // Because of the communications, detecting quiescence
    // requires monitoring the atomic boolean "done".
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Wait for end of execution remaining tasks
    th_pool.join();

    upcxx::barrier();

    ASSERT_EQ(local_sum, n_thread*n_rank);

    profiler.dump();
}

int main(int argc, char **argv) {

    upcxx::init();

    ::testing::InitGoogleTest(&argc, argv);

    const int return_flag = RUN_ALL_TESTS();

    upcxx::finalize();

    return return_flag;
}
