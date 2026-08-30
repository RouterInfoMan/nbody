#pragma once
#include <vector>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0)
        : stop(false) {
        if (num_threads <= 0)
            num_threads = std::max(1, (int)std::thread::hardware_concurrency());
        for (int i = 0; i < num_threads; i++) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto& w : workers) w.join();
    }

    int size() const { return static_cast<int>(workers.size()); }

    // `grain` is the minimum number of items worth handing to a worker. Loops
    // that cannot fill that many per thread run inline instead, which matters
    // for FMM: it issues dozens of parallel_for calls per step and the coarse
    // levels have only a handful of cells each, where dispatch costs more than
    // the work.
    void parallel_for(int begin, int end, const std::function<void(int, int)>& fn,
                      int grain = 1) {
        int n = end - begin;
        int num = size();
        if (n <= 0) return;
        if (grain > 1) num = std::min(num, std::max(1, n / grain));
        if (num <= 1 || n < num) {
            fn(begin, end);
            return;
        }

        int remaining = num;
        std::mutex done_mtx;
        std::condition_variable done_cv;

        int chunk = n / num;
        int extra = n % num;
        int offset = begin;

        for (int i = 0; i < num; i++) {
            int start = offset;
            int count = chunk + (i < extra ? 1 : 0);
            offset += count;

            {
                std::unique_lock<std::mutex> lock(mtx);
                tasks.push([&fn, &remaining, &done_cv, &done_mtx, start, count] {
                    fn(start, start + count);
                    std::lock_guard<std::mutex> lk(done_mtx);
                    if (--remaining == 0)
                        done_cv.notify_one();
                });
            }
            cv.notify_one();
        }

        std::unique_lock<std::mutex> lock(done_mtx);
        done_cv.wait(lock, [&remaining] { return remaining == 0; });
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop;
};
