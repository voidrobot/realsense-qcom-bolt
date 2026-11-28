// Copyright 2026 voidrobot
// SPDX-License-Identifier: Apache-2.0

#include "realsense_qcom_bolt/dmabuf_pool.hpp"

#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

struct MockDmaBuffer
{
    static inline std::atomic<int> unmap_count{0};
    static inline std::atomic<int> live_count{0};

    std::vector<uint8_t> storage;
    void* mapped{nullptr};

    explicit MockDmaBuffer(std::size_t bytes) : storage(bytes, 0xAA) { ++live_count; }
    ~MockDmaBuffer() { --live_count; }

    static std::shared_ptr<MockDmaBuffer> alloc(std::size_t bytes, const std::string&)
    {
        return std::make_shared<MockDmaBuffer>(bytes);
    }
    bool map() { mapped = storage.data(); return true; }
    bool unmap()
    {
        if (mapped)
        {
            mapped = nullptr;
            ++unmap_count;
        }
        return true;
    }
    bool sync_start() { return true; }
    bool sync_end() { return true; }
    void* addr() const { return mapped; }
};

int main()
{
    using Pool = realsense_qcom_bolt::BasicDmaBufferPool<MockDmaBuffer>;

    // 1. Reuse & zero_init
    {
        Pool pool(2);
        auto b1 = pool.acquire(1024);
        assert(b1 && b1->addr() && static_cast<uint8_t*>(b1->addr())[0] == 0);
        void* p1 = b1->addr();
        b1.reset();
        assert(pool.free_count() == 1);
        auto b2 = pool.acquire(1024);
        assert(b2->addr() == p1);
    }
    assert(MockDmaBuffer::live_count.load() == 0);

    // 2. Saturation transient auto-unmap + discard hook
    {
        std::atomic<int> hook_calls{0};
        Pool pool(2, "/dev/dma_heap/system", [&](const MockDmaBuffer&) { ++hook_calls; });
        int unmap_before = MockDmaBuffer::unmap_count.load();
        auto s1 = pool.acquire(512);
        auto s2 = pool.acquire(512);
        auto t = pool.acquire(512);
        t.reset();
        assert(MockDmaBuffer::unmap_count.load() == unmap_before + 1);
        assert(hook_calls.load() == 1);
    }

    // 3. Generation shrink invalidation
    {
        Pool pool(2);
        auto big = pool.acquire(4096);
        auto g1 = pool.generation();
        auto small = pool.acquire(1024);
        assert(pool.generation() == g1 + 1);
        int unmap_before = MockDmaBuffer::unmap_count.load();
        big.reset();
        assert(MockDmaBuffer::unmap_count.load() == unmap_before + 1);
        assert(pool.free_count() == 0);
    }

    // 4. Multithreaded stress
    {
        Pool pool(4);
        std::vector<std::thread> threads;
        for (int th = 0; th < 4; ++th)
        {
            threads.emplace_back([&]() {
                for (int i = 0; i < 500; ++i)
                {
                    auto b = pool.acquire(1024);
                    assert(b && b->addr());
                }
            });
        }
        for (auto& th : threads) th.join();
        assert(pool.free_count() <= 4);
    }
    assert(MockDmaBuffer::live_count.load() == 0);
    std::cout << "ALL_BOLT_DMABUF_POOL_TESTS_PASSED" << std::endl;
    return 0;
}
