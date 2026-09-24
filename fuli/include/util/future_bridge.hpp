#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <chrono>
#include <future>

namespace util {

// Bridges any std::future<T> — from std::async, from GpuFaissEngine's
// worker-thread queue, from anything — onto a boost::asio coroutine
// WITHOUT blocking the thread that's co_await-ing it.
//
// Why this needs to exist at all: std::future and boost::asio's
// coroutines (net::awaitable<T>) are two unrelated concurrency systems
// that don't know about each other. A std::future has no way to notify
// an io_context "hey, I'm ready now, wake up whoever's waiting on me" —
// the only thing you can do with a bare std::future is call .get() or
// .wait(), and BOTH of those block the calling OS thread until the
// result is ready.
//
// That's a real problem here because this server's io_context may be
// running on just one thread (see main.cpp's `net::io_context ioc{1}`).
// Every coroutine — every in-flight HTTP request — takes turns on that
// one thread. If any single coroutine calls fut.get() directly, it
// doesn't just block itself: it blocks that thread, which means EVERY
// other request being served concurrently freezes too, for as long as
// that future takes to become ready.
//
// The fix: instead of blocking on the future, repeatedly ask "are you
// ready yet?" (fut.wait_for(0ms), a non-blocking check) and if not,
// co_await a short asio timer before checking again. Each co_await on
// the timer SUSPENDS this coroutine and hands the thread back to the
// io_context, which is then free to run other coroutines (other
// requests, other timers, ...) during that 2ms gap. Once the future
// reports ready, fut.get() at the end is guaranteed to return
// immediately (no blocking) since we already confirmed it's ready.
//
// This is a polling loop, not a true "wake me up exactly when it's
// done" notification — it's a pragmatic MVP bridge, not the ideal
// long-term design. The proper fix is changing whatever produces the
// std::future (GpuFaissEngine, HealthChecker, ...) to instead complete
// via an asio completion token, so it posts its result directly onto
// the waiting coroutine's executor with zero polling delay. That's a
// bigger refactor, deliberately deferred — this bridge is what makes it
// safe to defer.
template <typename T>
boost::asio::awaitable<T> AwaitFuture(std::future<T> fut) {
  auto executor = co_await boost::asio::this_coro::executor;
  boost::asio::steady_timer timer(executor);
  while (fut.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    timer.expires_after(std::chrono::milliseconds(2));
    co_await timer.async_wait(boost::asio::use_awaitable);
  }
  co_return fut.get(); // re-throws here if the producer set an exception
}

} // namespace util
