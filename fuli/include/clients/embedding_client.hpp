#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <string>
#include <vector>

namespace clients {

// Thin async HTTP client for the HuggingFace text-embeddings-inference
// (TEI) container running bge-m3. Talks to TEI's *native* `/embed` route
// (not the OpenAI-compatible `/v1/embeddings` one) — the request/response
// shape is simpler: POST {"inputs": "text"} -> [[f32, f32, ...]].
//
// This class returns boost::asio::awaitable<T>, meaning callers must
// `co_await client.Embed(...)` from inside another coroutine (a function
// itself returning net::awaitable<...>). It does NOT block the calling
// thread — while waiting on the network, the io_context is free to run
// other coroutines (other in-flight HTTP requests, etc.) on the same
// thread. This is the same cooperative-multitasking model asio-grpc uses
// elsewhere in this project.
class EmbeddingClient {
public:
  // ioc must outlive this object — we don't own it, we just borrow its
  // executor for every connection we open.
  EmbeddingClient(boost::asio::io_context &ioc, std::string host,
                   std::string port);

  // Opens a new TCP connection, does one POST /embed, and returns the
  // resulting embedding vector for `text`. A new connection per call is
  // wasteful (no keep-alive/pooling) but simple to reason about — a
  // connection pool is a reasonable next optimization once this is a
  // bottleneck, not before.
  boost::asio::awaitable<std::vector<float>> Embed(std::string text) const;

private:
  boost::asio::io_context &ioc_; // currently unused directly (the executor
                                  // comes from co_await this_coro::executor
                                  // inside Embed()), kept for future pooling
  std::string host_;
  std::string port_;
};

} // namespace clients
