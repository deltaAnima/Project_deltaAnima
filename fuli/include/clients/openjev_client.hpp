#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <string>
#include "Third_Party/json.hpp"

namespace clients {

// Softmax-normalized NLI scores returned by the OpenJev /classify route.
// Order matches the reference `jev()` shell helper: logits[0]=contradiction,
// logits[1]=entailment, logits[2]=neutral.
struct OpenJevScores 
{
  double contradiction = 0.0;
  double entailment = 0.0;
  double neutral = 0.0;
};

// Thin async HTTP client for the OpenJev NLI classification service.
// Talks to OpenJev's `/classify` route: PUT {"text": "Premise: ...\n
// Hypothesis: ..."} -> {"embedding": [logit_contradiction, logit_entailment,
// logit_neutral], ...}. The raw logits are softmax-normalized client-side
// before being handed back.
//
// This class returns boost::asio::awaitable<T>, meaning callers must
// `co_await client.Classify(...)` from inside another coroutine (a function
// itself returning net::awaitable<...>). It does NOT block the calling
// thread — while waiting on the network, the io_context is free to run
// other coroutines (other in-flight HTTP requests, etc.) on the same
// thread. This is the same cooperative-multitasking model asio-grpc uses
// elsewhere in this project.
class OpenJevClient 
{
public:
  // ioc must outlive this object — we don't own it, we just borrow its
  // executor for every connection we open.
  OpenJevClient(boost::asio::io_context &ioc, std::string host,
                std::string port);

  // Opens a new TCP connection, does one PUT /classify, and returns
  // softmax-normalized [contradiction, entailment, neutral] scores for
  // the given premise/hypothesis pair. A new connection per call is
  // wasteful (no keep-alive/pooling) but simple to reason about — a
  // connection pool is a reasonable next optimization once this is a
  // bottleneck, not before.
  boost::asio::awaitable<OpenJevScores> Classify(
      std::string premise, std::string hypothesis) const;

  nlohmann::json HealthCheck() const;

private:
  boost::asio::io_context &ioc_; // currently unused directly (the executor
                                  // comes from co_await this_coro::executor
                                  // inside Classify()), kept for future pooling
  std::string host_;
  std::string port_;
};

} // namespace clients
