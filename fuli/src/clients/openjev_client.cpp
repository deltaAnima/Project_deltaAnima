#include "clients/openjev_client.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "Third_Party/json.hpp"

namespace clients {

namespace beast = boost::beast; // low-level async stream helpers (TCP, buffers)
namespace http = beast::http;   // HTTP/1.1 message building/parsing on top of beast
namespace net = boost::asio;    // the actual async I/O + coroutine machinery
using tcp = net::ip::tcp;
using json = nlohmann::json;

namespace {

// OpenJev returns raw logits under "embedding" in the order
// [contradiction, entailment, neutral]; normalize with a numerically
// stable softmax (subtract max before exponentiating).
OpenJevScores Softmax(const std::vector<double> &logits) 
{
  if (logits.size() != 3) 
  {
    throw std::runtime_error(
        "OpenJevClient: expected 3 logits, got " +
        std::to_string(logits.size()));
  }

  double max_logit = *std::max_element(logits.begin(), logits.end());

  double exps[3];
  double sum = 0.0;
  for (int i = 0; i < 3; ++i) 
  {
    exps[i] = std::exp(logits[i] - max_logit);
    sum += exps[i];
  }

  OpenJevScores scores;
  scores.contradiction = exps[0] / sum;
  scores.entailment = exps[1] / sum;
  scores.neutral = exps[2] / sum;
  return scores;
}

} // namespace

OpenJevClient::OpenJevClient(net::io_context &ioc, std::string host,
                              std::string port)
    : ioc_(ioc), host_(std::move(host)), port_(std::move(port)) {}

// net::awaitable<T> is C++20 coroutine sugar: this function's body can use
// co_await to suspend at each async step (resolve, connect, write, read)
// without blocking the OS thread. The `co_await net::this_coro::executor`
// line below just recovers "which io_context/strand am I currently
// running on", which every asio async_* call needs to know where to post
// its completion handler.
net::awaitable<OpenJevScores> OpenJevClient::Classify(
    std::string premise, std::string hypothesis) const {
  auto executor = co_await net::this_coro::executor;

  // resolver: turns host/port into actual connectable endpoints (mostly a
  // no-op for an IP literal, but would do real DNS lookup for a hostname).
  tcp::resolver resolver(executor);
  // tcp_stream: beast's wrapper around a plain asio TCP socket, adding
  // timeout support and the async_* overloads beast::http functions need.
  beast::tcp_stream stream(executor);

  auto endpoints =
      co_await resolver.async_resolve(this->host_, this->port_, net::use_awaitable);
  co_await stream.async_connect(endpoints, net::use_awaitable);

  // OpenJev's /classify expects {"text": "Premise: <p>\nHypothesis: <h>"}
  // and responds with raw logits under "embedding" (same field name as
  // the TEI /embed route, despite this being a classification result).
  std::string text =
      "Premise: " + premise + "\nHypothesis: " + hypothesis;
  json body{{"text", text}};

  http::request<http::string_body> req{http::verb::put, "/classify", 11};
  req.set(http::field::host, this->host_);
  req.set(http::field::user_agent, "fuli-orchestrator");
  req.set(http::field::content_type, "application/json");
  req.body() = body.dump();
  req.prepare_payload(); // fills in Content-Length from body().size()

  co_await http::async_write(stream, req, net::use_awaitable);

  beast::flat_buffer buffer; // scratch space async_read grows as needed
  http::response<http::string_body> res;
  co_await http::async_read(stream, buffer, res, net::use_awaitable);

  // We only ever do one request per connection (no keep-alive pooling
  // yet), so shut the write side down once we have our answer. Errors
  // here are expected/harmless (e.g. peer already closed) — hence the
  // error_code overload instead of the throwing one.
  beast::error_code error_code;
  stream.socket().shutdown(tcp::socket::shutdown_both, error_code);

  if (res.result() != http::status::ok) 
  {
    throw std::runtime_error("classify request failed: HTTP " +
                              std::to_string(res.result_int()) + " " +
                              res.body());
  }

  json parsed = json::parse(res.body());
  if (!parsed.contains("embedding") || !parsed.at("embedding").is_array()) 
  {
    throw std::runtime_error("unexpected /classify response shape: " +
                              res.body());
  }

  co_return Softmax(parsed.at("embedding").get<std::vector<double>>());
}

nlohmann::json OpenJevClient::HealthCheck() const 
{
  if(this->port_.empty() || this->host_.empty())
  {
    return nlohmann::json{
      {"status", "error"},
      {"message", "Class OpenJevClient -> host or port is empty"}
    };
  }

  namespace net = boost::asio;
  using tcp = net::ip::tcp;

  try
  {
    net::io_context io_context;
    tcp::resolver resolver(io_context);
    tcp::socket socket(io_context);

    auto endpoints = resolver.resolve(this->host_, this->port_);

    net::steady_timer timer(io_context);
    timer.expires_after(std::chrono::seconds(5)); // Set a timeout of 5 seconds

    boost::system::error_code connect_ec = net::error::would_block;

    //async connect try
    net::async_connect(socket, endpoints,
        [&connect_ec](const boost::system::error_code& ec, const tcp::endpoint&) {
        connect_ec = ec;
      });

    // close the socket if the timer expires
    timer.async_wait([&socket](const boost::system::error_code& ec) {
      if (!ec) {
        boost::system::error_code ignored_ec;
        socket.close(ignored_ec);
      }
    });

    // execute the event loop until the connect operation completes or the timer expires
    io_context.run();

    // check if the connection was successful
    if (connect_ec) 
    {
      return nlohmann::json{
        {"status", "unhealthy"},
        {"message", "Class OpenJevClient -> Failed to connect: " + connect_ec.message()}
      };
    }

    // If we reach here, the connection was successful
    boost::system::error_code ignored_ec;
    socket.shutdown(tcp::socket::shutdown_both, ignored_ec);
    socket.close(ignored_ec);

    return nlohmann::json{
      {"status", "healthy"},
      {"message", "Class OpenJevClient -> Can connect to host and port.\n"}
    };
  }
  catch (const std::exception& e)
  {
    return nlohmann::json{
      {"status", "error"},
      {"message", "Class OpenJevClient -> Exception during health check: " + std::string(e.what())}
    };
  }
}

} // namespace clients
