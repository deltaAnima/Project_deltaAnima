#include "clients/embedding_client.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <stdexcept>

#include "Third_Party/json.hpp"

namespace clients {

namespace beast = boost::beast; // low-level async stream helpers (TCP, buffers)
namespace http = beast::http;   // HTTP/1.1 message building/parsing on top of beast
namespace net = boost::asio;    // the actual async I/O + coroutine machinery
using tcp = net::ip::tcp;
using json = nlohmann::json;

EmbeddingClient::EmbeddingClient(net::io_context &ioc, std::string host,
                                  std::string port)
    : ioc_(ioc), host_(std::move(host)), port_(std::move(port)) {}

// net::awaitable<T> is C++20 coroutine sugar: this function's body can use
// co_await to suspend at each async step (resolve, connect, write, read)
// without blocking the OS thread. The `co_await net::this_coro::executor`
// line below just recovers "which io_context/strand am I currently
// running on", which every asio async_* call needs to know where to post
// its completion handler.
net::awaitable<std::vector<float>> EmbeddingClient::Embed(
    std::string text) const {
  auto executor = co_await net::this_coro::executor;

  // resolver: turns "127.0.0.1"/"8080" into actual connectable endpoints
  // (mostly a no-op for an IP literal like this, but would do real DNS
  // lookup for a hostname).
  tcp::resolver resolver(executor);
  // tcp_stream: beast's wrapper around a plain asio TCP socket, adding
  // timeout support and the async_* overloads beast::http functions need.
  beast::tcp_stream stream(executor);

  auto endpoints =
      co_await resolver.async_resolve(this->host_, this->port_, net::use_awaitable);
  co_await stream.async_connect(endpoints, net::use_awaitable);

  // TEI's /embed accepts {"inputs": "<text>"} for a single string (it
  // also accepts an array of strings for batching, unused here).
  // normalize=true asks TEI to L2-normalize the embedding server-side —
  // we want that because GpuFaissEngine uses plain L2 distance
  // (GpuIndexFlatL2), and L2 distance between normalized vectors ranks
  // results the same way cosine similarity would.
  json body{{"inputs", text}, {"normalize", true}};

  http::request<http::string_body> req{http::verb::post, "/embed", 11};
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
    throw std::runtime_error("embedding request failed: HTTP " +
                              std::to_string(res.result_int()) + " " +
                              res.body());
  }

  // TEI's native /embed returns a JSON array of embeddings, one row per
  // input string: [[0.01, -0.02, ...]]. Since we always send exactly one
  // input, we only ever need row 0.
  json parsed = json::parse(res.body());
  if (!parsed.is_array() || parsed.empty()) 
  {
    throw std::runtime_error("unexpected /embed response shape: " +
                              res.body());
  }
  co_return parsed.at(0).get<std::vector<float>>();
}

nlohmann::json EmbeddingClient::HealthCheck() const 
{
  if(this->port_.empty() || this->host_.empty())
  {
    return nlohmann::json{
      {"status", "error"},
      {"message", "Class EmbeddingClient -> host or port is empty"}
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
        {"message", "Class EmbeddingClient -> Failed to connect: " + connect_ec.message()}
      };
    }

    // If we reach here, the connection was successful
    boost::system::error_code ignored_ec;
    socket.shutdown(tcp::socket::shutdown_both, ignored_ec);
    socket.close(ignored_ec);

    return nlohmann::json{
      {"status", "healthy"},
      {"message", "Class EmbeddingClient -> Can connect to host and port.\n"}
    };
  }
  catch (const std::exception& e)
  {
    return nlohmann::json{
      {"status", "error"},
      {"message", "Class EmbeddingClient -> Exception during health check: " + std::string(e.what())}
    };
  }
}

} // namespace clients
