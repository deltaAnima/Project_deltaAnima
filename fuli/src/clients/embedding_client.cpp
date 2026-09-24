#include "clients/embedding_client.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <stdexcept>

#include "Third_Party/json.hpp"

namespace clients {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

EmbeddingClient::EmbeddingClient(net::io_context &ioc, std::string host,
                                  std::string port)
    : ioc_(ioc), host_(std::move(host)), port_(std::move(port)) {}

net::awaitable<std::vector<float>> EmbeddingClient::Embed(
    std::string text) const {
  auto executor = co_await net::this_coro::executor;

  tcp::resolver resolver(executor);
  beast::tcp_stream stream(executor);

  auto endpoints =
      co_await resolver.async_resolve(host_, port_, net::use_awaitable);
  co_await stream.async_connect(endpoints, net::use_awaitable);

  // normalize=true keeps L2 distance on the Faiss side equivalent to
  // cosine similarity, since GpuIndexFlatL2 doesn't do IP by default.
  json body{{"inputs", text}, {"normalize", true}};

  http::request<http::string_body> req{http::verb::post, "/embed", 11};
  req.set(http::field::host, host_);
  req.set(http::field::user_agent, "fuli-orchestrator");
  req.set(http::field::content_type, "application/json");
  req.body() = body.dump();
  req.prepare_payload();

  co_await http::async_write(stream, req, net::use_awaitable);

  beast::flat_buffer buffer;
  http::response<http::string_body> res;
  co_await http::async_read(stream, buffer, res, net::use_awaitable);

  beast::error_code ec;
  stream.socket().shutdown(tcp::socket::shutdown_both, ec);

  if (res.result() != http::status::ok) {
    throw std::runtime_error("embedding request failed: HTTP " +
                              std::to_string(res.result_int()) + " " +
                              res.body());
  }

  // TEI's native /embed returns [[f32, ...]] — one row per input string.
  json parsed = json::parse(res.body());
  if (!parsed.is_array() || parsed.empty()) {
    throw std::runtime_error("unexpected /embed response shape: " +
                              res.body());
  }
  co_return parsed.at(0).get<std::vector<float>>();
}

} // namespace clients
