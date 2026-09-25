#include "clients/llama_5090_client.hpp"

#include <algorithm>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <chrono>
#include <stdexcept>

namespace clients {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

namespace {

// Forces the model's reply to be exactly {"v": <num>, "a": <num>,
// "d": <num>} instead of free-form chat text — see
// Llama5090Client::EstimateVad's doc comment on why this matters more
// here than it would for, say, EmbeddingClient (there's no equivalent
// "the server just returns a vector" guarantee for a chat model).
json VadJsonSchema() {
  return json{
      {"type", "object"},
      {"properties",
       {
           {"v", {{"type", "number"}, {"minimum", -1}, {"maximum", 1}}},
           {"a", {{"type", "number"}, {"minimum", -1}, {"maximum", 1}}},
           {"d", {{"type", "number"}, {"minimum", -1}, {"maximum", 1}}},
       }},
      {"required", {"v", "a", "d"}},
  };
}

constexpr const char *kSystemPrompt =
    "You are a precise Valence-Arousal-Dominance (VAD) emotion analyzer. "
    "Given a piece of text, respond with ONLY a JSON object with three "
    "fields: v (valence: negative=-1 to positive=1), a (arousal: "
    "calm=-1 to excited=1), d (dominance: submissive=-1 to in-control=1). "
    "No other text.";

} // namespace

Llama5090Client::Llama5090Client(net::io_context &ioc, std::string host,
                                  std::string port)
    : ioc_(ioc), host_(std::move(host)), port_(std::move(port)) {}

net::awaitable<VadEstimate> Llama5090Client::EstimateVad(std::string text) const {
  auto executor = co_await net::this_coro::executor;

  tcp::resolver resolver(executor);
  beast::tcp_stream stream(executor);

  auto endpoints = co_await resolver.async_resolve(host_, port_, net::use_awaitable);
  co_await stream.async_connect(endpoints, net::use_awaitable);

  json body{
      {"messages",
       {
           {{"role", "system"}, {"content", kSystemPrompt}},
           {{"role", "user"}, {"content", text}},
       }},
      {"temperature", 0.0},
      {"response_format", {{"type", "json_object"}, {"schema", VadJsonSchema()}}},
  };

  http::request<http::string_body> req{http::verb::post, "/v1/chat/completions", 11};
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
    throw std::runtime_error("5090 chat completion failed: HTTP " +
                              std::to_string(res.result_int()) + " " + res.body());
  }

  // Standard OpenAI-compatible shape: choices[0].message.content is a
  // STRING containing the model's reply — here, that string is itself
  // the VAD JSON, so it needs a second json::parse.
  json outer = json::parse(res.body());
  std::string content = outer.at("choices").at(0).at("message").at("content").get<std::string>();
  json vad_json = json::parse(content);

  VadEstimate estimate;
  estimate.v = std::clamp(vad_json.at("v").get<float>(), -1.0f, 1.0f);
  estimate.a = std::clamp(vad_json.at("a").get<float>(), -1.0f, 1.0f);
  estimate.d = std::clamp(vad_json.at("d").get<float>(), -1.0f, 1.0f);
  co_return estimate;
}

nlohmann::json Llama5090Client::HealthCheck() const {
  if (host_.empty() || port_.empty()) {
    return {{"status", "error"},
            {"message", "Class Llama5090Client -> host or port is empty"}};
  }

  try {
    // Fully synchronous here (not the coroutine path EstimateVad uses)
    // — Beast's tcp_stream supports a plain expires_after() timeout on
    // synchronous calls directly, so this doesn't need the manual
    // timer+cancel dance RedisDbClient::HealthCheck needed for
    // boost::redis's async-only API.
    net::io_context local_ioc;
    tcp::resolver resolver(local_ioc);
    beast::tcp_stream stream(local_ioc);
    stream.expires_after(std::chrono::seconds(5));

    auto endpoints = resolver.resolve(host_, port_);
    stream.connect(endpoints);

    http::request<http::empty_body> req{http::verb::get, "/health", 11};
    req.set(http::field::host, host_);
    http::write(stream, req);

    beast::flat_buffer buffer;
    http::response<http::string_body> res;
    http::read(stream, buffer, res);

    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_both, ec);

    // llama.cpp's /health: 200 {"status":"ok"} when ready, 503 while
    // the model is still loading — both are "the server is there and
    // talking HTTP", only the first counts as fully healthy.
    if (res.result() == http::status::ok) {
      return {{"status", "healthy"},
              {"message", "Class Llama5090Client -> GET /health: " + res.body()}};
    }
    return {{"status", "unhealthy"},
            {"message", "Class Llama5090Client -> GET /health returned HTTP " +
                            std::to_string(res.result_int()) + ": " + res.body()}};
  } catch (const std::exception &e) {
    return {{"status", "unhealthy"},
            {"message", std::string("Class Llama5090Client -> ") + e.what()}};
  }
}

} // namespace clients
