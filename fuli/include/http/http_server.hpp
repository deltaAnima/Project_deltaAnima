#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <functional>
#include <map>
#include <string>

namespace httpsrv {

// A handler receives the raw HTTP request body (a JSON string, in
// practice) and returns the raw JSON response body as a string. It's a
// coroutine itself (net::awaitable<std::string>) so it's free to
// co_await other async work — calling EmbeddingClient, Faiss, Redis,
// etc. — without blocking the server's io_context thread while it does.
//
// Route keys are "METHOD /path", e.g. "POST /character/context" — see
// RegisterRoute. This is a minimal exact-match router, not a real
// framework: no path parameters, wildcards, or query-string parsing.
// That's fine for a single-endpoint MVP; reach for something like
// Boost.URL or a small router library if the route count grows.
using RequestHandler =
    std::function<boost::asio::awaitable<std::string>(std::string body)>;

// A small hand-rolled async HTTP/1.1 server built directly on
// Boost.Beast + Boost.Asio coroutines. There's no framework here (no
// Drogon/oatpp/Crow) — just enough Beast plumbing to accept connections,
// read one HTTP request at a time, dispatch it to a registered handler,
// and write the response back, all without blocking OS threads on I/O.
class HttpServer {
public:
  HttpServer(boost::asio::io_context &ioc, unsigned short port);

  // Not thread-safe to call concurrently with Run()/incoming requests —
  // register every route up front, before Run().
  void RegisterRoute(std::string method_and_path, RequestHandler handler);

  // Spawns the accept loop as a detached coroutine onto ioc (see
  // net::co_spawn in the .cpp). Returns immediately — you still need to
  // call ioc.run() yourself afterwards (see main.cpp) to actually drive
  // any of this.
  void Run();

private:
  // Both of these are themselves coroutines (net::awaitable<void>), each
  // looping forever until the process exits or an unrecoverable error
  // occurs. See http_server.cpp for the accept-loop / one-connection-per-
  // Session structure.
  boost::asio::awaitable<void> Listen();
  boost::asio::awaitable<void> Session(boost::asio::ip::tcp::socket socket);

  boost::asio::io_context &ioc_;
  unsigned short port_;
  std::map<std::string, RequestHandler> routes_;
};

} // namespace httpsrv
