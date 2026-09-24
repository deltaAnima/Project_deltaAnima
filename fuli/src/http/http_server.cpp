#include "http/http_server.hpp"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <iostream>

namespace httpsrv {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

HttpServer::HttpServer(net::io_context &ioc, unsigned short port)
    : ioc_(ioc), port_(port) {}

void HttpServer::RegisterRoute(std::string method_and_path,
                                RequestHandler handler) {
  routes_[std::move(method_and_path)] = std::move(handler);
}

void HttpServer::Run() {
  // net::co_spawn launches a coroutine onto an executor and returns
  // immediately — it does NOT run the coroutine synchronously here.
  // net::detached means "I don't want a handle to this coroutine and
  // don't care about its result/exceptions" — appropriate for Listen(),
  // which is meant to run forever until the process exits. (If Listen()
  // ever throws, detached means that exception is silently swallowed —
  // acceptable for an MVP, worth revisiting if the accept loop needs to
  // be more robust later.)
  net::co_spawn(ioc_, Listen(), net::detached);
}

// Runs forever: accept a connection, hand it off to its own Session
// coroutine (also detached, so Listen() doesn't wait for the connection
// to finish before accepting the next one), and go back to accepting.
// This is how one thread (or a small pool) can serve many concurrent
// connections — each co_await below suspends this coroutine without
// blocking the thread, which is free to run other coroutines meanwhile.
net::awaitable<void> HttpServer::Listen() {
  auto executor = co_await net::this_coro::executor;
  tcp::acceptor acceptor(executor, tcp::endpoint(tcp::v4(), port_));
  std::cout << "[HTTP] listening on :" << port_ << std::endl;

  for (;;) {
    tcp::socket socket = co_await acceptor.async_accept(net::use_awaitable);
    net::co_spawn(executor, Session(std::move(socket)), net::detached);
  }
}

// Handles ONE TCP connection, potentially multiple HTTP requests on it
// (HTTP/1.1 keep-alive) until the client disconnects or an error occurs.
net::awaitable<void> HttpServer::Session(tcp::socket socket) {
  beast::tcp_stream stream(std::move(socket));
  beast::flat_buffer buffer; // reused across requests on this connection

  try {
    for (;;) {
      http::request<http::string_body> req;
      co_await http::async_read(stream, buffer, req, net::use_awaitable);

      // Our routing key format: "POST /character/context". http::to_string
      // turns the parsed http::verb enum back into its string form so we
      // can concatenate it with the target path.
      std::string key =
          std::string(http::to_string(req.method())) + " " +
          std::string(req.target());

      http::response<http::string_body> res{http::status::not_found,
                                              req.version()};
      res.set(http::field::content_type, "application/json");
      res.keep_alive(req.keep_alive()); // honor whatever the client asked for

      auto it = routes_.find(key);
      if (it != routes_.end()) {
        try {
          // co_await the handler — this is where control leaves
          // http_server.cpp entirely and runs, e.g.,
          // Orchestrator::HandleContextRequest, which itself co_awaits
          // EmbeddingClient/Faiss/deltaEGO. Session() is suspended (not
          // blocked) for the whole duration.
          std::string body = co_await it->second(req.body());
          res.result(http::status::ok);
          res.body() = std::move(body);
        } catch (const std::exception &e) {
          // Turn any exception thrown by pipeline code (bad JSON, TEI
          // connection refused, Faiss dimension mismatch, etc.) into a
          // 500 with the error message, instead of letting it propagate
          // and kill this coroutine (and, worse, this whole connection's
          // handling) silently.
          res.result(http::status::internal_server_error);
          res.body() = std::string(R"({"error":")") + e.what() + "\"}";
        }
      } else {
        res.body() = R"({"error":"not found"})";
      }

      res.prepare_payload();
      co_await http::async_write(stream, res, net::use_awaitable);
      if (!res.keep_alive())
        break; // client asked us to close after this response
    }
  } catch (const boost::system::system_error &e) {
    // http::error::end_of_stream is the normal/expected way this loop
    // ends when the client closes the connection — don't log it as an
    // error. Anything else (a real network problem) gets logged.
    if (e.code() != http::error::end_of_stream) {
      std::cerr << "[HTTP] session error: " << e.what() << std::endl;
    }
  }

  beast::error_code ec;
  stream.socket().shutdown(tcp::socket::shutdown_send, ec); // best-effort;
                                                              // ignore errors,
                                                              // we're done
                                                              // either way
}

} // namespace httpsrv
