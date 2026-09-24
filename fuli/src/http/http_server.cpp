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
  net::co_spawn(ioc_, Listen(), net::detached);
}

net::awaitable<void> HttpServer::Listen() {
  auto executor = co_await net::this_coro::executor;
  tcp::acceptor acceptor(executor, tcp::endpoint(tcp::v4(), port_));
  std::cout << "[HTTP] listening on :" << port_ << std::endl;

  for (;;) {
    tcp::socket socket = co_await acceptor.async_accept(net::use_awaitable);
    net::co_spawn(executor, Session(std::move(socket)), net::detached);
  }
}

net::awaitable<void> HttpServer::Session(tcp::socket socket) {
  beast::tcp_stream stream(std::move(socket));
  beast::flat_buffer buffer;

  try {
    for (;;) {
      http::request<http::string_body> req;
      co_await http::async_read(stream, buffer, req, net::use_awaitable);

      std::string key =
          std::string(http::to_string(req.method())) + " " +
          std::string(req.target());

      http::response<http::string_body> res{http::status::not_found,
                                              req.version()};
      res.set(http::field::content_type, "application/json");
      res.keep_alive(req.keep_alive());

      auto it = routes_.find(key);
      if (it != routes_.end()) {
        try {
          std::string body = co_await it->second(req.body());
          res.result(http::status::ok);
          res.body() = std::move(body);
        } catch (const std::exception &e) {
          res.result(http::status::internal_server_error);
          res.body() = std::string(R"({"error":")") + e.what() + "\"}";
        }
      } else {
        res.body() = R"({"error":"not found"})";
      }

      res.prepare_payload();
      co_await http::async_write(stream, res, net::use_awaitable);
      if (!res.keep_alive())
        break;
    }
  } catch (const boost::system::system_error &e) {
    if (e.code() != http::error::end_of_stream) {
      std::cerr << "[HTTP] session error: " << e.what() << std::endl;
    }
  }

  beast::error_code ec;
  stream.socket().shutdown(tcp::socket::shutdown_send, ec);
}

} // namespace httpsrv
