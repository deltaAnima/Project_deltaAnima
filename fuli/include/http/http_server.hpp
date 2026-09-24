#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <functional>
#include <map>
#include <string>

namespace httpsrv {

// Handler receives the raw request body and returns the raw JSON response
// body. Route keys are "METHOD /path", e.g. "POST /character/context".
using RequestHandler =
    std::function<boost::asio::awaitable<std::string>(std::string body)>;

class HttpServer {
public:
  HttpServer(boost::asio::io_context &ioc, unsigned short port);

  void RegisterRoute(std::string method_and_path, RequestHandler handler);

  // Spawns the accept loop onto the io_context. Call ioc.run() afterwards.
  void Run();

private:
  boost::asio::awaitable<void> Listen();
  boost::asio::awaitable<void> Session(boost::asio::ip::tcp::socket socket);

  boost::asio::io_context &ioc_;
  unsigned short port_;
  std::map<std::string, RequestHandler> routes_;
};

} // namespace httpsrv
